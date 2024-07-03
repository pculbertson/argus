from pathlib import Path

import numpy as np
import torch
import warp as wp
import warp.sim

from argus import ROOT

# ######### #
# UTILITIES #
# ######### #


def get_model(path: str | Path, cube_size: float = 0.035, batch_size: int = 1) -> wp.sim.Model:
    """Creates a batched model from the supplied path."""
    # single robot
    robot_builder = wp.sim.ModelBuilder()
    wp.sim.parse_urdf(
        path,
        robot_builder,
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        density=1000,  # these settings taken from the quadruped example - critical!
        armature=0.01,
        stiffness=200,
        damping=1,
        contact_ke=1.0e4,
        contact_kd=1.0e2,
        contact_kf=1.0e2,
        contact_mu=1.0,
        limit_ke=1.0e4,
        limit_kd=1.0e1,
    )

    # add cube
    cube_idx_in_model = robot_builder.add_body()
    robot_builder.add_joint_free(cube_idx_in_model)
    robot_builder.add_shape_box(cube_idx_in_model, hx=cube_size, hy=cube_size, hz=cube_size)

    # create batched robot
    builder = wp.sim.ModelBuilder()
    for _ in range(batch_size):
        builder.add_builder(
            robot_builder,
            xform=wp.transform(np.zeros(3), wp.quat_identity()),  # [NOTE] for some reason, this really matters!!!
        )

    model = builder.finalize(requires_grad=True)
    model.ground = False
    bodies_per_env = model.body_count // batch_size
    cube_idxs = np.concatenate(
        [[cube_idx_in_model + i * bodies_per_env] * bodies_per_env for i in range(batch_size)]
    )  # (model.body_count,), for each body in the model, this is the corresponding cube index
    model.cube_idxs = wp.array(cube_idxs, dtype=int)
    return model


@wp.kernel
def get_cube_contact_points_and_sdf_vals(
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    shape_body: wp.array(dtype=int),
    cube_idxs: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    batch_size: int,
    # outputs
    batch_idxs: wp.array(dtype=int),
    contact_points_cube: wp.array(dtype=wp.vec3),
    contact_points_other: wp.array(dtype=wp.vec3),
    sdf_vals: wp.array(dtype=float),
):
    """See docstring in 1_two_link_collider.py."""
    contact_id = wp.tid()

    # step 1
    shape_idx_0 = rigid_contact_shape0[contact_id]
    shape_idx_1 = rigid_contact_shape1[contact_id]
    if shape_idx_0 == shape_idx_1:
        return  # ignore self-collisions

    # step 2
    body_idx_0 = shape_body[shape_idx_0]
    body_idx_1 = shape_body[shape_idx_1]

    cube_idx_0 = cube_idxs[body_idx_0]
    cube_idx_1 = cube_idxs[body_idx_1]
    if cube_idx_0 != body_idx_0 and cube_idx_1 != body_idx_1:
        return  # ignore collisions that don't involve the cube

    if cube_idx_0 == body_idx_0:
        is_cube_body0 = True
    else:
        is_cube_body0 = False

    # step 3
    P0_b0 = rigid_contact_point0[contact_id]  # first contact point in body 0's frame
    P1_b1 = rigid_contact_point1[contact_id]  # second contact point in body 1's frame
    X_Wb0 = body_q[body_idx_0]  # pose of body 0 in world frame
    X_Wb1 = body_q[body_idx_1]  # pose of body 1 in world frame

    # step 4
    P0_W = wp.transform_point(X_Wb0, P0_b0)  # first contact point in world frame
    P1_W = wp.transform_point(X_Wb1, P1_b1)  # second contact point in world frame

    # step 5
    n_W = rigid_contact_normal[contact_id]  # contact normal in world frame from point 2 to point 1
    sdf_value = wp.dot(n_W, P0_W - P1_W)  # signed distance between the contact points

    # computing which batch index this contact belongs to
    for env_idx in range(batch_size):
        cube_idx = cube_idxs[env_idx * batch_size]
        if body_idx_0 == cube_idx or body_idx_1 == cube_idx:
            batch_idxs[contact_id] = env_idx
            break

    if is_cube_body0:
        contact_points_cube[contact_id] = P0_W
        contact_points_other[contact_id] = P1_W
    else:
        contact_points_cube[contact_id] = P1_W
        contact_points_other[contact_id] = P0_W

    # setting the other outputs
    sdf_vals[contact_id] = sdf_value


class ComputeSignedDistances(torch.autograd.Function):
    """A custom autograd function to compute the SDF loss."""

    @staticmethod
    def forward(ctx, q_batched: torch.Tensor, model: wp.sim.Model) -> torch.Tensor:
        """The forward pass.

        Args:
            ctx: The context.
            q_batched: The batched joint states of the model. q_batched.reshape(-1)=(model.joint_dof_count,).
            model: The batched model.
        """
        wp.synchronize_device()

        # getting the warp tensor input from torch
        q = wp.from_torch(q_batched.reshape(-1))
        assert q.shape == model.joint_q.shape
        assert q.requires_grad

        # setting up the context
        ctx.q_batched_shape = q_batched.shape
        ctx.tape = wp.Tape()
        ctx.q = q
        ctx.model = model

        # setup for forward pass
        state = model.state()
        _qd = wp.zeros(model.joint_dof_count, device="cuda", requires_grad=False)  # dummy val

        # forward pass
        with ctx.tape:
            wp.sim.eval_fk(model, q, _qd, None, state)  # update model state with q
            wp.sim.collide(model, state)  # compute collisions

            # allocating inputs
            rigid_contact_count = model.rigid_contact_count.numpy()[0].item()  # total number of contacts
            rigid_contact_shape0 = model.rigid_contact_shape0[:rigid_contact_count]
            rigid_contact_shape1 = model.rigid_contact_shape1[:rigid_contact_count]
            rigid_contact_point0 = model.rigid_contact_point0[:rigid_contact_count]
            rigid_contact_point1 = model.rigid_contact_point1[:rigid_contact_count]
            rigid_contact_normal = model.rigid_contact_normal[:rigid_contact_count]

            # allocating outputs
            batch_idxs = wp.from_numpy(-np.ones(model.rigid_contact_max), dtype=int, device="cuda")  # outputs
            contact_points_cube = wp.zeros(model.rigid_contact_max, dtype=wp.vec3, device="cuda")  # outputs
            contact_points_other = wp.zeros(model.rigid_contact_max, dtype=wp.vec3, device="cuda")
            sdf_vals = wp.zeros(model.rigid_contact_max, dtype=wp.float32, device="cuda")

            wp.launch(
                kernel=get_cube_contact_points_and_sdf_vals,
                dim=rigid_contact_count,
                inputs=[
                    rigid_contact_shape0,
                    rigid_contact_shape1,
                    model.shape_body,
                    model.cube_idxs,
                    rigid_contact_point0,
                    rigid_contact_point1,
                    state.body_q,
                    rigid_contact_normal,
                    model.num_envs,
                ],
                outputs=[batch_idxs, contact_points_cube, contact_points_other, sdf_vals],
            )

        ctx.sdf_vals = sdf_vals
        wp.synchronize_device()
        return wp.to_torch(ctx.sdf_vals)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """The backward pass.

        Args:
            ctx: The context.
            grad_output: The gradient of the loss.
        """
        wp.synchronize_device()
        ctx.sdf_vals.grad = wp.from_torch(grad_output, dtype=wp.float32).contiguous()
        ctx.tape.backward()
        wp.synchronize_device()
        return wp.to_torch(ctx.tape.gradients[ctx.q]).reshape(ctx.q_batched_shape), None


def loss_function(q0_batch: torch.Tensor, model: wp.sim.Model) -> torch.Tensor:
    """The loss function to minimize.

    Args:
        q0_batch: The batched joint states of the model. q0_batch.shape=(batch_size, model.joint_dof_count).
        model: The batched model.

    Returns:
        The loss over all batches. Shape=(,).
    """
    compute_signed_distances = ComputeSignedDistances.apply
    sdf_vals = compute_signed_distances(q0_batch, model)
    relu_vals = torch.relu(-sdf_vals)  # 0 loss if signed distance is positive
    return torch.sum(relu_vals)  # sum over all batches


if __name__ == "__main__":
    # setup: copied from 1_two_link_collider.py
    cube_size = 0.035
    batch_size = 2
    model = get_model(f"{ROOT}/scripts/warp/dummy_with_mesh.urdf", cube_size=0.035, batch_size=batch_size)
    q0 = np.zeros(8)
    q0[1] = 0.5
    q0[7] = 1.0
    q0_batch_np = np.stack([q0] * batch_size)
    q0_batch = torch.tensor(q0_batch_np, device="cuda", dtype=torch.float32, requires_grad=True)  # (2, 8)

    # computing the loss and its gradient
    loss = loss_function(q0_batch, model)
    loss.backward()
    print(loss)
    print(q0_batch.grad)
    breakpoint()
