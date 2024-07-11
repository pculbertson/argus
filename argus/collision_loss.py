from pathlib import Path

import numpy as np
import pypose as pp
import torch
import warp as wp
import warp.sim

from argus import ROOT

# A dummy configuration to pad the batched input to the correct size.
# Puts the cube somewhere insane ([100, 100, 100]) where the collision checks should be fine. 
DUMMY_CONFIG = torch.cat([torch.zeros(16), torch.tensor([100., 100., 100., 0., 0., 0., 1.])])


def get_warp_model(path: str | Path, cube_size: float = 0.035, batch_size: int = 1) -> wp.sim.Model:
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
            # batch_idxs[contact_id] = env_idx
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
        
        # pad the input to the correct size
        ctx.B, dof = q_batched.shape
        assert ctx.B <= model.num_envs, f"Batch size {ctx.B} is greater than model.num_envs {model.num_envs}"

        if ctx.B < model.num_envs:
            # Pad q_batched dummy poses. 
            q_batched = torch.cat([q_batched, DUMMY_CONFIG.to(q_batched.device).repeat(model.num_envs - ctx.B, 1)], dim=0)

        # Finish all torch business before starting warp.
        wp.synchronize_device()

        # getting the warp tensor input from torch
        q = wp.from_torch(q_batched.reshape(-1), requires_grad=True)
        assert q.shape == model.joint_q.shape
        assert q.requires_grad

        # setting up the context
        ctx.q_batched_shape = q_batched.shape
        ctx.tape = wp.Tape()
        ctx.q = q
        ctx.model = model

        # setup for forward pass
        ctx.state = ctx.model.state()
        _qd = wp.zeros(ctx.model.joint_dof_count, device="cuda", requires_grad=False)  # dummy values

        # setting up the outputs
        ctx.sdf_vals = wp.zeros(ctx.model.rigid_contact_max, dtype=wp.float32, device="cuda", requires_grad=True)

        # forward pass
        with ctx.tape:
            wp.sim.eval_fk(ctx.model, ctx.q, _qd, None, ctx.state)  # update model state with q
            wp.sim.collide(ctx.model, ctx.state)  # compute collisions

            # allocating inputs
            rigid_contact_count = ctx.model.rigid_contact_count.numpy()[0].item()  # total number of contacts
            if rigid_contact_count > 0:

                # allocating outputs (unused here but needed for kernel)
                batch_idxs = wp.from_numpy(-np.ones(ctx.model.rigid_contact_max), dtype=int, device="cuda")  # outputs
                contact_points_cube = wp.zeros(ctx.model.rigid_contact_max, dtype=wp.vec3, device="cuda")  # outputs
                contact_points_other = wp.zeros(ctx.model.rigid_contact_max, dtype=wp.vec3, device="cuda")

                wp.launch(
                    kernel=get_cube_contact_points_and_sdf_vals,
                    dim=rigid_contact_count,
                    inputs=[
                        ctx.model.rigid_contact_shape0,
                        ctx.model.rigid_contact_shape1,
                        ctx.model.shape_body,
                        ctx.model.cube_idxs,
                        ctx.model.rigid_contact_point0,
                        ctx.model.rigid_contact_point1,
                        ctx.state.body_q,
                        ctx.model.rigid_contact_normal,
                        ctx.model.num_envs,
                    ],
                    outputs=[batch_idxs, contact_points_cube, contact_points_other, ctx.sdf_vals],
                )

            # ensure Warp operations complete before returning data to Torch
            wp.synchronize_device()

            sdf_outputs = wp.to_torch(ctx.sdf_vals)

            return sdf_outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """The backward pass.

        Args:
            ctx: The context.
            grad_output: The gradient of the loss.
        """
        # ensure Torch operations complete before running Warp
        wp.synchronize_device()

        ctx.tape.zero()
        ctx.sdf_vals.grad = wp.from_torch(grad_output, dtype=wp.float32).contiguous()
        ctx.tape.backward()

        # ensure Warp operations complete before returning data to Torch
        wp.synchronize_device()

        return wp.to_torch(ctx.tape.gradients[ctx.q]).reshape(ctx.q_batched_shape)[:ctx.B], None


def collision_loss_function(qrobot_batch: torch.Tensor, qcube_log_batch: torch.Tensor, warp_model: wp.sim.Model) -> torch.Tensor:
    """The loss function to minimize.

    Args:
        qrobot_batch: The batched joint states of the robot. Shape=(batch_size, n_r).
        qcube_batch: The batched joint states of the cube. Shape=(batch_size, 7).
        model: The batched model.

    Returns:
        The loss over all batches. Shape=(,).
    """
    qcube_batch = pp.se3(qcube_log_batch).Exp()
    compute_signed_distances = ComputeSignedDistances.apply
    q0_batch = torch.cat([qrobot_batch, qcube_batch], dim=-1)  # (batch_size, 8)
    sdf_vals = compute_signed_distances(q0_batch, warp_model)
    relu_vals = torch.relu(-sdf_vals)  # 0 loss if signed distance is positive
    return torch.sum(relu_vals)  # mean over all collision pairs

if __name__ == "__main__":
    # [DEBUG] pypose stuff
    ###########################################################
    # import pypose as pp
    # from torch.optim import Adam
    # asdf = pp.randn_SE3(requires_grad=True)
    # rand_target = pp.randn_SE3()
    # optimizer = Adam([asdf], lr=1e-3)

    # losses = []
    # for _ in range(100):
    #     loss = torch.sum((asdf @ rand_target.Inv()).Log() ** 2)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     losses.append(loss.item())
    #     print(f"loss: {loss}")
    ###########################################################

    import pypose as pp
    import warp.sim.render
    from torch.optim import SGD

    # setup: copied from 1_two_link_collider.py
    cube_size = 0.035
    batch_size = 32
    path = f"{ROOT}/scripts/warp/dummy_with_mesh.urdf"
    model = get_warp_model(path, cube_size=0.035, batch_size=batch_size)

    # making robot and cube states - proof of concept for loss decrease with pypose
    qr_batch = torch.tensor([0.0], device="cuda").repeat(batch_size, 1) 

    # Create a "vanilla" PyTorch parameter for the log of cube pose.
    # Note complicated init. is just to create a meaningful starting point.
    qc_log_batch = torch.nn.Parameter(
        pp.SE3(torch.tensor(
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )).Log().tensor().repeat(batch_size, 1),
        requires_grad=True,
    )
    optimizer = SGD([qc_log_batch], lr=1e-1)

    # computing the loss and its gradient
    for _ in range(10):
        loss = collision_loss_function(qr_batch, qc_log_batch, model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item(), qc_log_batch.grad.norm())