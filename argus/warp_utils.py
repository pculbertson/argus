import warp as wp
import warp.sim
from warp.sim.collide import box_sdf, box_sdf_grad
from argus import ROOT
from pathlib import Path
from typing import Optional
import numpy as np
import torch

wp.init()
# wp.config.verify_cuda = True


def get_leap_model(mujoco_path: Optional[Path] = None, batch_dim: int = 1):
    if mujoco_path is None:
        # Use default model path.
        leap_mjcf_model_path = Path(ROOT) / "mujoco" / "leap" / "leap_hand.xml"
        cube_model_path = (
            Path(ROOT) / "mujoco" / "common_assets" / "reorientation_cube.xml"
        )
    else:
        leap_mjcf_model_path = mujoco_path / "leap" / "leap_hand.xml"
        cube_model_path = mujoco_path / "common_assets" / "reorientation_cube.xml"

    # Create model builder.
    leap_builder = wp.sim.ModelBuilder()

    # Parse cube MJCF into Warp.
    wp.sim.parse_mjcf(
        str(cube_model_path),
        leap_builder,
        xform=None,
        enable_self_collisions=False,
        # collapse_fixed_joints=True,
    )

    # Parse Leap MJCF into Warp.
    wp.sim.parse_mjcf(
        str(leap_mjcf_model_path),
        leap_builder,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
    )

    # Add num_envs builders to final model.
    builder = wp.sim.ModelBuilder()
    for ii in range(batch_dim):
        builder.add_builder(leap_builder)

    # Build the model.
    model = builder.finalize(requires_grad=True)
    model.ground = False

    # Limit mesh collision pairs.
    # model.rigid_mesh_contact_max = 10

    # Set up bookkeeping for envs.
    bodies_per_env = model.body_count // batch_dim
    model.cube_ids = wp.from_numpy(bodies_per_env * np.arange(batch_dim), dtype=int)

    return model


@wp.kernel
def get_cube_sdf(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    point0: wp.array(dtype=wp.vec3),
    shape0: wp.array(dtype=int),
    point1: wp.array(dtype=wp.vec3),
    shape1: wp.array(dtype=int),
    contact_normal: wp.array(dtype=wp.vec3),
    cube_ids: wp.array(dtype=int),
    # outputs
    sdf_val: wp.array(dtype=float),
):
    """
    Kernel to compute the SDF value for the given contact point.
    """
    env_id, coll_id = wp.tid()
    shape_a = shape0[coll_id]
    shape_b = shape1[coll_id]

    if shape_a < 0 or shape_b < 0:
        return

    body_a = shape_body[shape_a]
    body_b = shape_body[shape_b]

    if body_a < 0 or body_b < 0:
        return

    # Check that one of the collisions is with this env's cube.
    if body_a != cube_ids[env_id] and body_b != cube_ids[env_id]:
        return

    n = contact_normal[coll_id]
    point_a = point0[coll_id]
    point_b = point1[coll_id]
    X_wb_a = body_q[body_a]
    X_wb_b = body_q[body_b]

    point_a_world = wp.transform_point(X_wb_a, point_a)
    point_b_world = wp.transform_point(X_wb_b, point_b)

    d = wp.dot(n, point_a_world - point_b_world)
    if d < 0:
        wp.atomic_add(sdf_val, env_id, -d)


# @wp.kernel
# def get_cube_sdf(
#     sdf_val: wp.array(dtype=float),
#     point0: wp.array(dtype=wp.vec3),
#     shape0: wp.array(dtype=int),
#     point1: wp.array(dtype=wp.vec3),
#     shape1: wp.array(dtype=int),
#     geo: wp.sim.ModelShapeGeometry,
#     cube_ids: wp.array(dtype=int),
# ):
#     """
#     Kernel to compute the SDF value for the given contact point.
#     """
#     # Query the thread ID.
#     env_id, coll_id = wp.tid()
#     bounds = geo.scale[cube_ids[env_id]]

#     if shape0[coll_id] == cube_ids[env_id]:
#         # Cube is the first shape in the pair.
#         cube_point = point0[coll_id]
#         wp.atomic_add(sdf_val, env_id, box_sdf(bounds, cube_point))
#     elif shape1[coll_id] == cube_ids[env_id]:
#         # Cube is the second shape in the pair.
#         cube_point = point1[coll_id]
#         wp.atomic_add(sdf_val, env_id, box_sdf(bounds, cube_point))


def sdf_loss_factory(leap_model: wp.sim.Model):
    dummy_qd = wp.zeros(leap_model.joint_dof_count, dtype=float, requires_grad=False)

    class SDFLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q_warp):
            assert q_warp.shape[0] == leap_model.num_envs
            assert (
                q_warp.shape[1] == leap_model.joint_coord_count // leap_model.num_envs
            )

            tape = wp.Tape()

            # Cache input.
            ctx.q_warp = wp.from_torch(q_warp.reshape(-1))

            ctx.state = leap_model.state()

            # Create dummy velocity for FK.
            ctx.qd_warp = dummy_qd

            # Allocate output.
            sdf_loss = wp.zeros(leap_model.num_envs, dtype=float)

            # Run SDF loss.
            with tape:
                warp.sim.eval_fk(leap_model, ctx.q_warp, ctx.qd_warp, None, ctx.state)
                warp.sim.collide(leap_model, ctx.state)

                breakpoint()
                wp.launch(
                    kernel=get_cube_sdf,
                    dim=(
                        leap_model.num_envs,
                        leap_model.rigid_contact_count.numpy()[0],
                    ),
                    inputs=[
                        ctx.state.body_q,
                        leap_model.shape_body,
                        leap_model.rigid_contact_point0,
                        leap_model.rigid_contact_shape0,
                        leap_model.rigid_contact_point1,
                        leap_model.rigid_contact_shape1,
                        leap_model.rigid_contact_normal,
                        leap_model.cube_ids,
                    ],
                    outputs=[sdf_loss],
                )

            ctx.tape = tape
            ctx.sdf_loss = sdf_loss

            return wp.to_torch(sdf_loss)

        @staticmethod
        def backward(ctx, grad_output):
            ctx.tape.backward(grads={ctx.sdf_loss: wp.from_torch(grad_output)})

            return wp.to_torch(ctx.tape.gradients[ctx.q_warp]).reshape(
                leap_model.num_envs, -1
            )

    return SDFLoss


if __name__ == "__main__":
    batch_dim = 5
    leap_model = get_leap_model(batch_dim=batch_dim)
    state = leap_model.state()
    example_q0 = np.array(
        [
            0.11553308912386835,
            0.025024442129183307,
            0.014601701013649514,
            -7.58546157048416e-05,
            0.24822135705068318,
            -0.00041132275562983425,
            0.9687032481434059,
            0.4475251747550032,
            -0.6313177576435757,
            0.8438006658498811,
            0.6732342167606099,
            0.2696511460178073,
            -0.1438198146159546,
            0.49475788581026126,
            0.5531799888313503,
            0.2654348834035911,
            0.6358605619274197,
            -0.08673486653636647,
            0.7723508599279184,
            0.7495066105075004,
            0.43643923829057957,
            0.43190393627560353,
            0.6054767544975855,
        ]
    )

    # q_batch = np.concatenate([example_q0] * batch_dim, axis=0)

    # warp.sim.eval_fk(
    #     leap_model, wp.from_numpy(q_batch, dtype=float), state.joint_qd, None, state
    # )
    # warp.sim.collide(leap_model, state)
    example_q1 = np.array(
        [
            0.13972541259426183,
            0.025549761307820128,
            0.01592872718495079,
            0.004999615076543016,
            0.5079323538083438,
            -0.023395202233069033,
            0.8610646853264128,
            0.29688820868341237,
            -0.47307474040755904,
            0.007716432872939806,
            1.0302352207274688,
            0.41706749847424535,
            -0.6423925531357679,
            -0.5596114023360442,
            0.4551523683424788,
            -0.25272842618178487,
            -0.07798047182888083,
            -0.22629965755988002,
            1.066798886837416,
            0.06789156756579362,
            0.6151034211328886,
            0.8447518071665147,
            1.5213544364041642,
        ]
    )
    example_q2 = example_q0.copy()
    example_q2[:3] = 0.0
    # example_q0[:3] = 0.0
    example_q0[2] = 1.0
    example_qd = np.zeros_like(example_q0[:-1])

    # # # cast example states to warp arrays.
    # # example_q0 = wp.from_numpy(example_q0, dtype=float, requires_grad=True)
    # # example_q1 = wp.from_numpy(example_q1, dtype=float, requires_grad=True)
    # # example_q2 = wp.from_numpy(example_q2, dtype=float, requires_grad=True)
    # # example_qd0 = wp.from_numpy(example_qd, dtype=float)
    # # example_qd1 = wp.from_numpy(example_qd, dtype=float)

    # # state0 = leap_model.state()
    # # state1 = leap_model.state()
    # # state0.joint_q = example_q0
    # # state0.joint_qd = example_qd0
    # # state1.joint_q = example_q1
    # # state1.joint_qd = example_qd1

    SDFLoss = sdf_loss_factory(leap_model)

    # q0_torch = torch.tensor(example_q0, requires_grad=True, device="cuda")
    # # q1_torch = torch.tensor(example_q1, requires_grad=True, device="cuda")
    # # q2_torch = torch.tensor(example_q2, requires_grad=True, device="cuda")
    q_batch = (
        torch.from_numpy(np.stack([example_q1] * batch_dim, axis=0)).float().cuda()
    )
    q_batch.requires_grad = True
    breakpoint()
    # q_batch = torch.randn(
    #     batch_dim,
    #     leap_model.joint_coord_count // batch_dim,
    #     device="cuda",
    #     requires_grad=True,
    # )
    with wp.ScopedTimer("sdf_loss", synchronize=True):
        sdf_loss = SDFLoss.apply(q_batch).mean()
    # with wp.ScopedTimer("sdf_loss_grad", synchronize=True):
    #     sdf_loss.backward()

    breakpoint()

    # tape = wp.Tape()
    # with tape:
    #     with wp.ScopedTimer("fk_eval", synchronize=True):
    #         warp.sim.eval_fk(leap_model, example_q0, example_qd0, None, state0)

    #     # with wp.ScopedTimer("fk_eval", synchronize=True):
    #     #     warp.sim.eval_fk(leap_model, example_q1, example_qd1, None, state1)

    #     with wp.ScopedTimer("collision", synchronize=True):
    #         wp.sim.collide(leap_model, state0)

    #     sdf_loss = wp.zeros(1, dtype=float, requires_grad=True)

    #     with wp.ScopedTimer("sdf_eval", synchronize=True):
    #         wp.launch(
    #             kernel=get_cube_sdf,
    #             dim=int(leap_model.rigid_contact_count.numpy()[0]),
    #             inputs=[
    #                 sdf_loss,
    #                 leap_model.rigid_contact_point0,
    #                 leap_model.rigid_contact_shape0,
    #                 leap_model.rigid_contact_point1,
    #                 leap_model.shape_geo,
    #             ],
    #             device="cuda",
    #         )

    #     with wp.ScopedTimer("sdf_grad", synchronize=True):
    #         tape.backward(loss=sdf_loss)

    # sdf_vals = wp.zeros_like(leap_model.rigid_contact_thickness)

    # with wp.ScopedTimer("collision", synchronize=True):
    #     wp.sim.collide(leap_model, state1)

    # with wp.ScopedTimer("sdf_eval", synchronize=True):
    #     wp.launch(
    #         kernel=get_cube_sdf,
    #         dim=int(leap_model.rigid_contact_count.numpy()[0]),
    #         inputs=[
    #             sdf_vals,
    #             leap_model.rigid_contact_point0,
    #             leap_model.rigid_contact_shape0,
    #             leap_model.rigid_contact_point1,
    #             leap_model.shape_geo,
    #         ],
    #         device="cuda",
    #     )

    # integrator = wp.sim.SemiImplicitIntegrator()
    # state0.clear_forces()

    # with wp.ScopedTimer("integrate", synchronize=True):
    #     integrator.simulate(leap_model, state0, state1, 0.01)

    # with wp.ScopedTimer("sdf_loss", synchronize=True):
    #     sdf_loss = SDFLoss.apply(q_batch).mean()
    # with wp.ScopedTimer("sdf_loss_grad", synchronize=True):
    #     sdf_loss.backward()

    # breakpoint()

    # print(state.body_q)
