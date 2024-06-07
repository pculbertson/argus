import warp as wp
import warp.sim
from argus import ROOT
from pathlib import Path
from typing import Optional
import numpy as np
import torch

wp.init()
wp.config.verify_cuda = True


def get_leap_model(mujoco_path: Optional[Path] = None, batch_dim: int = 1):
    if mujoco_path is None:
        # Use default model path.
        leap_mjcf_model_path = Path(ROOT) / "mujoco" / "leap" / "leap_hand.xml"
        cube_model_path = (
            Path(ROOT) / "mujoco" / "common_assets" / "reorientation_cube.xml"
        )
        leap_urdf_model_path = Path(ROOT) / "LeapProject/Assets/urdf/robot_copy.urdf"
    else:
        leap_mjcf_model_path = mujoco_path / "leap" / "leap_hand.xml"
        cube_model_path = mujoco_path / "common_assets" / "reorientation_cube.xml"

    # Create model builder.
    leap_builder = wp.sim.ModelBuilder()

    cube_size = 0.035
    cube0 = leap_builder.add_body()
    leap_builder.add_joint_free(cube0)
    leap_builder.add_shape_box(cube0, hx=cube_size, hy=cube_size, hz=cube_size)
    # # Parse cube MJCF into Warp.
    # wp.sim.parse_mjcf(
    #     str(cube_model_path),
    #     leap_builder,
    #     xform=None,
    #     enable_self_collisions=False,
    #     # collapse_fixed_joints=True,
    # )

    # Parse Leap MJCF into Warp.
    # wp.sim.parse_mjcf(
    #     str(leap_mjcf_model_path),
    #     leap_builder,
    #     enable_self_collisions=False,
    #     collapse_fixed_joints=True,
    # )
    wp.sim.parse_urdf(
        str(leap_urdf_model_path),
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

    # List cube ids.
    # cube_ids = [
    #     ii for ii in range(len(model.body_name)) if model.body_name[ii] == "cube"
    # ]
    cube_ids = (model.body_count // batch_dim) * np.arange(batch_dim)

    model.cube_ids = wp.array(cube_ids, dtype=int)

    # Limit mesh collision pairs.
    # model.rigid_mesh_contact_max = 10
    # model.rigid_contact_max = 10000
    return model


@wp.kernel
def get_contact_points(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    cube_ids: wp.array(dtype=int),
    # outputs
    contact_points_cube: wp.array2d(dtype=wp.vec3),
    contact_points_other: wp.array2d(dtype=wp.vec3),
    sdf_vals: wp.array2d(dtype=float),
):
    env_id, contact_id = warp.tid()
    if contact_id >= contact_count[0]:
        return

    shape_a = contact_shape0[contact_id]
    shape_b = contact_shape1[contact_id]

    if shape_a == shape_b:
        return

    body_a = -1
    body_b = -1

    if shape_a >= 0:
        body_a = shape_body[shape_a]

    if shape_b >= 0:
        body_b = shape_body[shape_b]

    # Check if the contact is this env's cube.
    if body_a != cube_ids[env_id] and body_b != cube_ids[env_id]:
        return

    n = contact_normal[contact_id]
    bx_a = contact_point0[contact_id]
    bx_b = contact_point1[contact_id]

    X_wb_a = body_q[body_a]
    wx_a = wp.transform_point(X_wb_a, bx_a)

    X_wb_b = body_q[body_b]
    wx_b = wp.transform_point(X_wb_b, bx_b)

    d = wp.dot(n, wx_a - wx_b)
    if d < 0:
        if body_a == cube_ids[env_id]:
            contact_points_cube[env_id, contact_id] = wx_a
            contact_points_other[env_id, contact_id] = wx_b
        else:
            contact_points_cube[env_id, contact_id] = wx_b
            contact_points_other[env_id, contact_id] = wx_a
        sdf_vals[env_id, contact_id] = -d


@wp.kernel
def batch_transform(
    in_points: wp.array(dtype=wp.vec3),
    transform: wp.array(dtype=wp.transform),
    # outputs
    out_points: wp.array(dtype=wp.vec3),
):
    point_id = wp.tid()
    out_points[point_id] = wp.transform_point(transform[0], in_points[point_id])


if __name__ == "__main__":
    batch_size = 2
    model = get_leap_model(batch_dim=batch_size)

    # Generate batched data
    example_q1 = np.array(
        [
            0.1615615252370641,
            0.029863455184370065,
            0.008785586532427017,
            -0.048186096924192484,
            0.7112424721246233,
            -0.10639920940589315,
            0.6931749087691116,
            0.39418120312144983,
            -0.6579748914971878,
            0.006664389558307797,
            1.0468940287717396,
            0.5231406168956857,
            -0.6847459339571726,
            -0.21710036714368028,
            -0.0901512012503739,
            0.0041409859377174035,
            -0.07967162505732361,
            0.25770139813479004,
            0.9659057024637947,
            -0.05869264666739396,
            1.2993330609370368,
            1.0707825783254097,
            1.0781484051945347,
        ]
    )

    example_q1 = np.array(
        [
            0.16156152,
            0.00878559,
            -0.02986345,
            0.52422136,
            -0.42768878,
            0.57816,
            -0.45607594,
            0.39418120312144983,
            -0.6579748914971878,
            0.006664389558307797,
            1.0468940287717396,
            0.5231406168956857,
            -0.6847459339571726,
            -0.21710036714368028,
            -0.0901512012503739,
            0.0041409859377174035,
            -0.07967162505732361,
            0.25770139813479004,
            0.9659057024637947,
            -0.05869264666739396,
            1.2993330609370368,
            1.0707825783254097,
            1.0781484051945347,
        ]
    )

    breakpoint()
    q_batch = np.concatenate([example_q1] * batch_size, axis=0)
    q_batch = wp.from_numpy(q_batch, device="cuda", dtype=float)

    # Generate state data + run FK/collider.
    state = model.state()

    # Create dummy qd for FK.
    qd_batch = wp.zeros(model.joint_dof_count, device="cuda")

    # Run FK.
    warp.sim.eval_fk(model, q_batch, qd_batch, None, state)

    # Run collider.
    warp.sim.collide(model, state)

    # Get contact points.
    contact_points_cube = wp.zeros(
        shape=(batch_size, model.rigid_contact_max), dtype=wp.vec3, device="cuda"
    )

    contact_points_other = wp.zeros(
        shape=(batch_size, model.rigid_contact_max), dtype=wp.vec3, device="cuda"
    )

    sdf_vals = wp.zeros(
        shape=(batch_size, model.rigid_contact_max), dtype=float, device="cuda"
    )

    breakpoint()

    wp.launch(
        kernel=get_contact_points,
        dim=(
            batch_size,
            model.rigid_contact_max,
        ),
        inputs=[
            state.body_q,
            model.shape_body,
            model.rigid_contact_count,
            model.rigid_contact_point0,
            model.rigid_contact_point1,
            model.rigid_contact_normal,
            model.rigid_contact_shape0,
            model.rigid_contact_shape1,
            model.cube_ids,
        ],
        outputs=[contact_points_cube, contact_points_other, sdf_vals],
    )

    breakpoint()
    contact_points_cube = contact_points_cube.numpy()
    contact_points_other = contact_points_other.numpy()

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=model.num_envs,
        specs=[[{"type": "scatter3d"}] * model.num_envs],
    )

    cube_size = model.shape_geo.scale.numpy()[0][0]
    cube_vertices = np.array(
        [
            [-cube_size, -cube_size, -cube_size],
            [-cube_size, -cube_size, cube_size],
            [-cube_size, cube_size, -cube_size],
            [-cube_size, cube_size, cube_size],
            [cube_size, -cube_size, -cube_size],
            [cube_size, -cube_size, cube_size],
            [cube_size, cube_size, -cube_size],
            [cube_size, cube_size, cube_size],
        ]
    )
    in_cube_vertices = wp.from_numpy(cube_vertices, dtype=wp.vec3)

    # Scatter contact points.
    for ii in range(batch_size):
        nonzero_inds = np.where(contact_points_cube[ii, :, 0] != 0)[0]
        fig.add_trace(
            go.Scatter3d(
                x=contact_points_cube[ii, nonzero_inds, 0],
                y=contact_points_cube[ii, nonzero_inds, 1],
                z=contact_points_cube[ii, nonzero_inds, 2],
                mode="markers",
                marker=dict(size=5, color="blue"),
            ),
            row=1,
            col=ii + 1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=contact_points_other[ii, nonzero_inds, 0],
                y=contact_points_other[ii, nonzero_inds, 1],
                z=contact_points_other[ii, nonzero_inds, 2],
                mode="markers",
                marker=dict(size=5, color="green"),
            ),
            row=1,
            col=ii + 1,
        )

        # Plot dotted lines between the contact pairs.
        # fig.add_trace(
        #     go.Scatter3d(
        #         x=np.stack(
        #             [
        #                 contact_points_cube[ii, nonzero_inds, 0],
        #                 contact_points_other[ii, nonzero_inds, 0],
        #             ],
        #             axis=0,
        #         ),
        #         y=np.stack(
        #             [
        #                 contact_points_cube[ii, nonzero_inds, 1],
        #                 contact_points_other[ii, nonzero_inds, 1],
        #             ],
        #             axis=0,
        #         ),
        #         z=np.stack(
        #             [
        #                 contact_points_cube[ii, nonzero_inds, 2],
        #                 contact_points_other[ii, nonzero_inds, 2],
        #             ],
        #             axis=0,
        #         ),
        #         mode="lines",
        #         line=dict(color="black", dash="dash"),
        #     ),
        #     row=1,
        #     col=ii + 1,
        # )

        # Add cubes.
        out_cube_vertices = wp.zeros_like(in_cube_vertices)
        cube0_state = state.body_q.numpy()[model.cube_ids.numpy()[ii]]
        cube0_state = wp.from_numpy(cube0_state, dtype=wp.transform)
        breakpoint()
        wp.launch(
            kernel=batch_transform,
            dim=(8,),
            inputs=[in_cube_vertices, cube0_state],
            outputs=[out_cube_vertices],
        )

        cube_vertices = out_cube_vertices.numpy()
        cube_edges = [
            [0, 1],
            [1, 3],
            [3, 2],
            [2, 0],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
            [4, 5],
            [5, 7],
            [7, 6],
            [6, 4],
        ]

        for edge in cube_edges:
            fig.add_trace(
                go.Scatter3d(
                    x=[cube_vertices[edge[0], 0], cube_vertices[edge[1], 0]],
                    y=[cube_vertices[edge[0], 1], cube_vertices[edge[1], 1]],
                    z=[cube_vertices[edge[0], 2], cube_vertices[edge[1], 2]],
                    mode="lines",
                    line=dict(color="blue"),
                ),
                row=1,
                col=ii + 1,
            )

        fig.update_layout(
            {
                f"scene{ii+1}_camera": dict(
                    up=dict(x=0, y=1, z=0),
                )
            }
        )

    fig.show()
    print(sdf_vals.numpy().max(-1))

    breakpoint()
