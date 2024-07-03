from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import warp as wp
import warp.render
import warp.sim
import warp.sim.render
from plotly.subplots import make_subplots

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


@wp.kernel
def batch_transform(
    in_points: wp.array(dtype=wp.vec3),
    transform: wp.array(dtype=wp.transform),
    # outputs
    out_points: wp.array(dtype=wp.vec3),
):
    """Transforms a batch of points by a given transform."""
    point_id = wp.tid()
    out_points[point_id] = wp.transform_point(transform[0], in_points[point_id])


# ########### #
# MAIN SCRIPT #
# ########### #

if __name__ == "__main__":
    # [1] creating batched model
    # the state of the batched model is a concatenation of the batched states
    # so, here it is of size batch_size * (1 + 7), where 1 is the joint state of
    # the 2-link robot and 7 is the (translation, quat_xyzw) state of the cube
    cube_size = 0.035
    batch_size = 2
    model = get_model(f"{ROOT}/scripts/warp/dummy_with_mesh.urdf", cube_size=0.035, batch_size=batch_size)
    q0 = np.zeros(8)  # creating state for a single robot
    q0[1] = 0.5  # x position of the cube is 0.5 meters out
    q0[7] = 1.0  # w coordinate of the quaternion
    q0_batch = wp.from_numpy(np.concatenate([q0] * batch_size), device="cuda", dtype=float, requires_grad=True)

    # [2] collide and set state
    state = model.state()
    qd0_batch = wp.zeros(model.joint_dof_count, device="cuda", requires_grad=True)  # (14,)
    wp.sim.eval_fk(model, q0_batch, qd0_batch, None, state)  # updates body states given initial joint states
    # wp.sim.eval_ik(model, state, state.joint_q, state.joint_qd)  # updates joint states w/body states
    wp.sim.collide(model, state)

    # [3] compute contact points - we truncate the contact arrays post collide call to save memory
    # TODO(ahl): will this cause kernel recompilation every time there is a different number of contacts?
    rigid_contact_count = model.rigid_contact_count.numpy()[0].item()  # total number of contacts

    rigid_contact_shape0 = model.rigid_contact_shape0[:rigid_contact_count]  # inputs
    rigid_contact_shape1 = model.rigid_contact_shape1[:rigid_contact_count]
    rigid_contact_point0 = model.rigid_contact_point0[:rigid_contact_count]
    rigid_contact_point1 = model.rigid_contact_point1[:rigid_contact_count]
    rigid_contact_normal = model.rigid_contact_normal[:rigid_contact_count]
    batch_idxs = wp.from_numpy(-np.ones(rigid_contact_count), dtype=int, device="cuda")  # outputs
    contact_points_cube = wp.zeros(rigid_contact_count, dtype=wp.vec3, device="cuda")  # outputs
    contact_points_other = wp.zeros(rigid_contact_count, dtype=wp.vec3, device="cuda")
    sdf_vals = wp.zeros(rigid_contact_count, dtype=float, device="cuda")

    wp.launch(
        kernel=get_cube_contact_points_and_sdf_vals,  # the name of the kernel to launch
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

    # [4] organize the data by batch element
    batch_idxs_numpy = batch_idxs.numpy()
    contact_points_cube_numpy = contact_points_cube.numpy()
    contact_points_other_numpy = contact_points_other.numpy()
    sdf_vals_numpy = sdf_vals.numpy()

    print(batch_idxs_numpy)

    contact_points_dict = {}
    for i in range(batch_size):
        idxs = np.where(batch_idxs_numpy == i)[0]
        contact_points_dict[i] = {
            "contact_points_cube": contact_points_cube_numpy[idxs],
            "contact_points_other": contact_points_other_numpy[idxs],
            "sdf_vals": sdf_vals_numpy[idxs],
        }

    # [5] visualize the data
    fig = make_subplots(
        rows=1,
        cols=model.num_envs,
        specs=[[{"type": "scatter3d"}] * model.num_envs],
    )
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

    for i in range(batch_size):
        # plot the contact points
        cp_dict = contact_points_dict[i]
        contact_points_cube = cp_dict["contact_points_cube"][cp_dict["sdf_vals"] < 0]
        contact_points_other = cp_dict["contact_points_other"][cp_dict["sdf_vals"] < 0]
        fig.add_trace(
            go.Scatter3d(
                x=contact_points_cube[:, 0],
                y=contact_points_cube[:, 1],
                z=contact_points_cube[:, 2],
                mode="markers",
                marker={"size": 5, "color": "blue"},
            ),
            row=1,
            col=i + 1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=contact_points_other[:, 0],
                y=contact_points_other[:, 1],
                z=contact_points_other[:, 2],
                mode="markers",
                marker={"size": 5, "color": "green"},
            ),
            row=1,
            col=i + 1,
        )

        # plot the cube
        in_cube_vertices = wp.from_numpy(cube_vertices, dtype=wp.vec3)
        X_WCube = wp.from_numpy(state.body_q.numpy()[model.cube_idxs.numpy()[i]], dtype=wp.transform)
        out_cube_vertices = wp.zeros_like(in_cube_vertices)
        wp.launch(
            kernel=batch_transform,
            dim=(8,),
            inputs=[in_cube_vertices, X_WCube],
            outputs=[out_cube_vertices],
        )

        cube_vertices_transformed = out_cube_vertices.numpy()
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
                    x=[cube_vertices_transformed[edge[0], 0], cube_vertices_transformed[edge[1], 0]],
                    y=[cube_vertices_transformed[edge[0], 1], cube_vertices_transformed[edge[1], 1]],
                    z=[cube_vertices_transformed[edge[0], 2], cube_vertices_transformed[edge[1], 2]],
                    mode="lines",
                    line={"color": "black"},
                ),
                row=1,
                col=i + 1,
            )

        fig.update_layout({f"scene{i + 1}_camera": {"up": {"x": 0, "y": 0, "z": 1}}})

    # show the plot after plotting all frames
    fig.show()
