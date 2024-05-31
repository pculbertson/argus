import warp as wp
import warp.sim
from warp.sim.collide import box_sdf, box_sdf_grad
from argus import ROOT
from pathlib import Path
from typing import Optional
import numpy as np

wp.init()


def get_leap_model(mujoco_path: Optional[Path] = None):
    if mujoco_path is None:
        # Use default model path.
        leap_mjcf_model_path = Path(ROOT) / "mujoco" / "leap" / "leap_hand.xml"
        cube_model_path = Path(ROOT) / "mujoco" / "common_assets" / "reorientation_cube.xml"
    else:
        leap_mjcf_model_path = mujoco_path / "leap" / "leap_hand.xml"
        cube_model_path = mujoco_path / "common_assets" / "reorientation_cube.xml"

    # Create model builder.
    builder = wp.sim.ModelBuilder()

    # Parse cube MJCF into Warp.
    wp.sim.parse_mjcf(
        str(cube_model_path), builder, xform=None, enable_self_collisions=False, collapse_fixed_joints=True
    )

    # Parse Leap MJCF into Warp.
    wp.sim.parse_mjcf(str(leap_mjcf_model_path), builder, enable_self_collisions=False, collapse_fixed_joints=True)

    # Build the model.
    model = builder.finalize(requires_grad=True)
    model.ground = False

    return model


@wp.kernel
def get_cube_sdf(
    sdf_val: wp.array(dtype=float),
    point0: wp.array(dtype=wp.vec3),
    shape0: wp.array(dtype=int),
    point1: wp.array(dtype=wp.vec3),
    geo: wp.sim.ModelShapeGeometry,
):
    """
    Kernel to compute the SDF value for the given contact point.
    """
    # Query the thread ID.
    tid = wp.tid()
    bounds = geo.scale[0]

    if shape0[tid] == 0:
        # Cube is the first shape in the pair.
        cube_point = point0[tid]
    else:
        # Cube is the second shape in the pair.
        cube_point = point1[tid]

    # Compute the SDF value.
    wp.atomic_add(sdf_val, 0, box_sdf(bounds, cube_point))


if __name__ == "__main__":
    leap_model = get_leap_model()
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
            0.431903936275603,
        ]
    )
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
        ]
    )
    example_qd = np.zeros_like(example_q0[:-1])

    # cast example states to warp arrays.
    example_q0 = wp.from_numpy(example_q0, dtype=float, requires_grad=True)
    example_q1 = wp.from_numpy(example_q1, dtype=float, requires_grad=True)
    example_qd0 = wp.from_numpy(example_qd, dtype=float)
    example_qd1 = wp.from_numpy(example_qd, dtype=float)

    state0 = leap_model.state()
    state1 = leap_model.state()
    # state0.joint_q = example_q0
    # state0.joint_qd = example_qd0
    # state1.joint_q = example_q1
    # state1.joint_qd = example_qd1

    tape = wp.Tape()
    with tape:

        with wp.ScopedTimer("fk_eval", synchronize=True):
            warp.sim.eval_fk(leap_model, example_q0, example_qd0, None, state0)

        # with wp.ScopedTimer("fk_eval", synchronize=True):
        #     warp.sim.eval_fk(leap_model, example_q1, example_qd1, None, state1)

        with wp.ScopedTimer("collision", synchronize=True):
            wp.sim.collide(leap_model, state0)

        sdf_loss = wp.zeros(1, dtype=float, requires_grad=True)

        with wp.ScopedTimer("sdf_eval", synchronize=True):

            wp.launch(
                kernel=get_cube_sdf,
                dim=int(leap_model.rigid_contact_count.numpy()[0]),
                inputs=[
                    sdf_loss,
                    leap_model.rigid_contact_point0,
                    leap_model.rigid_contact_shape0,
                    leap_model.rigid_contact_point1,
                    leap_model.shape_geo,
                ],
                device="cuda",
            )

        with wp.ScopedTimer("sdf_grad", synchronize=True):
            tape.backward(loss=sdf_loss)

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

    breakpoint()

    # print(state.body_q)
