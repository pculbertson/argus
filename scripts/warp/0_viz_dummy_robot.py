import numpy as np
import warp as wp
import warp.render
import warp.sim
import warp.sim.render

from argus import ROOT

# building the model
builder = wp.sim.ModelBuilder()
path = f"{ROOT}/scripts/warp/dummy.urdf"
wp.sim.parse_urdf(
    path,
    builder,
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
cube_size = 0.035
cube = builder.add_body()
builder.add_joint_free(cube)
builder.add_shape_box(
    cube,
    hx=cube_size,
    hy=cube_size,
    hz=cube_size,
    ke=1e5,
    kd=250.0,
    kf=500.0,
)
model = builder.finalize(requires_grad=True)
model.ground = False
model.gravity[:] = 0.0  # no gravity
model.joint_attach_ke = 16000.0
model.joint_attach_kd = 200.0
state = model.state()

# update state
q0 = np.zeros(8)  # creating state for a single robot
q0[0] = 0.5  # robot arm rotated 0.5 radians
q0[1] = 0.5  # x position of the cube is 0.5 meters out
q0[2] = -0.12  # y pos off center
q0[7] = 1.0  # w coordinate of the quaternion
q0 = wp.from_numpy(q0, device="cuda", dtype=wp.float32)  # (8,)
qd0 = wp.zeros(model.joint_dof_count, device="cuda")  # (14,)
wp.sim.eval_fk(model, q0, qd0, None, state)  # updates body states given initial joint states
wp.sim.eval_ik(model, state, state.joint_q, state.joint_qd)  # updates joint states w/body states

# renderer
renderer = wp.sim.render.SimRendererOpenGL(path=path, model=model, scaling=1.0, up_axis="Z")

# setting up the integrator and controller
# the SemiImplicit integrator only integrates body coordinates - need to call IK on the
# state after to retrieve joint coordinates. The Featherstone integrator takes care of it.
integrator = wp.sim.FeatherstoneIntegrator(model)
control = model.control()


def render(time, state):
    """Render the scene."""
    with wp.ScopedTimer("render"):
        renderer.begin_frame(time)
        renderer.render(state)
        renderer.end_frame()


# protip: WASD to pan the camera
with wp.ScopedDevice("cuda"):
    t = 0
    while True:
        # update the model state to be moving sinusoidally
        control.joint_act = wp.array([np.sin(t / 1000)], dtype=wp.float32)
        state.clear_forces()
        wp.sim.collide(model, state)
        integrator.simulate(model, state, state, 0.001, control=control)
        if np.isnan(state.joint_q.numpy()).any():
            breakpoint()  # this block checks for integrator instability
        render(t, state)
        t += 1
