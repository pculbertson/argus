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
model = builder.finalize(requires_grad=True)
model.ground = False
state = model.state()

# renderer
renderer = wp.sim.render.SimRendererOpenGL(path=path, model=model, scaling=15.0, up_axis="Z")

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
        state.clear_forces()
        control.joint_act = wp.array([np.sin(t / 1000)], dtype=wp.float32)
        integrator.simulate(model, state, state, 0.001, control=control)
        if np.isnan(state.joint_q.numpy()).any():
            breakpoint()  # this block checks for integrator instability
        model.joint_q = state.joint_q
        model.joint_qd = state.joint_qd
        render(t, state)
        t += 1
