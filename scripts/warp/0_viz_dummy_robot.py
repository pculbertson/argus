import numpy as np
import warp as wp
import warp.render
import warp.sim
import warp.sim.render

from argus import ROOT

builder = wp.sim.ModelBuilder()
path = f"{ROOT}/scripts/warp/dummy.urdf"
wp.sim.parse_urdf(
    path,
    builder,
    floating=False,
    enable_self_collisions=False,
    collapse_fixed_joints=True,
)
model = builder.finalize(requires_grad=True)
model.ground = False

# renderer
renderer = wp.sim.render.SimRendererOpenGL(path=path, model=model, scaling=15.0)


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
        render(t, model.state())
        t += 1

breakpoint()


# class Example:
#     def __init__(self):
#         # renderer
#         self.renderer = wp.render.OpenGLRenderer(vsync=False)
#         self.renderer.render_ground()

#         # model
#         builder = wp.sim.ModelBuilder()
#         wp.sim.parse_urdf(
#             f"{ROOT}/scripts/warp/dummy.urdf",
#             builder,
#             enable_self_collisions=False,
#             collapse_fixed_joints=True,
#         )
#         self.model = builder.finalize(requires_grad=True)
#         self.model.ground = False

#     def render(self):
#         time = self.renderer.clock_time
#         self.renderer.begin_frame(time)

#         self.renderer.end_frame()


# if __name__ == "__main__":
#     import argparse
#     import distutils.util

#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
#     parser.add_argument("--num_tiles", type=int, default=4, help="Number of viewports to render in a single frame.")
#     parser.add_argument(
#         "--show_plot",
#         type=lambda x: bool(distutils.util.strtobool(x.strip())),
#         default=True,
#         help="Display the pixels in an additional matplotlib figure.",
#     )
#     parser.add_argument("--render_mode", type=str, choices=("depth", "rgb"), default="depth", help="")
#     parser.add_argument(
#         "--split_up_tiles",
#         type=lambda x: bool(distutils.util.strtobool(x.strip())),
#         default=True,
#         help="Whether to split tiles into subplots when --show_plot is True.",
#     )
#     parser.add_argument("--custom_tile_arrangement", action="store_true", help="Apply custom tile arrangement.")

#     args = parser.parse_known_args()[0]

#     with wp.ScopedDevice(args.device):
#         example = Example(num_tiles=args.num_tiles, custom_tile_arrangement=args.custom_tile_arrangement)

#         channels = 1 if args.render_mode == "depth" else 3
#         while example.renderer.is_running():
#             example.render()

#             if args.show_plot and plt.fignum_exists(1):
#                 if args.split_up_tiles:
#                     pixel_shape = (args.num_tiles, example.renderer.tile_height, example.renderer.tile_width, channels)
#                 else:
#                     pixel_shape = (example.renderer.screen_height, example.renderer.screen_width, channels)

#                 if pixel_shape != pixels.shape:
#                     # make sure we resize the pixels array to the right dimensions if the user resizes the window
#                     pixels = wp.zeros(pixel_shape, dtype=wp.float32)

#                 example.renderer.get_pixels(pixels, split_up_tiles=args.split_up_tiles, mode=args.render_mode)

#                 if args.split_up_tiles:
#                     pixels_np = pixels.numpy()
#                     for i, img_plot in enumerate(img_plots):
#                         img_plot.set_data(pixels_np[i])
#                 else:
#                     img_plot.set_data(pixels.numpy())
#                 fig.canvas.draw()
#                 fig.canvas.flush_events()

#         example.renderer.clear()
