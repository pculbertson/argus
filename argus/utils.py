import fnmatch
import os
from typing import Callable

import numpy as np
import pypose as pp
import torch
from scipy.spatial.transform import Rotation as R

# ###################### #
# CONVENTION CONVERSIONS #
# ###################### #


def convert_pose_mjpc_to_unity(pose_mjpc: np.ndarray) -> np.ndarray:
    """Converts a pose from Mujoco's coordinate system to Unity's coordinate system.

    The differences between the coordinate systems are the handedness of the frames and the direction of the axes. The
    quaternion convention is also different.

    Args:
        pose_mjpc: Pose in Mujoco's coordinate system. Shape=(..., 7), where the last 4 elements are the quaternion in
            wxyz convention. The mjpc coordinate system is right-handed, with +x "forward," +y "right," and +z "up."

    Returns:
        pose_unity: Pose in Unity's coordinate system. Shape=(..., 7), where the last 4 elements are the quaternion in
            xyzw convention. The Unity coordinate system is left-handed, with +z "forward," +x "left," and +y "up."
    """
    # translations - here, we use the improper rotation matrix to rotate the translation vector
    R_mjpc_to_unity_left_hand = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    trans_mjpc = pose_mjpc[..., :3, None]  # (..., 3, 1)
    trans_unity = (R_mjpc_to_unity_left_hand @ trans_mjpc).squeeze(-1)  # (..., 3)

    # rotations
    q_wxyz = pose_mjpc[..., 3:]  # (..., 4)
    q_xyzw = np.concatenate([q_wxyz[..., 1:], q_wxyz[..., :1]], axis=-1)  # (..., 4)
    quat_unity = np.concatenate(
        [
            -q_xyzw[..., 1:2],  # -y, rotation about y in mjpc is rotation about -x in unity
            q_xyzw[..., 2:3],  # z, rotation about z in mjpc is rotation about y in unity
            q_xyzw[..., 0:1],  # x, rotation about x in mjpc is rotation about x in unity
            -q_xyzw[..., 3:4],  # -w, switch angle sign for right to left hand convention
        ],
        axis=-1,
    )
    quat_unity[quat_unity[..., 3] < 0] = -quat_unity[quat_unity[..., 3] < 0]  # return with positive w

    # concatenating and returning
    pose_unity = np.concatenate([trans_unity, quat_unity], axis=-1)  # (..., 7)
    return pose_unity


def convert_pose_unity_to_mjpc(pose_unity: np.ndarray) -> np.ndarray:
    """Converts a pose from Unity's coordinate system to Mujoco's coordinate system.

    Inverse operation of `convert_pose_mjpc_to_unity`. For more info, check its docstring.
    """
    # translations
    R_unity_to_mjpc_left_hand = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    trans_unity = pose_unity[..., :3, None]  # (..., 3, 1)
    trans_mjpc = (R_unity_to_mjpc_left_hand @ trans_unity).squeeze(-1)  # (..., 3)

    # rotations
    q_xyzw = pose_unity[..., 3:]  # (..., 4)
    q_wxyz = np.concatenate([q_xyzw[..., -1:], q_xyzw[..., :-1]], axis=-1)  # (..., 4)
    quat_mjpc = np.concatenate(
        [
            -q_wxyz[..., 0:1],  # -w, switch angle sign for left to right hand convention
            q_wxyz[..., 3:4],  # z, rotation about z in unity is rotation about x in mjpc
            -q_wxyz[..., 1:2],  # -x, rotation about x in unity is rotation about -y in mjpc
            q_wxyz[..., 2:3],  # y, rotation about y in unity is rotation about z in mjpc
        ],
        axis=-1,
    )
    quat_mjpc[quat_mjpc[..., 0] < 0] = -quat_mjpc[quat_mjpc[..., 0] < 0]  # return with positive w

    # concatenating and returning
    pose_mjpc = np.concatenate([trans_mjpc, quat_mjpc], axis=-1)  # (..., 7)
    return pose_mjpc


def convert_unity_quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Converts a quaternion in the Unity convention to Euler angles in degrees.

    Useful for debugging by manually inputing Euler angles into the Unity editor.

    Args:
        quat: Quaternion in xyzw convention with Unity's axes. Shape=(..., 4).

    Returns:
        euler: Euler angles in degrees. Shape=(..., 3), where the last dimension is the roll-pitch-yaw angles.
    """
    euler = R.from_quat(quat).as_euler("XYZ", degrees=True)
    return euler


def xyzwxyz_to_xyzxyzw_SE3(xyzwxyz: torch.Tensor) -> torch.Tensor:
    """Converts a tensor of 7d poses with quats from (w, x, y, z) to (x, y, z, w) order.

    Args:
        xyzwxyz: The tensor of poses of shape (..., 7) whose quats are in (w, x, y, z) order.

    Returns:
        xyzxyzw: The tensor now with (x, y, z, w) order.
    """
    return torch.cat(
        (
            xyzwxyz[..., :3],  # translations
            xyzwxyz[..., -3:],  # the (qx, qy, qz) components
            xyzwxyz[..., -4:-3],  # the qw component
        ),
        dim=-1,
    )


def xyzxyzw_to_xyzwxyz_SE3(xyzxyzw: torch.Tensor) -> torch.Tensor:
    """Converts a tensor of 7d poses with quats from (x, y, z, w) to (w, x, y, z) order.

    Args:
        xyzxyzw: The tensor of poses of shape (..., 7) whose quats are in (x, y, z, w) order.

    Returns:
        xyzwxyz: The tensor now with (w, x, y, z) order.
    """
    return torch.cat(
        (
            xyzxyzw[..., :3],  # translations
            xyzxyzw[..., -1:],  # the qw component
            xyzxyzw[..., -4:-1],  # the (qx, qy, qz) components
        ),
        dim=-1,
    )


# ########## #
# EVALUATION #
# ########## #


def time_torch_fn(fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, float]:
    """Time a torch function.

    Source: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#demonstrating-speedups

    Args:
        fn: The function to time.

    Returns:
        result: The result of the function.
        time: The time taken to execute the function.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


# ######### #
# INFERENCE #
# ######### #


def get_pose(images: torch.Tensor, model: torch.nn.Module) -> pp.LieTensor:
    """Get the pose of the cube from the images.

    Args:
        images: The images of shape (B, 3 * n_cams, W, H), concatenated along the channel dimension.
        model: The model to use.

    Returns:
        pose: The predicted pose of the cube expressed as a 7d pose. The quaternion elements are in (x, y, z, w) order.
    """
    return pp.se3(model(images)).Exp()


# ######## #
# PRINTING #
# ######## #


def _get_tree_string(path: str, extension: str, indent="") -> str:
    """Returns a tree of the requested path as a string if the leaves match the extension.

    Args:
        path: The path to print the tree of.
        extension: The extension to filter the files by.
        indent: The current indentation level.

    Returns:
        A string representation of the directory tree.
    """
    tree_string = ""

    # get all files in current path
    items = os.listdir(path)
    items.sort()

    # Filter out files that don't match the extension
    items = [
        item for item in items if os.path.isdir(os.path.join(path, item)) or fnmatch.fnmatch(item, f"*.{extension}")
    ]

    for i, item in enumerate(items):
        full_path = os.path.join(path, item)

        # Add with tree formatting
        if i == len(items) - 1:  # Last item in the directory
            tree_string += indent + "└── " + item + "\n"
            new_indent = indent + "    "
        else:  # Not the last item
            tree_string += indent + "├── " + item + "\n"
            new_indent = indent + "│   "

        # If the item is a directory, recursively get its tree string
        if os.path.isdir(full_path):
            tree_string += _get_tree_string(full_path, extension, new_indent)

    return tree_string


def get_tree_string(path: str, extension: str) -> str:
    """Prints the tree of the requested path as a string if the leaves match the extension in blue.

    Args:
        path: The path to print the tree of.
        extension: The extension to filter the files by.

    Returns:
        A string representation of the directory tree starting from the given path.
    """
    BLUE = "\033[94m"
    RESET = "\033[0m"
    return BLUE + path + "\n" + _get_tree_string(path, extension) + RESET
