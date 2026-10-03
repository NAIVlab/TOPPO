import argparse
import json
import os

import numpy as np
import torch

from mimickit.anim.motion import LoopMode, Motion
from mimickit.util import torch_util


SPHERICAL_JOINTS = {
    "chest": "abdomen",
    "neck": "neck",
    "right_hip": "right_hip",
    "right_ankle": "right_ankle",
    "right_shoulder": "right_shoulder",
    "left_hip": "left_hip",
    "left_ankle": "left_ankle",
    "left_shoulder": "left_shoulder",
}

MIMICKIT_DOF_ORDER = [
    ("abdomen", 3),
    ("neck", 3),
    ("right_shoulder", 3),
    ("right_elbow", 1),
    ("left_shoulder", 3),
    ("left_elbow", 1),
    ("right_hip", 3),
    ("right_knee", 1),
    ("right_ankle", 3),
    ("left_hip", 3),
    ("left_knee", 1),
    ("left_ankle", 3),
]

DEEPMIMIC_FRAME_LAYOUT = [
    ("chest", 4),
    ("neck", 4),
    ("right_hip", 4),
    ("right_knee", 1),
    ("right_ankle", 4),
    ("right_shoulder", 4),
    ("right_elbow", 1),
    ("left_hip", 4),
    ("left_knee", 1),
    ("left_ankle", 4),
    ("left_shoulder", 4),
    ("left_elbow", 1),
]


def _load_deepmimic_motion(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = np.asarray(data["Frames"], dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != 44:
        raise ValueError(f"Expected DeepMimic humanoid3d frames of shape (N, 44), got {frames.shape}")

    loop = data.get("Loop", "wrap").lower()
    if loop == "wrap":
        loop_mode = LoopMode.WRAP
    elif loop == "none" or loop == "clamp":
        loop_mode = LoopMode.CLAMP
    else:
        raise ValueError(f"Unsupported DeepMimic loop mode: {loop}")

    return frames, loop_mode


def _estimate_fps(frame_dt):
    positive_dt = frame_dt[frame_dt > 0.0]
    if positive_dt.size == 0:
        raise ValueError("DeepMimic motion does not contain positive frame durations")
    return int(round(1.0 / float(np.median(positive_dt))))


def _dm_pos_to_mimickit(pos):
    out = np.empty_like(pos)
    out[:, 0] = pos[:, 0]
    out[:, 1] = -pos[:, 2]
    out[:, 2] = pos[:, 1]
    return out


def _quat_wxyz_to_xyzw(q):
    out = np.empty_like(q)
    out[:, 0] = q[:, 1]
    out[:, 1] = q[:, 2]
    out[:, 2] = q[:, 3]
    out[:, 3] = q[:, 0]
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.maximum(norm, 1.0e-8)


def _basis_change_quat(q_xyzw):
    # DeepMimic humanoid3d uses y-up with right side on z+. MimicKit's MJCF
    # humanoid is z-up with right side on y-. This is a +90 degree rotation
    # around x: (x, y, z)_dm -> (x, -z, y)_mimickit.
    q = torch.as_tensor(q_xyzw, dtype=torch.float32)
    c = np.sqrt(0.5)
    q_basis = torch.tensor([c, 0.0, 0.0, c], dtype=torch.float32)
    q_basis = torch.broadcast_to(q_basis, q.shape)
    q_basis_inv = torch_util.quat_conjugate(q_basis)
    return torch_util.quat_mul(torch_util.quat_mul(q_basis, q), q_basis_inv)


def _quat_wxyz_to_expmap_mimickit(q_wxyz):
    q_xyzw = _quat_wxyz_to_xyzw(q_wxyz)
    q_mimickit = _basis_change_quat(q_xyzw)
    return torch_util.quat_to_exp_map(q_mimickit).numpy()


def _parse_joint_channels(frames):
    idx = 8
    joints = {}
    for name, size in DEEPMIMIC_FRAME_LAYOUT:
        joints[name] = frames[:, idx:idx + size]
        idx += size
    assert idx == frames.shape[1]
    return joints


def _convert_joint_dofs(joints):
    dofs = {}

    for dm_name, mimickit_name in SPHERICAL_JOINTS.items():
        dofs[mimickit_name] = _quat_wxyz_to_expmap_mimickit(joints[dm_name])

    dofs["right_knee"] = -joints["right_knee"]
    dofs["left_knee"] = -joints["left_knee"]
    dofs["right_elbow"] = joints["right_elbow"]
    dofs["left_elbow"] = -joints["left_elbow"]

    ordered = []
    for name, size in MIMICKIT_DOF_ORDER:
        val = dofs[name]
        if val.shape[1] != size:
            raise ValueError(f"Joint {name} expected {size} channels, got {val.shape[1]}")
        ordered.append(val)

    return np.concatenate(ordered, axis=-1)


def convert_deepmimic_to_mimickit(input_file, output_file):
    dm_frames, loop_mode = _load_deepmimic_motion(input_file)
    fps = _estimate_fps(dm_frames[:, 0])

    root_pos = _dm_pos_to_mimickit(dm_frames[:, 1:4])
    root_rot = _quat_wxyz_to_expmap_mimickit(dm_frames[:, 4:8])
    joint_dofs = _convert_joint_dofs(_parse_joint_channels(dm_frames))
    frames = np.concatenate([root_pos, root_rot, joint_dofs], axis=-1).astype(np.float32)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    Motion(loop_mode=loop_mode, fps=fps, frames=frames).save(output_file)

    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Frames: {frames.shape}")
    print(f"FPS: {fps}")
    print(f"Loop mode: {loop_mode.name.lower()}")


def main():
    parser = argparse.ArgumentParser(description="Convert DeepMimic humanoid3d motion txt to MimicKit pkl.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    convert_deepmimic_to_mimickit(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
