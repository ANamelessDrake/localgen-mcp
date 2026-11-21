"""Synthetic pose clips for the tests, built without Blender or any mocap file.

synthetic_raw() makes the input bake_pose_clips.py would hand to pose_math.build_clip: a canonical
humanoid whose local rotations are known (so the tests can check the clip records them), posed
by forward kinematics, optionally turned to face another way, moved forward (root motion) and
scaled to centimetres. Motion, in a 21 frame clip at 24 fps: the right arm winds up over the
shoulder with the elbow bent (frame STRIKE - 4) and strikes forward with the elbow straight at
frame STRIKE; the left elbow bends 90 degrees at frame ELBOW_FRAME; the legs swing a little.
"""

from __future__ import annotations

import math

import numpy as np

from localgen_mcp import pose_math as pm

FRAMES = 21
FPS = 24.0
STRIKE = 10
ELBOW_FRAME = 12
REST = {  # metres, facing -Y, left at +X, feet on the floor
    "pelvis": (0, 0, 1.0), "spine": (0, 0, 1.1), "chest": (0, 0, 1.3), "neck": (0, 0, 1.48), "head": (0, 0, 1.56),
    "shoulder_l": (0.19, 0, 1.44), "elbow_l": (0.19, 0, 1.16), "wrist_l": (0.19, 0, 0.9), "hand_l": (0.19, 0, 0.8),
    "shoulder_r": (-0.19, 0, 1.44), "elbow_r": (-0.19, 0, 1.16), "wrist_r": (-0.19, 0, 0.9), "hand_r": (-0.19, 0, 0.8),
    "hip_l": (0.1, 0, 0.95), "knee_l": (0.1, 0, 0.52), "ankle_l": (0.1, 0, 0.09), "toe_l": (0.1, -0.13, 0.02),
    "hip_r": (-0.1, 0, 0.95), "knee_r": (-0.1, 0, 0.52), "ankle_r": (-0.1, 0, 0.09), "toe_r": (-0.1, -0.13, 0.02),
}
X = (1.0, 0.0, 0.0)


def _rest_all() -> np.ndarray:
    stature = (REST["head"][2] - REST["toe_l"][2]) / (1 - pm.HEAD_TOP)
    out = np.zeros((len(pm.NAMES), 3))
    for j, p in REST.items():
        out[pm.INDEX[j]] = p
    h = np.array(REST["head"], dtype=float)
    out[pm.INDEX["head_top"]] = h + [0, 0, pm.HEAD_TOP * stature]
    out[pm.INDEX["face"]] = h + [0, -pm.FACE[0] * stature, pm.FACE[1] * stature]
    return out


def local_rotations() -> np.ndarray:
    """(FRAMES, J, 4) known local rotations, all about +X (a positive angle turns a hanging segment
    backwards, towards +Y)."""
    loc = np.tile(np.array([1.0, 0, 0, 0]), (FRAMES, len(pm.NAMES), 1))
    for f in range(FRAMES):
        # right shoulder: back and up to the wind-up, a fast swing forward at STRIKE, then back
        if f <= STRIKE - 4:
            sh, el = -150.0 * f / (STRIKE - 4), 90.0 * f / (STRIKE - 4)
        elif f <= STRIKE:
            t = (f - (STRIKE - 4)) / 4
            sh, el = -150.0 + 240.0 * t, 90.0 * (1 - t)
        else:
            t = (f - STRIKE) / (FRAMES - 1 - STRIKE)
            sh, el = 90.0 * (1 - t), 0.0
        loc[f, pm.INDEX["elbow_r"]] = pm.qaxis(X, -sh)  # negative swings a hanging arm forward
        loc[f, pm.INDEX["wrist_r"]] = pm.qaxis(X, -el)
        loc[f, pm.INDEX["wrist_l"]] = pm.qaxis(X, 90.0 * min(1.0, f / ELBOW_FRAME))
        swing = 20.0 * math.sin(2 * math.pi * f / (FRAMES - 1))
        loc[f, pm.INDEX["knee_l"]] = pm.qaxis(X, swing)
        loc[f, pm.INDEX["knee_r"]] = pm.qaxis(X, -swing)
    return loc


def synthetic_raw(yaw: float = 0.0, travel: float = 0.0, unit: float = 1.0) -> dict:
    """bake_pose_clips-style input: world points and bone rotations, the figure turned by yaw degrees
    about Z, moved `travel` metres forward over the clip, in units of `unit` metres."""
    rest_all = _rest_all()
    offsets = np.array([rest_all[j] - rest_all[p] if p >= 0 else np.zeros(3) for j, p in enumerate(pm.PARENTS)])
    loc = local_rotations()
    root = np.tile(rest_all[0], (FRAMES, 1))
    root[:, 1] -= np.linspace(0, travel, FRAMES)
    pos, world = pm.forward(offsets, loc, root)
    turn = pm.qaxis([0, 0, 1], yaw)
    pos = pm.qrot(turn, pos) / unit
    world = pm.qmul(turn, world)
    rest_rot = np.tile(turn, (1, 1))
    real = [j for j in pm.NAMES if j not in pm.SYNTHETIC]
    bones = [f"b_{j}" for j in pm.NAMES if j not in pm.SYNTHETIC] + ["b_skull"]
    specs = {j: {"point": ["head", f"b_{j}"], "carrier": f"b_{j}", "edge": f"b_{j}"} for j in real}
    for j, off in (("head_top", [0.0, pm.HEAD_TOP]), ("face", list(pm.FACE))):
        specs[j] = {"point": ["offset", "b_skull", off], "carrier": "b_skull", "edge": "b_skull"}
    rot_by_bone = {f"b_{j}": world[:, pm.INDEX[j]] for j in real}
    rot_by_bone["b_skull"] = world[:, pm.INDEX["head"]]
    return {
        "fps": FPS, "joints": specs, "bones": bones,
        "rest_points": {j: (pm.qrot(turn, rest_all[pm.INDEX[j]]) / unit).tolist() for j in real},
        "points": pos[:, [pm.INDEX[j] for j in real]].tolist(),
        "rest_rotations": np.tile(rest_rot, (len(bones), 1)).tolist(),
        "rotations": np.stack([rot_by_bone[b] for b in bones], 1).tolist(),
    }


def synthetic_clip(name: str = "fixture_attack", **kw) -> dict:
    return pm.build_clip(synthetic_raw(**kw), name=name)
