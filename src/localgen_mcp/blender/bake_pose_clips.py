"""Bake mocap clips (FBX or BVH) into pose clip JSON files (runs inside Blender's Python).

    blender -b --factory-startup -P bake_pose_clips.py -- '<json spec>'

spec: {"clips": [{"input": path, "output": JSON path, "name": clip name, "action": tag or null,
"loop": bool or null, "weapon_hand": "left"/"right" or null, "take": action name or null,
"root_motion": "in_place" or "keep"}], "report": path}.

Per clip: import the file, take the armature with the most bones and its action (or the action
whose name contains "take"), map its bones to the canonical humanoid joints by name
(pose_math.map_skeleton), sample every frame of the action, and hand the sampled points and bone
rotations to pose_math.build_clip, which normalises facing, floor and root motion, derives the
local rotations and the metadata. pose_math.py is loaded by file path from the package folder.
A clip that fails is reported with its error and the batch goes on. The report (written to
spec["report"] and printed as the last stdout line "POSE_BAKE_RESULT {json}") lists every clip.
"""
import importlib.util
import json
import math
import os
import sys
import time
import traceback

import bpy
import mathutils

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("pose_math", os.path.join(os.path.dirname(HERE), "pose_math.py"))
pose_math = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pose_math)


def load(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if path.lower().endswith(".bvh"):
        bpy.ops.import_anim.bvh(filepath=path, update_scene_fps=True, update_scene_duration=True)
    else:
        bpy.ops.import_scene.fbx(filepath=path, automatic_bone_orientation=False)
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if not arms:
        raise ValueError("no armature in the file")
    return max(arms, key=lambda a: len(a.data.bones))


def pick_action(arm, take):
    actions = list(bpy.data.actions)
    current = arm.animation_data.action if arm.animation_data else None
    if take:
        named = [a for a in actions if take.lower() in a.name.lower()]
        if not named:
            raise ValueError(f"no action named like {take!r} (actions: {[a.name for a in actions]})")
        current = named[0]
        if arm.animation_data is None:
            arm.animation_data_create()
        arm.animation_data.action = current
    if current is None:
        raise ValueError("the armature has no animation")
    return current


def rest_point(arm, spec):
    kind = spec["point"][0]
    mw = arm.matrix_world
    bones = arm.data.bones
    if kind == "head":
        return mw @ bones[spec["point"][1]].head_local
    if kind == "tail":
        return mw @ bones[spec["point"][1]].tail_local
    a, b, t = spec["point"][1:]
    return (mw @ bones[a].head_local).lerp(mw @ bones[b].head_local, t)


def world_rot(arm, matrix):
    return (arm.matrix_world @ matrix).to_3x3().normalized().to_quaternion()


def bake(item):
    arm = load(item["input"])
    action = pick_action(arm, item.get("take"))
    mapping = pose_math.map_skeleton([(b.name, b.parent.name if b.parent else None) for b in arm.data.bones])
    specs = mapping["joints"]
    real = [j for j in pose_math.NAMES if specs[j]["point"][0] != "offset"]
    bones = sorted({s["edge"] for s in specs.values()} | {s["carrier"] for s in specs.values()})
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    start, end = action.frame_range
    first, last = int(math.floor(start + 1e-6)), int(math.ceil(end - 1e-6))
    mw = arm.matrix_world
    # Some rigs animate a bone above the pelvis with a turn and a scale (a root bone keyed at 90
    # degrees and x100), so the armature's rest pose is not in the space the frames are. The rest
    # pose is moved by that bone's transform at the first frame, which puts both in one space.
    above = arm.data.bones[specs["pelvis"]["carrier"]].parent
    scene.frame_set(first)
    if above is not None:
        fix = (mw @ arm.pose.bones[above.name].matrix) @ (mw @ above.matrix_local).inverted()
    else:
        fix = mathutils.Matrix.Identity(4)
    rest_pts = {j: fix @ rest_point(arm, specs[j]) for j in real}
    # A point rides on its carrier bone: world(t) = M_carrier(t) M_carrier(rest)^-1 p_rest.
    rest_inv = {b: (fix @ mw @ arm.data.bones[b].matrix_local).inverted() for b in bones}
    rest_rot = [list((fix @ mw @ arm.data.bones[b].matrix_local).to_3x3().normalized().to_quaternion())
                for b in bones]
    points, rots = [], []
    for f in range(first, last + 1):
        scene.frame_set(f)
        pb = arm.pose.bones
        mats = {b: mw @ pb[b].matrix for b in bones}
        points.append([list(mats[specs[j]["carrier"]] @ (rest_inv[specs[j]["carrier"]] @ rest_pts[j])) for j in real])
        rots.append([list(world_rot(arm, pb[b].matrix)) for b in bones])
    raw = {"fps": fps, "joints": specs, "rest_points": {j: list(rest_pts[j]) for j in real}, "points": points,
           "bones": bones, "rest_rotations": rest_rot, "rotations": rots, "joint_bones": mapping["bones"]}
    source = {"file": os.path.basename(item["input"]), "action": action.name, "naming": mapping["naming"],
              "frame_range": [first, last]}
    clip = pose_math.build_clip(raw, name=item["name"], action=item.get("action"), loop=item.get("loop"),
                                weapon_hand=item.get("weapon_hand"), root_motion=item.get("root_motion", "in_place"),
                                source=source)
    with open(item["output"], "w") as fh:
        json.dump(clip, fh, separators=(",", ":"))
    meta = clip["metadata"]
    return {"name": item["name"], "path": item["output"], "frames": clip["frame_count"], "fps": clip["fps"],
            "naming": mapping["naming"], "action": meta["action"], "loop": meta["loop"],
            "impact_frame": meta["impact_frame"], "weapon_hand": meta["weapon_hand"],
            "key_frames": meta["key_frames"], "stature": meta["stature"], "root_travel": meta["root_travel"],
            "source_file": source["file"], "joint_bones": mapping["bones"]}


def main():
    spec = json.loads(sys.argv[sys.argv.index("--") + 1])
    t0 = time.time()
    results = []
    for item in spec["clips"]:
        try:
            results.append({"ok": True, **bake(item)})
        except Exception as exc:  # one bad file must not stop the batch; the report names it
            traceback.print_exc()
            results.append({"ok": False, "name": item["name"], "input": item["input"],
                            "error": str(exc)[:500]})
    report = {"clips": results, "seconds": round(time.time() - t0, 2)}
    with open(spec["report"], "w") as fh:
        json.dump(report, fh)
    print("POSE_BAKE_RESULT " + json.dumps(report), flush=True)


main()
