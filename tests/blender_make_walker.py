"""Test fixture, run inside Blender: a synthetic humanoid clip for bake_pose_library.

    blender -b --factory-startup -P tests/blender_make_walker.py -- OUT_DIR

Writes fixture_attack_R.fbx and fixture_attack_R.bvh: one armature with Unity Humanoid bone names
(Hips, Spine, Chest, Neck, Head, LeftShoulder, LeftUpperArm, LeftLowerArm, LeftHand,
LeftUpperLeg, LeftLowerLeg, LeftFoot, LeftToes and the Right ones) under a Root bone, plus a
twist bone the mapping must skip. Built facing -Y in metres, then the armature object is turned
90 degrees about Z, so the character faces +X in the file. 21 frames at 24 fps:
  - Root moves 1 m along the character's forward direction (root motion);
  - LeftLowerArm bends 90 degrees about its local X axis from frame 1 to frame 11 and holds;
  - RightUpperArm swings back to 100 degrees by frame 5, then forward to -90 at frame 11 (the
    fastest motion of the clip), then back to 0 at frame 21.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

out = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = 24
scene.frame_start, scene.frame_end = 1, 21

BONES = [  # name, parent, head, tail (metres, facing -Y, left at +X)
    ("Root", None, (0, 0, 0), (0, 0.2, 0)),
    ("Hips", "Root", (0, 0, 1.0), (0, 0, 1.1)),
    ("Spine", "Hips", (0, 0, 1.1), (0, 0, 1.3)),
    ("Chest", "Spine", (0, 0, 1.3), (0, 0, 1.48)),
    ("Neck", "Chest", (0, 0, 1.48), (0, 0, 1.56)),
    ("Head", "Neck", (0, 0, 1.56), (0, 0, 1.75)),
    ("LeftUpperLeg", "Hips", (0.1, 0, 0.95), (0.1, 0, 0.52)),
    ("LeftLowerLeg", "LeftUpperLeg", (0.1, 0, 0.52), (0.1, 0, 0.09)),
    ("LeftFoot", "LeftLowerLeg", (0.1, 0, 0.09), (0.1, -0.13, 0.02)),
    ("LeftToes", "LeftFoot", (0.1, -0.13, 0.02), (0.1, -0.2, 0.02)),
    ("LeftShoulder", "Chest", (0.03, 0, 1.44), (0.19, 0, 1.44)),
    ("LeftUpperArm", "LeftShoulder", (0.19, 0, 1.44), (0.19, 0, 1.16)),
    ("LeftUpperArmTwist", "LeftUpperArm", (0.19, 0, 1.3), (0.19, 0, 1.2)),
    ("LeftLowerArm", "LeftUpperArm", (0.19, 0, 1.16), (0.19, 0, 0.9)),
    ("LeftHand", "LeftLowerArm", (0.19, 0, 0.9), (0.19, 0, 0.8)),
]
for name, parent, head, tail in list(BONES):
    if name.startswith("Left"):
        mirror = lambda p: (-p[0], p[1], p[2])  # noqa: E731
        BONES.append((name.replace("Left", "Right"), parent.replace("Left", "Right") if parent else None,
                      mirror(head), mirror(tail)))

data = bpy.data.armatures.new("rig")
arm = bpy.data.objects.new("rig", data)
scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
for name, parent, head, tail in BONES:
    b = data.edit_bones.new(name)
    b.head, b.tail = Vector(head), Vector(tail)
    b.roll = 0.0
    if parent:
        b.parent = data.edit_bones[parent]
        b.use_connect = False
bpy.ops.object.mode_set(mode="POSE")
for pb in arm.pose.bones:
    pb.rotation_mode = "XYZ"


def key(bone, frame, rot=None, loc=None):
    pb = arm.pose.bones[bone]
    if rot is not None:
        pb.rotation_euler = rot
        pb.keyframe_insert("rotation_euler", frame=frame)
    if loc is not None:
        pb.location = loc
        pb.keyframe_insert("location", frame=frame)


# Root's local Y is world +Y (the bone points along it); the character walks towards -Y.
key("Root", 1, loc=(0, 0, 0))
key("Root", 21, loc=(0, -1.0, 0))
key("LeftLowerArm", 1, rot=(0, 0, 0))
key("LeftLowerArm", 11, rot=(math.radians(90), 0, 0))
for frame, deg in ((1, 0), (5, 100), (11, -90), (21, 0)):
    key("RightUpperArm", frame, rot=(math.radians(deg), 0, 0))
for fc in arm.animation_data.action.fcurves if hasattr(arm.animation_data.action, "fcurves") else []:
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"
bpy.ops.object.mode_set(mode="OBJECT")
arm.rotation_euler = (0, 0, math.radians(90))
bpy.context.view_layer.update()

arm.select_set(True)
base = os.path.join(out, "fixture_attack_R")
bpy.ops.export_scene.fbx(filepath=base + ".fbx", use_selection=True, add_leaf_bones=False, bake_anim=True,
                         bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False,
                         apply_scale_options="FBX_SCALE_ALL")
bpy.ops.export_anim.bvh(filepath=base + ".bvh", frame_start=1, frame_end=21, root_transform_only=False)
print("WALKER_DONE", base)
