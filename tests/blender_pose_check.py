"""Test helper, run inside Blender: load a rigged GLB or FBX, pose bones, report how the skin moved.

    blender -b --factory-startup -P tests/blender_pose_check.py -- MODEL OUT.json '{"bone": degrees, ...}'

Each named bone is rotated about its local X axis (rig_creature's bend axis).
Writes the armature's bones, the skinned mesh's vertex groups and weight coverage,
the rest positions of the vertices and how far each one moved, from the evaluated
depsgraph (the Armature modifier applied), in world space.
"""
import json
import math
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = argv[0], argv[1]
pose = json.loads(argv[2]) if len(argv) > 2 else {}

bpy.ops.wm.read_factory_settings(use_empty=True)
if src.lower().endswith(".fbx"):
    bpy.ops.import_scene.fbx(filepath=src)
else:
    bpy.ops.import_scene.gltf(filepath=src)
# the glTF importer adds a bone display shape in its own "glTF_not_exported" collection: not part of the model
model = [o for o in bpy.data.objects if not any(c.name.startswith("glTF_not_exported") for c in o.users_collection)]
arms = [o for o in model if o.type == "ARMATURE"]
meshes = [o for o in model if o.type == "MESH"]
res = {"armatures": len(arms), "meshes": len(meshes)}
if len(arms) != 1 or len(meshes) != 1:
    json.dump(res, open(out, "w"))
    raise SystemExit(0)
rig, obj = arms[0], meshes[0]
res["bones"] = {b.name: {"parent": b.parent.name if b.parent else None,
                         "head": list(rig.matrix_world @ b.head_local), "tail": list(rig.matrix_world @ b.tail_local)}
                for b in rig.data.bones}
res["armature_modifier"] = any(m.type == "ARMATURE" and m.object == rig for m in obj.modifiers)
res["groups"] = [g.name for g in obj.vertex_groups]
names = {g.index: g.name for g in obj.vertex_groups}
sums, counts = [], []
for v in obj.data.vertices:
    ws = [g.weight for g in v.groups if names.get(g.group) in res["bones"] and g.weight > 1e-6]
    sums.append(sum(ws))
    counts.append(len(ws))
res["weighted_share"] = sum(1 for s in sums if abs(s - 1) < 0.01) / len(sums)
res["max_influences"] = max(counts)


def world_verts():
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    co = [list(ev.matrix_world @ v.co) for v in me.vertices]
    ev.to_mesh_clear()
    return co


rest = world_verts()
for name, deg in pose.items():
    pb = rig.pose.bones[name]
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (math.radians(deg), 0, 0)
bpy.context.view_layer.update()
posed = world_verts()
res["rest"] = rest
res["moved"] = [math.dist(a, b) for a, b in zip(rest, posed)]
res["posed"] = posed
json.dump(res, open(out, "w"))
