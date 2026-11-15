"""Test helper, run inside Blender: import a LOD chain (GLB or FBX) and describe it as JSON.

    blender -b --factory-startup -P tests/blender_inspect_lods.py -- IN.glb|IN.fbx

Prints "LODS {json}": per mesh object its name, face and triangle counts, materials,
and (for every object but the first LOD) how far its UVs sit from LOD0's UVs at the
nearest point of LOD0's surface, in UV units. This is measured on the exported file,
independently of the retopology script's own report.
"""
import json
import sys

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform

src = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
if src.lower().endswith(".fbx"):
    bpy.ops.import_scene.fbx(filepath=src)
else:
    bpy.ops.import_scene.gltf(filepath=src)
objs = sorted((o for o in bpy.context.scene.objects if o.type == "MESH"), key=lambda o: o.name)


def world_tris(obj):
    me = obj.data
    me.calc_loop_triangles()
    co = [obj.matrix_world @ v.co for v in me.vertices]
    uv = me.uv_layers.active.data
    return [([co[i] for i in t.vertices], [uv[k].uv.copy() for k in t.loops]) for t in me.loop_triangles]


lod0 = next((o for o in objs if o.name.endswith("_LOD0")), None)
base = world_tris(lod0) if lod0 is not None else None
if base is not None:
    pts = [p for tri, _ in base for p in tri]
    tree = BVHTree.FromPolygons(pts, [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(base))])

out = []
for o in objs:
    me = o.data
    entry = {"name": o.name, "faces": len(me.polygons), "triangles": sum(len(p.vertices) - 2 for p in me.polygons),
             "materials": [s.material.name if s.material else None for s in o.material_slots],
             "uv_layers": [u.name for u in me.uv_layers]}
    if base is not None and o is not lod0 and me.uv_layers:
        errors = []
        for tri, uv in world_tris(o):
            p = (tri[0] + tri[1] + tri[2]) / 3
            loc, _n, idx, _d = tree.find_nearest(p)
            btri, buv = base[idx]
            at = barycentric_transform(loc, *btri, *(u.to_3d() for u in buv))
            errors.append((at.xy - (uv[0] + uv[1] + uv[2]) / 3).length)
        errors = np.array(errors)
        entry["uv_error_median"] = float(np.median(errors))
        entry["uv_error_p95"] = float(np.percentile(errors, 95))
    out.append(entry)
images = sorted(i.name for i in bpy.data.images if i.users)
print("LODS " + json.dumps({"objects": out, "materials": sorted(m.name for m in bpy.data.materials if m.users),
                            "images": images}), flush=True)
