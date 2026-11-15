"""Assemble separately generated part meshes into one model and give it a rigid rig.

    blender -b --factory-startup -P assemble_parts.py -- SPEC.json

SPEC: {"concept": image path, "full_mesh": GLB of the whole object (placement
reference), "parts": [{"name", "mesh", "box": [x0, y0, x1, y1] in concept pixels,
"parent": name or null}], "rig": bool, "out_glb": path, "out_fbx": path}

Placement. Every mesh comes from the same front view, so a part's box in the
concept maps to a box in the full mesh's front projection (X right, Z up; the
mesh faces -Y). Each part is scaled uniformly to fit its box and centred on it;
its depth (Y) is the centre of the full mesh's vertices inside that box.

Rig. Parts that touch are joined: the hierarchy is the given parents, else a
minimum spanning tree over the gaps between parts, rooted at the largest part.
A child's pivot is the centre of its vertices nearest its parent (the joint).
One bone per part, from its pivot towards its centre; each part mesh is
parented rigidly to its bone, so posing a bone moves exactly that part.
The last stdout line is a JSON report starting with "ASSEMBLE_RESULT ".
"""
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bpy_util  # noqa: E402

spec = json.load(open(sys.argv[sys.argv.index("--") + 1]))
report = {"parts": {}, "warnings": []}

bpy.ops.wm.read_factory_settings(use_empty=True)


import_glb, verts = _bpy_util.import_glb, _bpy_util.verts


def silhouette_box(path):
    """Pixel box of the object in the concept: pixels that differ from the border colour."""
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "Non-Color"
    w, h = img.size
    px = np.empty(w * h * img.channels, dtype=np.float32)
    img.pixels.foreach_get(px)
    rgb = px.reshape(h, w, img.channels)[::-1, :, :3]  # Blender stores rows bottom-up
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    mask = np.abs(rgb - bg).max(axis=2) > 0.12
    rows = np.nonzero(mask.mean(axis=1) > 0.004)[0]
    cols = np.nonzero(mask.mean(axis=0) > 0.004)[0]
    if not len(rows) or not len(cols):
        raise SystemExit("could not find the object in the concept image")
    return [int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1], (w, h)


# Image to world mapping from the full mesh and the concept's silhouette.
full = import_glb(spec["full_mesh"], "_full")
fv = verts(full)
(fx0, fy0, fx1, fy1), (W, H) = silhouette_box(spec["concept"])
mx0, mx1 = fv[:, 0].min(), fv[:, 0].max()
mz0, mz1 = fv[:, 2].min(), fv[:, 2].max()
sx, sz = (mx1 - mx0) / (fx1 - fx0), (mz1 - mz0) / (fy1 - fy0)
s = (sx * sz) ** 0.5
height = mz1 - mz0
report["concept_box"] = [fx0, fy0, fx1, fy1]
if abs(sx / sz - 1) > 0.15:
    report["warnings"].append(f"the full mesh and the concept silhouette disagree in aspect ({sx / sz:.2f}); "
                              "placement may be off")


def to_world(px, py):
    return mx0 + (px - fx0) * s, mz1 - (py - fy0) * s


parts = {}
for p in spec["parts"]:
    obj = import_glb(p["mesh"], p["name"])
    v = verts(obj)
    x0, y0, x1, y1 = p["box"]
    wx0, wz1 = to_world(x0, y0)
    wx1, wz0 = to_world(x1, y1)
    pw, ph = np.ptp(v[:, 0]), np.ptp(v[:, 2])
    k = float(((wx1 - wx0) / pw * (wz1 - wz0) / ph) ** 0.5)
    inside = fv[(fv[:, 0] >= wx0) & (fv[:, 0] <= wx1) & (fv[:, 2] >= wz0) & (fv[:, 2] <= wz1)]
    if len(inside) < 20:
        report["warnings"].append(f"{p['name']}: the full mesh has almost nothing inside its box; depth guessed")
        depth = float(np.median(fv[:, 1]))
    else:
        lo_y, hi_y = np.percentile(inside[:, 1], [10, 90])
        depth = float((lo_y + hi_y) / 2)
    centre = (v.min(axis=0) + v.max(axis=0)) / 2
    target = np.array([(wx0 + wx1) / 2, depth, (wz0 + wz1) / 2])
    obj.matrix_world = (Matrix.Translation(Vector(target)) @ Matrix.Scale(k, 4)
                        @ Matrix.Translation(Vector(-centre)))
    bpy.context.view_layer.update()
    pv = verts(obj)
    parts[p["name"]] = {"obj": obj, "v": pv, "parent": p.get("parent"),
                        "volume": float(np.prod(np.ptp(pv, axis=0)))}
    report["parts"][p["name"]] = {"faces": len(obj.data.polygons), "scale": round(k, 4)}
bpy.data.objects.remove(full, do_unlink=True)


def tree_of(v):
    t = KDTree(len(v))
    for i, co in enumerate(v):
        t.insert(co, i)
    t.balance()
    return t


trees = {n: tree_of(d["v"]) for n, d in parts.items()}
rng = np.random.default_rng(0)


def gap(a, b):
    """Smallest distance from part a's surface to part b's (sampled)."""
    va = parts[a]["v"]
    sample = va[rng.choice(len(va), min(3000, len(va)), replace=False)]
    return min(trees[b].find(co)[2] for co in sample)


names = list(parts)
root = max(names, key=lambda n: parts[n]["volume"])
given = {n: d["parent"] for n, d in parts.items() if d["parent"]}
for child, parent in given.items():
    if parent not in parts:
        raise SystemExit(f"part {child!r} has unknown parent {parent!r}")
parent_of = {root: None}
gaps = {}
# Prim's algorithm from the root over part-to-part gaps; given parents are honoured.
while len(parent_of) < len(names):
    best = None
    for child in names:
        if child in parent_of:
            continue
        candidates = [given[child]] if child in given else list(parent_of)
        for par in candidates:
            if par not in parent_of:
                continue
            key = (child, par)
            if key not in gaps:
                gaps[key] = gap(child, par)
            if best is None or gaps[key] < best[2]:
                best = (child, par, gaps[key])
    if best is None:  # a given parent is not attached yet; attach the rest first
        child = next(c for c in names if c not in parent_of and given.get(c) not in parent_of
                     and given.get(c) is not None)
        raise SystemExit(f"parent chain for {child!r} does not reach the root {root!r}")
    child, par, d = best
    parent_of[child] = par
    if d > 0.05 * height:
        report["warnings"].append(f"{child} is {d / height:.0%} of the model height away from {par}; "
                                  "check its box (or set its parent)")


def pivot(child, par):
    v = parts[child]["v"]
    sample = v[rng.choice(len(v), min(6000, len(v)), replace=False)]
    dist = np.array([trees[par].find(co)[2] for co in sample])
    near = sample[dist <= max(dist.min() * 1.5, dist.min() + 0.01 * height)]
    return near.mean(axis=0)


for name in names:
    v = parts[name]["v"]
    parts[name]["centre"] = (v.min(axis=0) + v.max(axis=0)) / 2
    parts[name]["pivot"] = parts[name]["centre"] if parent_of[name] is None else pivot(name, parent_of[name])

objects = [parts[n]["obj"] for n in names]
if spec.get("rig", True):
    arm_data = bpy.data.armatures.new("rig")
    rig = bpy.data.objects.new("rig", arm_data)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bones = {}
    order = [root] + [n for n in names if n != root]
    while order:  # parents before children
        for n in list(order):
            if parent_of[n] is None or parent_of[n] in bones:
                b = arm_data.edit_bones.new(n)
                head = Vector(parts[n]["pivot"])
                tail = Vector(parts[n]["centre"])
                if (tail - head).length < 0.03 * height:
                    away = head - Vector(parts[parent_of[n]]["centre"]) if parent_of[n] else Vector((0, 0, 1))
                    tail = head + (away.normalized() if away.length else Vector((0, 0, 1))) * 0.08 * height
                b.head, b.tail = head, tail
                if parent_of[n]:
                    b.parent = bones[parent_of[n]]
                bones[n] = b
                order.remove(n)
    bpy.ops.object.mode_set(mode="OBJECT")
    for n in names:
        obj = parts[n]["obj"]
        mw = obj.matrix_world.copy()
        obj.parent = rig
        obj.parent_type = "BONE"
        obj.parent_bone = n
        obj.matrix_world = mw
    objects.append(rig)

for n in names:
    v = parts[n]["v"]
    report["parts"][n].update({"parent": parent_of[n], "pivot": [round(float(c), 4) for c in parts[n]["pivot"]],
                               "bbox": [[round(float(c), 4) for c in v.min(axis=0)],
                                        [round(float(c), 4) for c in v.max(axis=0)]]})
report["root"] = root

_bpy_util.export_glb_fbx(objects, spec["out_glb"], spec["out_fbx"])
report["glb"], report["fbx"] = spec["out_glb"], spec["out_fbx"]
print("ASSEMBLE_RESULT " + json.dumps(report), flush=True)
