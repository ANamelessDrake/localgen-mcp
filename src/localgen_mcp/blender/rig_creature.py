"""Soft rig for an organic creature: bone chains with smooth skin weights.

    blender -b --factory-startup -P rig_creature.py -- SPEC_JSON

SPEC_JSON (a JSON string): {"input": GLB, "out_glb": path, "out_fbx": path,
"preset": "auto" | "chain" | "spine" | "wings", "bones": int or null,
"axis": "x" | "y" | "z" | [x, y, z] or null, "root_end": "auto" | "min" | "max" |
[x, y, z] or null, "wing_chains": int, "wing_bones": int, "weights": "auto" |
"distance", "name": bone name prefix for chain or null}

Frame. The mesh's principal axes (PCA of the vertices, snapped to the world axes
when all three are within 15 degrees of them, as generated meshes usually are)
give its length, width and thickness. 'auto' picks 'chain' for a long thin mesh,
'wings' for a flat broad one and 'spine' otherwise.

Chains. A chain's joints are the centroids of slices across its axis, so the
bones follow a gently curved mesh. 'wings' finds the thin axis (the surface
normal), then the left-right axis as the in-plane axis the mesh is mirror
symmetric across (the wider one on a tie), runs a spine along the body length
through the central strip, and fans wing_chains chains per side from the strip
out to the wing edge, each sector holding an equal share of the wing's vertices.

Bone axes. Every bone's local Y runs along it; wing and spine bones of 'wings'
have local Z along the surface normal, so rotating about local X flaps up and
down. Chain bones have local Z towards world +Z (world +Y for a vertical chain).

Weights. 'auto' tries Blender's heat weights (parent with automatic weights),
which often fail on generated, non manifold meshes. If more than 1% of the
vertices get (almost) no weight, or have their largest weight on a bone more
than twice as far as their nearest bone (heat "succeeding" with nonsense),
every weight is recomputed with a distance falloff from the bone segments
(1 / (d + eps)^4, eps 1% of the mesh diagonal); a smaller share is filled
with it. Weights are averaged over vertices at the same position (glTF splits
vertices at UV seams, which would tear otherwise), cut to the 4 largest per
vertex and normalised.

The last stdout line is a JSON report starting with "RIG_RESULT ".
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bpy_util  # noqa: E402

spec = json.loads(sys.argv[sys.argv.index("--") + 1])
MAX_INFLUENCES = 4
HEAT_GAP_LIMIT = 0.01  # share of unweighted or misassigned vertices above which heat weights are discarded
report = {"input": spec["input"], "preset_requested": spec["preset"], "warnings": []}

bpy.ops.wm.read_factory_settings(use_empty=True)
obj = _bpy_util.import_glb(spec["input"], spec.get("mesh_name") or "creature")
obj.vertex_groups.clear()
for m in [m for m in obj.modifiers if m.type == "ARMATURE"]:
    obj.modifiers.remove(m)
V = _bpy_util.verts(obj)
report["vertices"] = len(V)
lo, hi = V.min(axis=0), V.max(axis=0)
diag = float(np.linalg.norm(hi - lo))
WORLD = np.eye(3)


def r(v):
    return [round(float(c), 4) for c in v]


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def positive(a):
    """Sign convention: the largest component points the positive way."""
    return a if a[np.argmax(np.abs(a))] > 0 else -a


def principal_axes(P):
    evals, evecs = np.linalg.eigh(np.cov((P - P.mean(axis=0)).T))
    axes = [positive(evecs[:, i]) for i in (2, 1, 0)]  # largest spread first
    idx = [int(np.argmax(np.abs(a))) for a in axes]
    if len(set(idx)) == 3 and all(abs(a[i]) > math.cos(math.radians(15)) for a, i in zip(axes, idx)):
        axes = [WORLD[i].copy() for i in idx]
    axes = sorted(axes, key=lambda a: -np.ptp(P @ a))
    return axes


def parse_axis(a):
    if a is None:
        return None
    if isinstance(a, str):
        return WORLD["xyz".index(a.lower())].copy()
    return positive(unit(a))


AXES = principal_axes(V)
EXT = [float(np.ptp(V @ a)) for a in AXES]
report["extents"] = [round(e, 4) for e in EXT]
report["principal_axes"] = [r(a) for a in AXES]


def choose_preset():
    e1, e2, e3 = EXT
    if e1 >= 3 * e2:
        return "chain", f"long and thin (length {e1 / e2:.1f} times the width)"
    if e3 <= 0.6 * e2 and e2 >= 0.4 * e1:
        return "wings", f"flat and broad (thickness {e3 / e2:.2f} of the width, width {e2 / e1:.2f} of the length)"
    return "spine", f"compact body (proportions {e1:.2f} : {e2:.2f} : {e3:.2f})"


preset = spec["preset"]
if preset == "auto":
    preset, why = choose_preset()
    report["auto_reason"] = why
report["preset"] = preset


# -- chains ----------------------------------------------------------------


def slice_joints(P, a, n):
    """n + 1 joints from the min to the max end of P along a: slice centroids at evenly spaced stations,
    each moved onto its station exactly."""
    t = P @ a
    t0, t1 = float(t.min()), float(t.max())
    step = (t1 - t0) / n
    joints = []
    for k in np.linspace(t0, t1, n + 1):
        half = 0.5 * step
        sel = np.abs(t - k) <= half
        while sel.sum() < 3 and half < (t1 - t0):
            half *= 2
            sel = np.abs(t - k) <= half
        c = P[sel].mean(axis=0)
        joints.append(c + (k - c @ a) * a)
    return np.array(joints)


def end_radius(P, a, at_max):
    """Mean distance from the axis of the vertices in the last 15% of the length at one end."""
    t = P @ a
    t0, t1 = t.min(), t.max()
    sel = t >= t1 - 0.15 * (t1 - t0) if at_max else t <= t0 + 0.15 * (t1 - t0)
    Q = P[sel]
    c = Q.mean(axis=0)
    off = (Q - c) - np.outer((Q - c) @ a, a)
    return float(np.linalg.norm(off, axis=1).mean())


def root_is_max(P, a, root_end, default_rule):
    """Whether the chain starts at the max end of P along a, and why."""
    if isinstance(root_end, list):
        p = np.array(root_end, dtype=np.float64)
        t = P @ a
        return bool(abs(t.max() - p @ a) < abs(t.min() - p @ a)), "the end nearest the given point"
    if root_end in ("min", "max"):
        return root_end == "max", f"the {root_end} end along the axis, as given"
    if default_rule == "low" and abs(a[2]) >= 0.7:
        return bool(a[2] < 0), "the lower end (a vertical chain grows from its base)"
    rmin, rmax = end_radius(P, a, False), end_radius(P, a, True)
    if abs(rmax - rmin) <= 0.05 * max(rmin, rmax, 1e-9):
        return False, "the min end along the axis (both ends are equally thick)"
    return rmax > rmin, "the thicker end (the base of a tapering limb, the head of a body)"


def chain(P, a, n, root_end, rule):
    at_max, why = root_is_max(P, a, root_end, rule)
    j = slice_joints(P, a, n)
    return (j[::-1] if at_max else j), why


bones = []  # dicts: name, head, tail, parent (name or None), z (roll reference), connect


def add_chain(prefix, joints, parent, zref):
    names = []
    for i in range(len(joints) - 1):
        name = f"{prefix}_{i + 1:02d}"
        bones.append({"name": name, "head": joints[i], "tail": joints[i + 1], "z": zref,
                      "parent": names[-1] if names else parent, "connect": bool(names)})
        names.append(name)
    return names


def segment_distance(p, h, t):
    d = t - h
    u = np.clip((p - h) @ d / max(d @ d, 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - h - u * d))


def chain_zref(a):
    up = WORLD[2] if abs(a[2]) < 0.9 else WORLD[1]
    return unit(up - (up @ a) * a)


if preset in ("chain", "spine"):
    a = parse_axis(spec.get("axis"))
    a = AXES[0] if a is None else a
    n = spec.get("bones") or (8 if preset == "chain" else 6)
    prefix = spec.get("name") or ("tentacle" if preset == "chain" else "spine")
    joints, why = chain(V, a, n, spec.get("root_end"), "low" if preset == "chain" else "thick")
    add_chain(prefix, joints, None, chain_zref(a))
    report["axis"] = r(a)
    report["root"] = {"position": r(joints[0]), "reason": why}

else:  # wings
    N = AXES[2]  # thinnest: the surface normal
    given = parse_axis(spec.get("axis"))
    if given is not None:
        L = unit(given - (given @ N) * N)
        S = positive(unit(np.cross(N, L)))
        report["lateral_reason"] = "perpendicular to the given body axis"
    else:
        tree = KDTree(len(V))
        for i, co in enumerate(V):
            tree.insert(co, i)
        tree.balance()
        sample = V[np.random.default_rng(0).choice(len(V), min(3000, len(V)), replace=False)]

        def asymmetry(s):
            mid = (V @ s).min() + np.ptp(V @ s) / 2
            mirrored = sample - 2 * np.outer(sample @ s - mid, s)
            return float(np.mean([tree.find(co)[2] for co in mirrored])) / diag

        p1, p2 = AXES[0], AXES[1]
        a1, a2 = asymmetry(p1), asymmetry(p2)
        report["mirror_asymmetry"] = [{"axis": r(p1), "asymmetry": round(a1, 4)},
                                      {"axis": r(p2), "asymmetry": round(a2, 4)}]
        if abs(a1 - a2) <= 0.2 * max(a1, a2) + 0.002:
            S, L = p1, p2  # equally symmetric both ways: the wider one is the wingspan
            report["lateral_reason"] = "symmetric both ways; the wingspan is the wider axis"
        elif a1 < a2:
            S, L = p1, p2
            report["lateral_reason"] = "the axis the mesh is mirror symmetric across"
        else:
            S, L = p2, p1
            report["lateral_reason"] = "the axis the mesh is mirror symmetric across"
    s = V @ S
    s_mid = float(s.min() + np.ptp(s) / 2)
    half_span = float(np.ptp(s) / 2)
    strip = 0.12 * half_span
    core = V[np.abs(s - s_mid) <= strip]
    if len(core) < 10:
        core = V[np.argsort(np.abs(s - s_mid))[:max(10, len(V) // 20)]]
    n_spine = spec.get("bones") or 6
    t = V @ L
    tl0, tl1 = float(t.min()), float(t.max())
    width_min = float(np.ptp(s[t <= tl0 + 0.2 * (tl1 - tl0)]))
    width_max = float(np.ptp(s[t >= tl1 - 0.2 * (tl1 - tl0)]))
    root_end = spec.get("root_end")
    if isinstance(root_end, list) or root_end in ("min", "max"):
        head_max, why = root_is_max(core, L, root_end, "thick")
    elif abs(width_max - width_min) <= 0.05 * max(width_max, width_min):
        head_max, why = False, "the min end along the body axis (both ends are equally wide)"
    else:
        head_max, why = width_max > width_min, "the wider end (the head; a tail is narrow)"
    sj = slice_joints(core, L, n_spine)
    sj = sj[::-1] if head_max else sj
    sj = sj + np.outer(s_mid - sj @ S, S)  # exactly on the mirror plane
    spine = add_chain("spine", sj, None, N)
    to_head = L if head_max else -L
    report["axis"] = r(-to_head)  # from the head towards the tail
    report["lateral_axis"] = r(S)
    report["normal_axis"] = r(N)
    report["root"] = {"position": r(sj[0]), "reason": why}
    centre = sj.mean(axis=0)
    K, M = int(spec["wing_chains"]), int(spec["wing_bones"])
    tree_n = KDTree(len(V))
    for i, co in enumerate(V):
        tree_n.insert(co, i)
    tree_n.balance()
    for side, sign in (("L", 1.0), ("R", -1.0)):
        w = sign * (s - s_mid)
        sel = w > strip
        if sel.sum() < 10 * K:
            report["warnings"].append(f"wing {side}: too few vertices outside the body strip; no wing chains")
            continue
        W = V[sel]
        u = (W - centre) @ to_head
        lat = sign * (W @ S - s_mid)
        ang = np.arctan2(u, lat)  # +90 degrees points at the head
        edges = np.quantile(ang, np.linspace(0, 1, K + 1))
        for k in range(K):
            # chain 1 is the sector nearest the head
            a0, a1 = edges[K - k - 1], edges[K - k]
            in_sec = (ang >= a0) & (ang <= a1)
            Q = W[in_sec]
            theta = float(np.median(ang[in_sec]))
            d = unit(math.cos(theta) * sign * S + math.sin(theta) * to_head)
            root = centre + d * (strip / max(abs(math.cos(theta)), 0.2))  # where the ray leaves the body strip
            reach = float(((Q - centre) @ d).max())
            start = float((root - centre) @ d)
            if reach <= start:
                report["warnings"].append(f"wing {side}{k + 1}: sector does not reach past the body strip")
                continue
            tip = root + d * (reach - start)
            pts = np.linspace(root, tip, M + 1)
            radius = 0.6 * (reach - start) / M
            for p in pts:  # sit each joint at the local mid thickness
                near = [idx for (_, idx, _) in tree_n.find_range(p, radius)]
                if near:
                    p += (float(np.median(V[near] @ N)) - p @ N) * N
            nearest = min(bones[:len(spine)], key=lambda b: segment_distance(root, b["head"], b["tail"]))
            add_chain(f"wing_{side}{k + 1}", pts, nearest["name"], N)


# -- armature --------------------------------------------------------------

arm_data = bpy.data.armatures.new("rig")
rig = bpy.data.objects.new("rig", arm_data)
bpy.context.scene.collection.objects.link(rig)
bpy.ops.object.select_all(action="DESELECT")
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
eb = {}
for b in bones:
    e = arm_data.edit_bones.new(b["name"])
    e.head, e.tail = Vector(b["head"]), Vector(b["tail"])
    e.align_roll(Vector(b["z"]))
    if b["parent"]:
        e.parent = eb[b["parent"]]
        e.use_connect = b["connect"]
    eb[b["name"]] = e
bpy.ops.object.mode_set(mode="OBJECT")
names = [b["name"] for b in bones]
heads = np.array([b["head"] for b in bones])
tails = np.array([b["tail"] for b in bones])


# -- weights ---------------------------------------------------------------


def read_weights():
    idx = {g.index: names.index(g.name) for g in obj.vertex_groups if g.name in names}
    Wt = np.zeros((len(V), len(names)))
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group in idx:
                Wt[v.index, idx[g.group]] = g.weight
    return Wt


def bone_distances(P):
    """Distance from every point to every bone segment, (len(P), bones)."""
    d = tails - heads
    l2 = np.maximum((d * d).sum(axis=1), 1e-12)
    out = np.empty((len(P), len(names)))
    for i in range(0, len(P), 20000):
        X = P[i:i + 20000, None, :] - heads[None]
        t = np.clip((X * d[None]).sum(axis=2) / l2, 0.0, 1.0)
        out[i:i + 20000] = np.linalg.norm(X - t[..., None] * d[None], axis=2)
    return out


def distance_weights(dist):
    """Inverse distance (power 4) to every bone segment, eps = 1% of the mesh diagonal, normalised."""
    w = 1.0 / (dist + 0.01 * diag) ** 4
    return w / w.sum(axis=1, keepdims=True)


mode = spec.get("weights", "auto")
heat = {}
if mode == "auto":
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        Wt = read_weights()
    except RuntimeError as exc:
        report["warnings"].append(f"heat weights raised: {exc}")
        Wt = np.zeros((len(V), len(names)))
    dist = bone_distances(V)
    gap = Wt.sum(axis=1) < 0.01
    # Heat can also "succeed" with nonsense, for example an outer shell whose view of the bones is blocked
    # by an inner one ends up on whichever bone it could see: a vertex whose main bone is much farther
    # away than its nearest bone counts as misassigned.
    main = dist[np.arange(len(V)), Wt.argmax(axis=1)]
    wrong = ~gap & (main > 2.0 * dist.min(axis=1) + 0.05 * diag)
    bad = gap | wrong
    heat = {"unweighted_vertices": int(gap.sum()), "unweighted_share": round(float(gap.mean()), 4),
            "misassigned_vertices": int(wrong.sum()), "misassigned_share": round(float(wrong.mean()), 4)}
    if bad.mean() > HEAT_GAP_LIMIT:
        method = (f"distance (heat weights left {gap.mean():.1%} of the vertices unweighted and "
                  f"{wrong.mean():.1%} on a far bone)")
        Wt = distance_weights(dist)
    elif bad.any():
        method = f"heat, {int(bad.sum())} unweighted or misassigned vertices given distance weights"
        Wt[bad] = distance_weights(dist[bad])
    else:
        method = "heat"
else:
    method = "distance"
    Wt = distance_weights(bone_distances(V))

# Same position, same weights (vertices split at UV seams or hard edges must move together).
key = np.round((V - lo) / (1e-5 * diag + 1e-12)).astype(np.int64)
_, inverse = np.unique(key, axis=0, return_inverse=True)
inverse = inverse.reshape(-1)
sums = np.zeros((inverse.max() + 1, Wt.shape[1]))
np.add.at(sums, inverse, Wt)
counts = np.bincount(inverse)
Wt = sums[inverse] / counts[inverse, None]
# At most 4 influences, normalised.
if Wt.shape[1] > MAX_INFLUENCES:
    cut = np.argsort(Wt, axis=1)[:, :-MAX_INFLUENCES]
    np.put_along_axis(Wt, cut, 0.0, axis=1)
Wt[Wt < 1e-4] = 0.0
total = Wt.sum(axis=1, keepdims=True)
Wt = np.divide(Wt, total, out=np.zeros_like(Wt), where=total > 0)

obj.vertex_groups.clear()
groups = [obj.vertex_groups.new(name=n) for n in names]
for b, g in enumerate(groups):
    nz = np.nonzero(Wt[:, b])[0]
    for i in nz:
        g.add([int(i)], float(Wt[i, b]), "REPLACE")
if not any(m.type == "ARMATURE" for m in obj.modifiers):
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
if obj.parent != rig:
    obj.parent = rig  # the rig sits at the origin unrotated, so no inverse is needed
for m in obj.modifiers:
    if m.type == "ARMATURE":
        m.object = rig

influences = (Wt > 0).sum(axis=1)
covered = np.abs(Wt.sum(axis=1) - 1.0) < 1e-3
per_bone = (Wt > 0.1).sum(axis=0)
report["weights"] = {
    "method": method, "heat": heat or None, "coverage": round(float(covered.mean()), 4),
    "unweighted_vertices": int((~covered).sum()), "max_influences": int(influences.max()),
    "mean_influences": round(float(influences.mean()), 2),
    "vertices_per_bone": {n: int(c) for n, c in zip(names, per_bone)},
}
idle = [n for n, c in zip(names, per_bone) if c == 0]
if idle:
    report["warnings"].append(f"bones with no vertex weighted above 0.1: {idle}")
report["bones"] = [{"name": b["name"], "parent": b["parent"], "head": r(b["head"]), "tail": r(b["tail"]),
                    "length": round(float(np.linalg.norm(b["tail"] - b["head"])), 4)} for b in bones]
report["bone_count"] = len(bones)
report["posing"] = ("Each bone's local Y runs along it. Rotate about local X to bend it towards its local Z: "
                    + ("the surface normal for 'wings' (flap up and down)." if preset == "wings" else
                       "world +Z (world +Y for a vertical chain)."))

_bpy_util.export_glb_fbx([obj, rig], spec["out_glb"], spec["out_fbx"], armature=True)
report["glb"], report["fbx"] = spec["out_glb"], spec["out_fbx"]
print("RIG_RESULT " + json.dumps(report), flush=True)
