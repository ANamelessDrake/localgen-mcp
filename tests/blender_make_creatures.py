"""Test fixture, run inside Blender: synthetic creatures for rig_creature.

    blender -b --factory-startup -P tests/blender_make_creatures.py -- OUT_DIR

tentacle.glb: a closed tube along +X from x = 0 to 2, radius 0.15 at x = 0
tapering to 0.05 at x = 2 (the thick end is the base).
manta.glb: a flat diamond, wingspan 2 along X (tips at x = +-1), body length
1.2 along Y (head at y = +0.6, tail at y = -0.6), 0.06 thick in Z, closed.
shelled.glb: tentacle.glb with a second closed tube inside it at half the
radius; the inner shell hides the bones from the outer one, so Blender's heat
weights put most of the outer shell on a far bone.
nested.glb: tentacle.glb with a closed sphere inside another beside it (like
an eye in its socket); no vertex of the inner sphere can see a bone, Blender's
heat solver fails ("failed to find solution") and leaves every vertex
unweighted, as it does on real generated meshes.
"""
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix

out = sys.argv[sys.argv.index("--") + 1]


def export(obj, path):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True)


def new_object(name, bm):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    return o


def solidify(o, thickness):
    mod = o.modifiers.new("solid", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = 0.0
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.modifier_apply(modifier=mod.name)
    return o


def tube(bm, scale=1.0):
    rings, seg = 41, 16
    rows = []
    for i in range(rings):
        x = 2.0 * i / (rings - 1)
        rad = (0.15 - 0.1 * x / 2.0) * scale
        rows.append([bm.verts.new((x, rad * math.cos(2 * math.pi * j / seg), rad * math.sin(2 * math.pi * j / seg)))
                     for j in range(seg)])
    for i in range(rings - 1):
        for j in range(seg):
            bm.faces.new((rows[i][j], rows[i][(j + 1) % seg], rows[i + 1][(j + 1) % seg], rows[i + 1][j]))
    bm.faces.new(list(reversed(rows[0])))
    bm.faces.new(rows[-1])


def tentacle(inner=False, nested=False):
    bm = bmesh.new()
    tube(bm)
    if inner:
        tube(bm, 0.5)
    if nested:
        for radius in (0.1, 0.05):
            bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=radius,
                                      matrix=Matrix.Translation((1.0, 0.35, 0.0)))
    return new_object("tentacle", bm)


def diamond(name):
    bm = bmesh.new()
    n = 24
    grid = {}
    for i in range(n + 1):
        for j in range(n + 1):
            u, v = 2 * i / n - 1, 2 * j / n - 1
            grid[i, j] = bm.verts.new(((u + v) / 2, (v - u) / 2 * 0.6, 0.0))
    for i in range(n):
        for j in range(n):
            bm.faces.new((grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]))
    return solidify(new_object(name, bm), 0.06)


for name, make in (("tentacle", tentacle), ("manta", lambda: diamond("manta")),
                   ("shelled", lambda: tentacle(inner=True)),
                   ("nested", lambda: tentacle(nested=True))):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    export(make(), os.path.join(out, name + ".glb"))
