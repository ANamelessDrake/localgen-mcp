"""Test fixture, run inside Blender: a two-part model with a known answer.

    blender -b --factory-startup -P tests/blender_make_parts.py -- OUT_DIR

Writes concept.png (a front view drawn to match), full.glb (body and arm together)
and body.glb / arm.glb (the parts at the wrong scale and position, as a part
generator would return them). True placement, in Blender coordinates (Z up,
front facing -Y): body x -0.25..0.25, y -0.2..0.2, z 0..1; arm x 0.25..0.45,
y -0.1..0.1, z 0.4..0.9 (the shoulder is the face at x = 0.25).
"""
import os
import sys

import bpy
import numpy as np

out = sys.argv[sys.argv.index("--") + 1]
BODY = ((-0.25, 0.25), (-0.2, 0.2), (0.0, 1.0))
ARM = ((0.25, 0.45), (-0.1, 0.1), (0.4, 0.9))


def box(name, lim, scale=1.0, offset=(0, 0, 0)):
    (x0, x1), (y0, y1), (z0, z1) = lim
    bpy.ops.mesh.primitive_cube_add(size=1)
    o = bpy.context.active_object
    o.name = name
    sub = o.modifiers.new("s", "SUBSURF")
    sub.subdivision_type = "SIMPLE"
    sub.levels = 3
    bpy.ops.object.modifier_apply(modifier=sub.name)
    o.scale = ((x1 - x0) * scale, (y1 - y0) * scale, (z1 - z0) * scale)
    o.location = ((x0 + x1) / 2 * scale + offset[0], (y0 + y1) / 2 * scale + offset[1],
                  (z0 + z1) / 2 * scale + offset[2])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return o


def export(objs, path):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True)


bpy.ops.wm.read_factory_settings(use_empty=True)
export([box("body", BODY), box("arm", ARM)], os.path.join(out, "full.glb"))
bpy.ops.wm.read_factory_settings(use_empty=True)
export([box("body", BODY, 3.0, (5, 2, -1))], os.path.join(out, "body.glb"))
bpy.ops.wm.read_factory_settings(use_empty=True)
export([box("arm", ARM, 0.4, (-2, 1, 3))], os.path.join(out, "arm.glb"))

# Front view, 400 px per unit: px = 100 + (x + 0.25) * 400, py = 50 + (1 - z) * 400.
W = H = 512
img = np.full((H, W, 4), (0.8, 0.8, 0.8, 1.0), dtype=np.float32)
for (x0, x1), _, (z0, z1) in (BODY, ARM):
    img[int(50 + (1 - z1) * 400):int(50 + (1 - z0) * 400), int(100 + (x0 + 0.25) * 400):int(100 + (x1 + 0.25) * 400), :3] = 0.2
im = bpy.data.images.new("concept", W, H)
im.pixels.foreach_set(img[::-1].ravel())
im.filepath_raw = os.path.join(out, "concept.png")
im.file_format = "PNG"
im.save()
