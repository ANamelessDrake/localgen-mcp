"""Test fixture, run inside Blender: a sphere whose +X half is painted white (full detail) and
-X half black, with a plain texture, exported as GLB with its vertex colours.

    blender -b --factory-startup -P tests/blender_make_painted.py -- OUT.glb
"""
import sys

import bpy
import numpy as np

out = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64)
obj = bpy.context.active_object
attr = obj.data.color_attributes.new("detail", "FLOAT_COLOR", "POINT")
for i, v in enumerate(obj.data.vertices):
    c = 1.0 if v.co.x > 0 else 0.0
    attr.data[i].color = (c, c, c, 1.0)
obj.data.color_attributes.active_color = attr
img = bpy.data.images.new("albedo", 16, 16)
img.pixels.foreach_set(np.tile([0.5, 0.5, 0.5, 1.0], 256).astype(np.float32))
img.update()
img.pack()
mat = bpy.data.materials.new("m")
mat.use_nodes = True
tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
tex.image = img
mat.node_tree.links.new(tex.outputs["Color"], mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
obj.data.materials.append(mat)
kw = {"export_vertex_color": "ACTIVE"} if "export_vertex_color" in bpy.ops.export_scene.gltf.get_rna_type().properties else {"export_colors": True}
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", **kw)
