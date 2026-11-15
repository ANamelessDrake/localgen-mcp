"""Test fixture, run inside Blender: write a textured GLB with a known base colour.

    blender -b --factory-startup -P tests/blender_make_textured.py -- OUT.glb

A subdivided cube (about 6k faces) with an image texture whose every texel is
sRGB (200, 60, 30), plus a thin bar so the shape is not a plain blob.
"""
import sys

import bpy
import numpy as np

out = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=1)
cube = bpy.context.active_object
mod = cube.modifiers.new("sub", "SUBSURF")
mod.levels = 5
bpy.ops.object.modifier_apply(modifier=mod.name)
bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=1.4, location=(0, 0, 0.6))
bar = bpy.context.active_object
cube.select_set(True)
bpy.context.view_layer.objects.active = cube
bpy.ops.object.join()
obj = bpy.context.active_object
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project()
bpy.ops.object.mode_set(mode="OBJECT")

img = bpy.data.images.new("albedo", 64, 64)
img.pixels.foreach_set(np.tile([200 / 255, 60 / 255, 30 / 255, 1.0], 64 * 64).astype(np.float32))
img.update()
img.pack()
mat = bpy.data.materials.new("m")
mat.use_nodes = True
tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
tex.image = img
mat.node_tree.links.new(tex.outputs["Color"], mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
obj.data.materials.clear()
obj.data.materials.append(mat)
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB")
print("FIXTURE_FACES", len(obj.data.polygons))
