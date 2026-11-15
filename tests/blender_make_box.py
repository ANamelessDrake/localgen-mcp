"""Test fixture, run inside Blender: a box with a nose on its front, written as a GLB.

    blender -b --factory-startup -P tests/blender_make_box.py -- OUT.glb [YAW_DEGREES]

The box is 1.0 wide (X), 0.3 deep (Y) and 0.8 tall (Z); the nose (0.2 x 0.2 x 0.2) sticks out of
the front face, at -Y, the way generated meshes face. So the whole object is 0.5 deep, and in the
rig's left view (the object's own left side) the nose points at the image's left. YAW turns the
whole object about Z (counterclockwise seen from above), to stand in for a mesh that does not face -Y.
"""
import math
import sys

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
out = args[0]
yaw = float(args[1]) if len(args) > 1 else 0.0
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=1)
box = bpy.context.active_object
box.scale = (1.0, 0.3, 0.8)
bpy.ops.mesh.primitive_cube_add(size=0.2, location=(0, -0.25, 0))
box.select_set(True)
bpy.context.view_layer.objects.active = box
bpy.ops.object.join()
box.rotation_euler = (0, 0, math.radians(yaw))
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
mat = bpy.data.materials.new("m")
mat.use_nodes = True
mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.6, 0.1, 0.05, 1)
bpy.context.active_object.data.materials.append(mat)
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB")
