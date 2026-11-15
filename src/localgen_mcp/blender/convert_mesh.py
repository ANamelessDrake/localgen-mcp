"""Re-serialise a 3D model as a fresh GLB holding geometry and UVs only.

    blender -b --factory-startup -P convert_mesh.py -- IN OUT.glb

IN may be GLB, glTF, OBJ, STL, PLY or FBX. Every material and image is dropped
(texture_mesh repaints from scratch): model formats can name other files on
disk as textures (OBJ through its .mtl, FBX, glTF), and those must never reach
the output. glTF buffer URIs are refused before this script runs
(model_files.require_self_contained). The last stdout line is a JSON report
starting with "CONVERT_RESULT ".
"""
import json
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]
ext = os.path.splitext(src)[1].lower()
bpy.ops.wm.read_factory_settings(use_empty=True)
importers = {
    ".glb": lambda: bpy.ops.import_scene.gltf(filepath=src),
    ".gltf": lambda: bpy.ops.import_scene.gltf(filepath=src),
    ".obj": lambda: bpy.ops.wm.obj_import(filepath=src),
    ".stl": lambda: bpy.ops.wm.stl_import(filepath=src),
    ".ply": lambda: bpy.ops.wm.ply_import(filepath=src),
    ".fbx": lambda: bpy.ops.import_scene.fbx(filepath=src),
}
if ext not in importers:
    raise SystemExit(f"unsupported model type {ext}")
importers[ext]()
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and len(o.data.polygons)]
if not meshes:
    raise SystemExit("no mesh found in " + src)
for o in meshes:
    o.data.materials.clear()
for img in list(bpy.data.images):
    bpy.data.images.remove(img)
for mat in list(bpy.data.materials):
    bpy.data.materials.remove(mat)
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", use_selection=True, export_materials="NONE",
                          export_image_format="NONE", export_extras=False, export_cameras=False,
                          export_lights=False)
print("CONVERT_RESULT " + json.dumps({"input": src, "output": dst, "objects": len(meshes),
                                      "faces": sum(len(o.data.polygons) for o in meshes)}), flush=True)
