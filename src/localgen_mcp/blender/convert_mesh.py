"""Re-serialise a 3D model as a fresh GLB (geometry, UVs, materials and textures only).

    blender -b --factory-startup -P convert_mesh.py -- IN OUT.glb

IN may be GLB, glTF, OBJ, STL, PLY or FBX. Only what Blender imported as a
mesh is written out, so the uploaded file never carries bytes from the input
beyond the model itself. The last stdout line is a JSON report starting with
"CONVERT_RESULT ".
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
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", use_selection=True, export_image_format="AUTO",
                          export_extras=False, export_cameras=False, export_lights=False)
print("CONVERT_RESULT " + json.dumps({"input": src, "output": dst, "objects": len(meshes),
                                      "faces": sum(len(o.data.polygons) for o in meshes)}), flush=True)
