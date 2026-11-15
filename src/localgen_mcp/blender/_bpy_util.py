"""Helpers shared by the bundled Blender scripts (runs inside Blender's Python).

A script imports it after putting its own folder on sys.path:

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _bpy_util
"""
import bpy
import numpy as np


def import_glb(path, name):
    """Import a GLB as one mesh object: every mesh joined, parents cleared and transforms applied.
    Everything else the file brought in (empties, cameras, armatures) is removed."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    new_names = [o.name for o in new]
    meshes = [o for o in new if o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"no mesh in {path}")
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for n in new_names:  # empties and other leftovers (joined meshes are already gone)
        o = bpy.data.objects.get(n)
        if o is not None and o != obj:
            bpy.data.objects.remove(o, do_unlink=True)
    obj.name = name
    return obj


def verts(obj):
    """World space vertex positions as an (N, 3) array."""
    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    mw = np.array(obj.matrix_world)
    return co @ mw[:3, :3].T + mw[:3, 3]


def export_glb_fbx(objects, glb, fbx, armature=False):
    """Write the given objects as a GLB and an FBX with embedded textures, rest pose, no animation."""
    bpy.ops.object.select_all(action="DESELECT")
    for o in objects:
        o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB", use_selection=True,
                              export_image_format="AUTO", export_animations=False)
    extra = {"use_armature_deform_only": True} if armature else {}
    bpy.ops.export_scene.fbx(filepath=fbx, use_selection=True, path_mode="COPY", embed_textures=True,
                             add_leaf_bones=False, bake_anim=False, apply_scale_options="FBX_SCALE_ALL", **extra)
