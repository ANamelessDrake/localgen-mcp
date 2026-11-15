"""Add bump (as a tangent space normal map), roughness and metallic to a textured GLB.

    blender -b --factory-startup -P texture_maps.py -- IN.glb OUT.glb BUMP ROUGHNESS METALLIC

The base color and occlusion maps are kept as they are. The normal map is baked
from a height signal (the base color's luminance through a Bump node, BUMP is
its strength, 0 for none) on the mesh itself, so painted seams, scuffs and
panel lines read as relief. Roughness is ROUGHNESS, varied by up to 0.15 with
the luminance (darker, worn areas rougher); metallic is the constant METALLIC.
An FBX with embedded textures is written beside the GLB. The last stdout line
is a JSON report starting with "TEXTURE_MAPS_RESULT ".

Runs inside Blender's own Python (bpy); it is not imported by the MCP server.
"""
import json
import os
import sys
import time

import bpy

t0 = time.time()
argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]
bump, roughness, metallic = float(argv[2]), float(argv[3]), float(argv[4])
report = {"input": src, "output": dst, "bump": bump, "roughness": roughness, "metallic": metallic}

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
mat = obj.active_material
nt = mat.node_tree
bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output)
base = next((l.from_node for l in bsdf.inputs["Base Color"].links if l.from_node.type == "TEX_IMAGE"), None)
if base is None:
    raise SystemExit("the mesh has no base color texture to derive maps from")
size = tuple(base.image.size)
report["texture_size"] = list(size)


def target(name):
    img = bpy.data.images.new(name, size[0], size[1], alpha=False, float_buffer=False)
    img.colorspace_settings.name = "Non-Color"  # raw values in and out, whatever the OCIO setup
    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = img
    return img, node


# Height from the base color's luminance.
lum = nt.nodes.new("ShaderNodeRGBToBW")
nt.links.new(base.outputs["Color"], lum.inputs["Color"])

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 4
scene.render.bake.use_selected_to_active = False
scene.render.bake.margin = 8
surface = [l.from_socket for l in out.inputs["Surface"].links]

# Roughness: the base value, rougher (by up to 0.15) where the paint is darker.
rough_img, rough_node = target("roughness")
rmap = nt.nodes.new("ShaderNodeMapRange")
rmap.inputs["From Min"].default_value, rmap.inputs["From Max"].default_value = 0.0, 1.0
rmap.inputs["To Min"].default_value = min(1.0, roughness + 0.15)
rmap.inputs["To Max"].default_value = max(0.0, roughness - 0.05)
nt.links.new(lum.outputs["Val"], rmap.inputs["Value"])
emit = nt.nodes.new("ShaderNodeEmission")
nt.links.new(rmap.outputs["Result"], emit.inputs["Color"])
nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
nt.nodes.active = rough_node
t = time.time()
bpy.ops.object.bake(type="EMIT")
report["roughness_bake_seconds"] = round(time.time() - t, 1)
nt.nodes.remove(emit)
for s in surface:
    nt.links.new(s, out.inputs["Surface"])

# Normal map from the bump, baked on the mesh itself.
normal_img = None
if bump > 0:
    normal_img, normal_node = target("normal")
    bump_node = nt.nodes.new("ShaderNodeBump")
    bump_node.inputs["Strength"].default_value = min(bump, 1.0)
    bump_node.inputs["Distance"].default_value = 0.002 * max(obj.dimensions)
    nt.links.new(lum.outputs["Val"], bump_node.inputs["Height"])
    old_normal = [l.from_socket for l in bsdf.inputs["Normal"].links]
    if old_normal:
        nt.links.new(old_normal[0], bump_node.inputs["Normal"])
    nt.links.new(bump_node.outputs["Normal"], bsdf.inputs["Normal"])
    nt.nodes.active = normal_node
    t = time.time()
    bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT")
    report["normal_bake_seconds"] = round(time.time() - t, 1)
    nt.nodes.remove(bump_node)
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(normal_node.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

nt.nodes.remove(rmap)
nt.nodes.remove(lum)
nt.links.new(rough_node.outputs["Color"], bsdf.inputs["Roughness"])
for l in list(bsdf.inputs["Metallic"].links):
    nt.links.remove(l)
bsdf.inputs["Metallic"].default_value = metallic
for img in (rough_img, normal_img):
    if img is not None:
        img.pack()

bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", use_selection=True, export_image_format="AUTO")
fbx = os.path.splitext(dst)[0] + ".fbx"
bpy.ops.export_scene.fbx(filepath=fbx, use_selection=True, path_mode="COPY", embed_textures=True,
                         mesh_smooth_type="FACE", apply_scale_options="FBX_SCALE_ALL")
report["fbx"] = fbx
report["faces"] = len(obj.data.polygons)
report["seconds"] = round(time.time() - t0, 1)
print("TEXTURE_MAPS_RESULT " + json.dumps(report), flush=True)
