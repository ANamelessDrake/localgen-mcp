"""Seam check, PBR maps and previews for a tileable base color (generate_material).

    blender -b --factory-startup -P material_maps.py -- '<json spec>'

spec: {"mode": "check" | "maps", "base": PNG path, optional "untiled" (a PNG to check too), and for "maps": "prefix" (output path
prefix), "normal_strength", "directx", "invert_height", "roughness", "roughness_variation",
"ao_strength", "preview" ("sphere" | "plane" | "none"), "preview_size"}.

"check" reports seam_metrics of the base color only. "maps" writes <prefix>_height.png
(16 bit), _normal_gl.png or _normal_dx.png, _roughness.png, _ao.png, _tile3x3.png and
_preview.png, and reports the seam metrics of the base color and of every map. The maths is
in pbr_maps.py (numpy). The last stdout line is "MATERIAL_MAPS_RESULT {json}".

Colour: the base color is loaded as Non-Color, so its pixels are its raw bytes whatever
OCIO setup Blender has; maps are written by pbr_maps.png_bytes, not by Blender, so their
bytes are exactly the computed values. For the preview render every map is loaded as
Non-Color and the base color is decoded from sRGB with a Gamma node.

Runs inside Blender's own Python (bpy); it is not imported by the MCP server.
"""
import json
import math
import os
import sys
import time

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbr_maps  # noqa: E402

t0 = time.time()
spec = json.loads(sys.argv[sys.argv.index("--") + 1])
report = {"mode": spec["mode"], "base": spec["base"]}

bpy.ops.wm.read_factory_settings(use_empty=True)


def load_rgb(path):
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "Non-Color"  # raw bytes, set on the file image before reading
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    rgb = px.reshape(h, w, 4)[::-1, :, :3]  # Blender stores the bottom row first
    return np.round(rgb * 255.0) / 255.0


rgb = load_rgb(spec["base"])
report["size"] = [rgb.shape[1], rgb.shape[0]]
report["seams"] = {"base_color": pbr_maps.seam_metrics(rgb)}
if spec.get("untiled"):  # the image before it was made to tile, for comparison
    report["seams"]["untiled"] = pbr_maps.seam_metrics(load_rgb(spec["untiled"]))

if spec["mode"] == "maps":
    prefix = spec["prefix"]
    directx = bool(spec.get("directx"))
    t = time.time()
    height = pbr_maps.height_from_albedo(rgb, invert=bool(spec.get("invert_height")))
    normal = pbr_maps.normal_from_height(height, float(spec.get("normal_strength", 1.0)), directx=directx)
    rough = pbr_maps.roughness_map(height, rgb, float(spec.get("roughness", 0.7)),
                                   float(spec.get("roughness_variation", 0.25)))
    ao = pbr_maps.ambient_occlusion(height, float(spec.get("ao_strength", 0.6)))
    report["maps_seconds"] = round(time.time() - t, 1)
    files = {
        "height": (prefix + "_height.png", pbr_maps.to_uint16(height)),
        "normal": (prefix + ("_normal_dx.png" if directx else "_normal_gl.png"), pbr_maps.to_uint8(normal)),
        "roughness": (prefix + "_roughness.png", pbr_maps.to_uint8(rough)),
        "ambient_occlusion": (prefix + "_ao.png", pbr_maps.to_uint8(ao)),
        "tile_3x3": (prefix + "_tile3x3.png", pbr_maps.to_uint8(pbr_maps.tile(rgb, 3, 2048))),
    }
    for path, arr in files.values():
        pbr_maps.write_png(path, arr)
    report["files"] = {k: v[0] for k, v in files.items()}
    report["normal_convention"] = "DirectX (-Y, green down)" if directx else "OpenGL (+Y, green up)"
    for key, arr in (("height", height), ("normal", normal), ("roughness", rough), ("ambient_occlusion", ao)):
        report["seams"][key] = pbr_maps.seam_metrics(arr)
    report["stats"] = {"height_mean": round(float(height.mean()), 3), "roughness_mean": round(float(rough.mean()), 3),
                       "roughness_range": [round(float(rough.min()), 3), round(float(rough.max()), 3)],
                       "ao_mean": round(float(ao.mean()), 3)}

    shape = spec.get("preview", "sphere")
    if shape != "none":
        t = time.time()
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = int(spec.get("preview_samples", 32))
        scene.view_settings.view_transform = "Standard"
        size = int(spec.get("preview_size", 768))
        scene.render.resolution_x, scene.render.resolution_y = size, size * 2 // 3

        mat = bpy.data.materials.new("material")
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes["Principled BSDF"]

        def tex(path, mapping):
            img = bpy.data.images.load(path)
            img.colorspace_settings.name = "Non-Color"
            n = nt.nodes.new("ShaderNodeTexImage")
            n.image = img
            n.extension = "REPEAT"
            nt.links.new(mapping, n.inputs["Vector"])
            return n.outputs["Color"]

        coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        tiles = float(spec.get("preview_tiles", 2.0))
        mapping.inputs["Scale"].default_value = (tiles, tiles if shape == "plane" else tiles / 2, 1.0)
        nt.links.new(coord.outputs["UV"], mapping.inputs["Vector"])
        uv = mapping.outputs["Vector"]
        gamma = nt.nodes.new("ShaderNodeGamma")  # sRGB bytes to linear, approximately
        gamma.inputs["Gamma"].default_value = 2.2
        nt.links.new(tex(spec["base"], uv), gamma.inputs["Color"])
        mix = nt.nodes.new("ShaderNodeMix")  # base color times AO
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        nt.links.new(gamma.outputs["Color"], mix.inputs[6])
        nt.links.new(tex(files["ambient_occlusion"][0], uv), mix.inputs[7])
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
        rough_sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(tex(files["roughness"][0], uv), rough_sep.inputs["Color"])
        nt.links.new(rough_sep.outputs[0], bsdf.inputs["Roughness"])
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        ncol = tex(files["normal"][0], uv)
        if directx:  # Blender reads OpenGL normal maps: flip green back for the preview
            sep = nt.nodes.new("ShaderNodeSeparateColor")
            inv = nt.nodes.new("ShaderNodeMath")
            inv.operation = "SUBTRACT"
            inv.inputs[0].default_value = 1.0
            comb = nt.nodes.new("ShaderNodeCombineColor")
            nt.links.new(ncol, sep.inputs["Color"])
            nt.links.new(sep.outputs[0], comb.inputs[0])
            nt.links.new(sep.outputs[1], inv.inputs[1])
            nt.links.new(inv.outputs[0], comb.inputs[1])
            nt.links.new(sep.outputs[2], comb.inputs[2])
            ncol = comb.outputs["Color"]
        nt.links.new(ncol, nmap.inputs["Color"])
        nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

        if shape == "plane":
            bpy.ops.mesh.primitive_plane_add(size=2.0)
            obj = bpy.context.object
            cam_loc, cam_rot = (0.0, -1.9, 1.25), (math.radians(58), 0.0, 0.0)
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=1.0)
            obj = bpy.context.object
            bpy.ops.object.shade_smooth()
            cam_loc, cam_rot = (0.0, -4.2, 0.9), (math.radians(78), 0.0, 0.0)
        obj.data.materials.append(mat)
        cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
        cam.location, cam.rotation_euler = cam_loc, cam_rot
        cam.data.lens = 50 if shape == "sphere" else 32
        scene.collection.objects.link(cam)
        scene.camera = cam
        sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
        sun.data.energy = 3.5
        sun.data.angle = math.radians(2)
        sun.rotation_euler = (math.radians(60), 0.0, math.radians(-70))  # low, raking light shows the relief
        scene.collection.objects.link(sun)
        world = bpy.data.worlds.new("w")
        world.use_nodes = True
        world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.05, 0.06, 1)
        world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
        scene.world = world
        preview = prefix + "_preview.png"
        scene.render.filepath = preview
        scene.render.image_settings.file_format = "PNG"
        bpy.ops.render.render(write_still=True)
        report["files"]["preview"] = preview
        report["preview_seconds"] = round(time.time() - t, 1)

report["seconds"] = round(time.time() - t0, 1)
print("MATERIAL_MAPS_RESULT " + json.dumps(report), flush=True)
