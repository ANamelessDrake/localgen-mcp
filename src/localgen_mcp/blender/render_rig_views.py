"""Colour renders of a GLB from Pixal3D's rig cameras, on a plain background (generate_turnaround's guide).

    blender -b --factory-startup -P render_rig_views.py -- '<json spec>'

spec: {"mesh": GLB path, "prefix": output path prefix, "views": {name: azimuth degrees},
"size": pixels, "background": [r, g, b] bytes, "samples": Cycles samples, optional "match": an
image of the object's front}.

Facing: a TRELLIS.2 mesh is not always turned so that its front faces the rig's front camera (a
robot came out facing +X, a moth facing -Y). With "match", the mesh is first rendered small from
the four axis aligned yaws, each render and the match image are cropped to their silhouette and
compared (silhouette aspect, silhouette overlap and colour), and every camera is turned by the
best yaw; the report's "yaw" says by how much.

Writes <prefix>_<view>.png (RGB, size x size) per view, and reports each view's silhouette box
[x0, y0, x1, y1] in pixels (top left origin, from the render's alpha above 0.5). The mesh is framed as the rig assumes and
the cameras are the rig's (_rig.py), so every render shares one scale and the rig framing.
Lighting is a uniform white world: a matte surface shows about its own colour, a little darker in
crevices, with no cast shadows or highlights, which is the flat lighting turnaround views ask for.

Colour: texture files are read as Non-Color (raw bytes) and decoded from sRGB with a Gamma node;
each view renders transparent to a float EXR (linear, no view transform), and the PNG is composed
here with numpy (sRGB encode, over the background bytes) into a Non-Color byte image, so the
bytes written are exactly the computed values whatever OCIO setup this Blender build has.
The last stdout line is "RIG_VIEWS_RESULT {json}".
"""
import json
import os
import sys
import time

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rig  # noqa: E402

t0 = time.time()
spec = json.loads(sys.argv[sys.argv.index("--") + 1])
size = int(spec.get("size", 1024))
bg = np.array(spec.get("background", [216, 216, 216]), dtype=np.float32) / 255.0

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=spec["mesh"])
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not meshes:
    raise SystemExit("no mesh in " + spec["mesh"])
for o in bpy.context.scene.objects:  # importer leftovers such as bone display shapes
    if o.type != "MESH" and o.type != "EMPTY":
        o.hide_render = True
bbox = _rig.frame(meshes)

# Base colour textures: raw bytes in, decoded with a Gamma node (sRGB to linear, approximately).
for mat in {s.material for o in meshes for s in o.material_slots if s.material}:
    if not mat.use_nodes:
        continue
    nt = mat.node_tree
    for node in list(nt.nodes):
        if node.type != "TEX_IMAGE" or node.image is None:
            continue
        srgb = node.image.colorspace_settings.name.lower().startswith("srgb")
        node.image.colorspace_settings.name = "Non-Color"
        if srgb:
            for link in list(node.outputs["Color"].links):
                gamma = nt.nodes.new("ShaderNodeGamma")
                gamma.inputs["Gamma"].default_value = 2.2
                nt.links.new(node.outputs["Color"], gamma.inputs["Color"])
                nt.links.new(gamma.outputs["Color"], link.to_socket)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = int(spec.get("samples", 32))
scene.cycles.use_denoising = True
scene.render.resolution_x = scene.render.resolution_y = size
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.view_settings.view_transform = "Standard"
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.image_settings.color_mode = "RGBA"
world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world
cam = _rig.add_camera(scene)


def render(azimuth, res, samples, path):
    """sRGB colour (0..1, Blender's bottom up rows) and alpha of one view."""
    scene.render.resolution_x = scene.render.resolution_y = res
    scene.cycles.samples = samples
    _rig.place(cam, azimuth)
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "Non-Color"
    px = np.array(img.pixels[:], dtype=np.float32).reshape(res, res, 4)
    bpy.data.images.remove(img)
    os.remove(path)
    rgb, a = np.clip(px[..., :3], 0, 1), np.clip(px[..., 3:4], 0, 1)
    # Cycles' transparent film is premultiplied: un-premultiply, then encode.
    lin = np.where(a > 1e-6, rgb / np.maximum(a, 1e-6), 0.0)
    enc = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(np.clip(lin, 0, 1), 1 / 2.4) - 0.055)
    return enc, a


def normalised(rgb, mask, n=48):
    """The silhouette's box resampled to n x n (colour, mask) and the box's aspect ratio."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    iy = (y0 + (np.arange(n) + 0.5) * (y1 - y0) / n).astype(int)
    ix = (x0 + (np.arange(n) + 0.5) * (x1 - x0) / n).astype(int)
    m = mask[np.ix_(iy, ix)]
    c = np.where(m[..., None], rgb[np.ix_(iy, ix)], 0.5)
    return c, m, (x1 - x0) / (y1 - y0)


def facing(match):
    """The yaw (0, 90, 180 or 270) whose render looks most like the match image, and every score."""
    img = bpy.data.images.load(match)
    img.colorspace_settings.name = "Non-Color"  # raw bytes
    w, h = img.size
    ref = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, img.channels)[..., :3]
    bpy.data.images.remove(img)
    border = np.concatenate([ref[0], ref[-1], ref[:, 0], ref[:, -1]])
    ref_mask = np.abs(ref - np.median(border, axis=0)).max(axis=-1) > 0.08  # plain background
    target = normalised(ref, ref_mask)
    scores = {}
    for yaw in (0, 90, 180, 270):
        rgb, a = render(yaw, 160, 8, f"{spec['prefix']}_match_{yaw}.exr")
        cand = normalised(rgb, a[..., 0] > 0.5)
        if target is None or cand is None:
            return 0, {}
        scores[yaw] = round(float(np.abs(cand[0] - target[0]).mean() + 0.5 * (cand[1] != target[1]).mean()
                                  + 0.5 * abs(np.log(cand[2] / target[2]))), 4)
    return min(scores, key=scores.get), scores


yaw, scores = facing(spec["match"]) if spec.get("match") else (0, {})
paths, boxes = {}, {}
for name, az in spec["views"].items():
    enc, a = render(float(az) + yaw, size, int(spec.get("samples", 32)), f"{spec['prefix']}_{name}.exr")
    ys, xs = np.nonzero(a[::-1, :, 0] > 0.5)  # Blender's rows run bottom up
    boxes[name] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
    out_rgb = enc * a + bg * (1 - a)
    out = np.concatenate([out_rgb, np.ones_like(a)], axis=-1)
    png = f"{spec['prefix']}_{name}.png"
    im = bpy.data.images.new(f"out_{name}", size, size, alpha=False, float_buffer=False)
    im.colorspace_settings.name = "Non-Color"
    im.pixels.foreach_set(out.astype(np.float32).ravel())
    im.filepath_raw = png
    im.file_format = "PNG"
    im.save()
    bpy.data.images.remove(im)
    paths[name] = png

print("RIG_VIEWS_RESULT " + json.dumps({"mesh": spec["mesh"], "views": paths, "boxes": boxes, "yaw": yaw,
                                         "yaw_scores": scores, "bbox": [round(v, 4) for v in bbox],
                                         "seconds": round(time.time() - t0, 1)}), flush=True)
