"""Pixal3D multi-view: textured mesh (GLB) from up to four views of one object.

ComfyUI's template 3d_pixal3d_multi_views.json runs the same TRELLIS.2
pipeline (identical sampler, CFG, upsample and bake settings) with two
changes, which is how this graph is derived from trellis2_api.json:
the Pixal3D multi-view diffusion model, and Pixal3DMultiViewConditioning fed
by one background removal and crop (pad 1.1) per view. Views are front, left,
back and right, 90 degrees apart, at the same scale; the first one given is
the front the mesh is posed to.

framing="crop" (the template's way) crops each view to its own outline. That
keeps the scale only when the object's widest extent is the same in every view
(a standing figure, whose height dominates); for an object wider from the
front than the side it stretches the side views, and the mesh comes out too
deep. framing="as_framed" uses square views exactly as given, for renders or
turnarounds already framed like the rig: the object centred, about 1/1.1 of
the frame at its widest, one scale for all views.
"""

from __future__ import annotations

import copy

from . import Workflow
from .trellis2 import _GRAPH, _SAMPLERS

SLOTS = ("front", "left", "back", "right")
FRAMINGS = ("crop", "as_framed")


def build(p: dict) -> dict:
    g = copy.deepcopy(_GRAPH)
    g["40"]["inputs"]["unet_name"] = "pixal3d_multiview_int8_convrot.safetensors"
    if p.get("framing", "crop") not in FRAMINGS:
        raise ValueError(f"framing must be one of {FRAMINGS}")
    for nid in ("122", "192", "312"):  # single image load, mask and crop
        g.pop(nid, None)
    cond = {"clip_vision_model": ["15", 0], "fov": float(p["fov"])}
    for i, slot in enumerate(SLOTS):
        name = p.get("images", {}).get(slot)
        if not name:
            continue
        load, mask, out = (str(500 + 10 * i + k) for k in range(3))
        g[load] = {"class_type": "LoadImage", "inputs": {"image": name}}
        if p.get("remove_background", True):
            g[mask] = {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["193", 0], "image": [load, 0]}}
        else:
            # The image's own alpha: LoadImage's mask output is 1 - alpha, so invert it.
            g[mask] = {"class_type": "InvertMask", "inputs": {"mask": [load, 1]}}
        if p.get("framing", "crop") == "as_framed":
            # Keep the view's framing; attach the foreground as alpha and let the
            # conditioning node composite it on black. JoinImageWithAlpha inverts
            # its mask input (LoadImage convention), so give it the background.
            inv = str(500 + 10 * i + 3)
            g[inv] = {"class_type": "InvertMask", "inputs": {"mask": [mask, 0]}}
            g[out] = {"class_type": "JoinImageWithAlpha", "inputs": {"image": [load, 0], "alpha": [inv, 0]}}
        else:
            g[out] = {"class_type": "ImageCropToMask", "inputs": {
                "images": [load, 0], "masks": [mask, 0], "width": 1024, "height": 1024,
                "pad_factor": 1.1, "grow_mask": 0, "background": "#000000"}}
        cond[slot] = [out, 0]
    if not p.get("remove_background", True):
        g.pop("193", None)
    # Same node id as the single-image conditioning, so every downstream link still holds.
    g["299"] = {"class_type": "Pixal3DMultiViewConditioning", "inputs": cond}
    for i, nid in enumerate(_SAMPLERS):
        g[nid]["inputs"]["seed"] = int(p["seed"]) + i
    g["94"]["inputs"]["target_resolution"] = p["resolution"]
    g["288"]["inputs"]["value"] = p["texture_size"]
    g["186"]["inputs"]["target_face_count"] = p["face_count"]
    g["900"]["inputs"]["filename_prefix"] = p["prefix"]
    return g


WORKFLOW = Workflow(
    name="pixal3d-multiview",
    modality="mesh",
    description=("Pixal3D multi-view int8 (MIT): PBR textured GLB from front, left, back and right views of one "
                 "object (at least two recommended), same TRELLIS.2 pipeline and VAEs"),
    models=[
        "diffusion_models/pixal3d_multiview_int8_convrot.safetensors",
        "vae/trellis_2_shape_vae_bf16.safetensors",
        "vae/trellis_2_texture_vae_bf16.safetensors",
        "clip_vision/dino_v3_L_naf_fp32.safetensors",
        "background_removal/birefnet.safetensors",
    ],
    defaults={"resolution": 1024, "texture_size": 2048, "face_count": 300000, "remove_background": True, "fov": 20.0,
              "framing": "crop"},
    limits={"resolution": (1024, 2048), "texture_size": (512, 8192), "face_count": (5000, 2000000)},
    build=build,
    image_slots=SLOTS,
    min_images=1,
)
