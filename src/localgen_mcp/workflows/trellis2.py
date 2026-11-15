"""TRELLIS.2 image to textured mesh (GLB).

The graph in trellis2_api.json is ComfyUI's own template
(3d_pixal3d_trellis2_image_to_model, switch set to TRELLIS.2), converted to
API format by scripts/make_trellis2_workflow.py. Stages: BiRefNet background
removal, crop, DINOv3 conditioning, sparse structure, shape at 512, shape
upsample, texture, remesh, decimate, UV unwrap, PBR bake, SaveGLB.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from . import Workflow

_GRAPH = json.loads((Path(__file__).with_name("trellis2_api.json")).read_text())
_SAMPLERS = ["3", "18", "23", "12"]  # structure, shape 512, shape upsampled, texture


def build(p: dict) -> dict:
    g = copy.deepcopy(_GRAPH)
    g["122"]["inputs"]["image"] = p["images"]["image"]
    for i, nid in enumerate(_SAMPLERS):
        g[nid]["inputs"]["seed"] = int(p["seed"]) + i
    g["94"]["inputs"]["target_resolution"] = p["resolution"]
    g["288"]["inputs"]["value"] = p["texture_size"]
    g["186"]["inputs"]["target_face_count"] = p["face_count"]
    g["900"]["inputs"]["filename_prefix"] = p["prefix"]
    if not p.get("remove_background", True):
        # Use the image's own alpha channel as the mask instead of BiRefNet.
        # LoadImage's mask output is 1 - alpha (background = 1), so invert it.
        g["alpha"] = {"class_type": "InvertMask", "inputs": {"mask": ["122", 1]}}
        g["312"]["inputs"]["masks"] = ["alpha", 0]
        del g["192"], g["193"]
    return g


WORKFLOW = Workflow(
    name="trellis2",
    modality="mesh",
    description="Microsoft TRELLIS.2 4B (MIT), image to PBR textured GLB, native ComfyUI nodes",
    models=[
        "diffusion_models/trellis_2_int8_convrot.safetensors",
        "vae/trellis_2_shape_vae_bf16.safetensors",
        "vae/trellis_2_texture_vae_bf16.safetensors",
        "clip_vision/dino_v3_L_naf_fp32.safetensors",
        "background_removal/birefnet.safetensors",
    ],
    defaults={"resolution": 1024, "texture_size": 2048, "face_count": 300000, "remove_background": True},
    limits={"resolution": (1024, 2048), "texture_size": (512, 8192), "face_count": (5000, 2000000)},
    build=build,
    image_slots=("image",),
    min_images=1,
    is_default=True,
)
