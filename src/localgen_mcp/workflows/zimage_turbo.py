"""Z-Image Turbo text to image.

Graph and settings taken from /ai/ComfyUI/zimage.py (which matches ComfyUI's
built-in Z-Image Turbo template): CLIPLoader type lumina2,
ModelSamplingAuraFlow shift 3, KSampler 8 steps cfg 1 res_multistep/simple,
negative via ConditioningZeroOut.
"""

from __future__ import annotations

from . import Workflow


def build(p: dict) -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": p["text_encoder"], "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": p["prompt"], "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3}},
        "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": p["width"], "height": p["height"], "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["6", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["7", 0],
            "seed": p["seed"], "steps": p["steps"], "cfg": 1, "sampler_name": "res_multistep",
            "scheduler": "simple", "denoise": 1}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": p["prefix"]}},
    }


WORKFLOW = Workflow(
    name="z-image-turbo",
    modality="image",
    description="Z-Image Turbo 6B (Apache 2.0), 8 step text to image, about 8 s per 1024x1024 image warm",
    models=[
        "diffusion_models/z_image_turbo_bf16.safetensors",
        "text_encoders/qwen_3_4b_fp8_mixed.safetensors",
        "vae/ae.safetensors",
    ],
    defaults={"width": 1024, "height": 1024, "steps": 8, "text_encoder": "qwen_3_4b_fp8_mixed.safetensors"},
    limits={"width": (256, 2048), "height": (256, 2048), "steps": (1, 50)},
    build=build,
    is_default=True,
)
