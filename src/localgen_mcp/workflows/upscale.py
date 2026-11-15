"""Upscale an image 2x or 4x with a super resolution model, through native ComfyUI nodes.

UpscaleModelLoader loads the weights (spandrel picks the architecture) and
ImageUpscaleWithModel runs them. That node already tiles: 512 px tiles with a
32 px overlap, halved down to 128 px if the GPU runs out of memory, so large
images need no tiling here. Given an RGBA image it upscales the colour with the
model and the alpha channel bilinearly, so alpha is kept by joining LoadImage's
mask back onto the image first.

Both models are 4x. For 2x the 4x result is downscaled by half with Lanczos,
so one model file serves both scales.
"""

from __future__ import annotations

from . import Workflow

NATIVE_SCALE = 4
SCALES = (2, 4)


def build(p: dict) -> dict:
    scale = int(p["scale"])
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}, got {scale}")
    g: dict = {
        "load": {"class_type": "LoadImage", "inputs": {"image": p["images"]["image"]}},
        "model": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": p["model_file"]}},
    }
    image = ["load", 0]
    if p.get("alpha"):
        # LoadImage's mask is 1 - alpha, which is what JoinImageWithAlpha expects (it inverts).
        g["rgba"] = {"class_type": "JoinImageWithAlpha", "inputs": {"image": image, "alpha": ["load", 1]}}
        image = ["rgba", 0]
    g["upscale"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["model", 0], "image": image}}
    image = ["upscale", 0]
    if scale != NATIVE_SCALE:
        g["resize"] = {"class_type": "ImageScaleBy", "inputs": {
            "image": image, "upscale_method": p["downscale_filter"], "scale_by": scale / NATIVE_SCALE}}
        image = ["resize", 0]
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": image, "filename_prefix": p["prefix"]}}
    return g


def _workflow(name: str, model_file: str, description: str, is_default: bool = False) -> Workflow:
    return Workflow(
        name=name,
        modality="upscale",
        description=description,
        models=[f"upscale_models/{model_file}"],
        defaults={"scale": 4, "alpha": False, "model_file": model_file, "downscale_filter": "lanczos"},
        limits={"scale": (2, 4)},
        build=build,
        image_slots=("image",),
        min_images=1,
        is_default=is_default,
    )


WORKFLOWS = [
    _workflow("realesrgan-x4plus", "RealESRGAN_x4plus.pth",
              "Real-ESRGAN x4plus (BSD-3-Clause), general 4x upscaler for photos, renders and concept art; "
              "removes mild noise and compression", is_default=True),
    _workflow("pbrify-v4", "4x-PBRify_UpscalerV4.safetensors",
              "4x-PBRify_UpscalerV4 (CC0-1.0, DAT2), 4x upscaler for game textures, trained on CC0 ambientCG "
              "textures; keeps fine surface detail, slower"),
]
