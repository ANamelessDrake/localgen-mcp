"""FLUX.2 Klein 4B: text to image, or editing guided by up to four reference images.

Graph from ComfyUI's template image_flux2_klein_image_edit_4b_distilled.json
(flattened from its subgraphs): Qwen3 4B text encoder (CLIPLoader type flux2),
each reference image scaled to about 1 megapixel, VAE encoded and attached to
both the positive and the zeroed negative conditioning with ReferenceLatent,
Flux2Scheduler 4 steps, euler, CFGGuider cfg 1. The output takes the first
reference image's size unless width and height are given.
"""

from __future__ import annotations

from . import Workflow

SLOTS = ("ref1", "ref2", "ref3", "ref4")


def build(p: dict) -> dict:
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b-fp8.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": p["prompt"], "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
    }
    pos, neg = ["4", 0], ["5", 0]
    first_scaled = None
    refs = [p.get("images", {}).get(s) for s in SLOTS]
    for i, name in enumerate(r for r in refs if r):
        base = 20 + 10 * i
        load, scale, enc, rpos, rneg = (str(base + k) for k in range(5))
        g[load] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g[scale] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": [load, 0], "upscale_method": "nearest-exact", "megapixels": 1.0, "resolution_steps": 1}}
        g[enc] = {"class_type": "VAEEncode", "inputs": {"pixels": [scale, 0], "vae": ["3", 0]}}
        g[rpos] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": [enc, 0]}}
        g[rneg] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": neg, "latent": [enc, 0]}}
        pos, neg = [rpos, 0], [rneg, 0]
        first_scaled = first_scaled or scale

    if p.get("width") and p.get("height"):
        width, height = int(p["width"]), int(p["height"])
    elif first_scaled:
        g["6"] = {"class_type": "GetImageSize", "inputs": {"image": [first_scaled, 0]}}
        width, height = ["6", 0], ["6", 1]
    else:
        width, height = 1024, 1024

    g.update({
        "7": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "Flux2Scheduler", "inputs": {"steps": p["steps"], "width": width, "height": height}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": p["seed"]}},
        "11": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": pos, "negative": neg, "cfg": 1}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["10", 0], "guider": ["11", 0], "sampler": ["9", 0], "sigmas": ["8", 0], "latent_image": ["7", 0]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": p["prefix"]}},
    })
    return g


WORKFLOW = Workflow(
    name="flux2-klein-4b",
    modality="image",
    description=("FLUX.2 Klein 4B fp8 (Apache 2.0): text to image, or edit and combine up to 4 reference "
                 "images guided by the prompt (for example 'put the person from image 1 in the room from image 2')"),
    models=[
        "diffusion_models/flux-2-klein-4b-fp8.safetensors",
        "text_encoders/qwen_3_4b.safetensors",
        "vae/flux2-vae.safetensors",
    ],
    defaults={"width": None, "height": None, "steps": 4},
    limits={"width": (256, 2048), "height": (256, 2048), "steps": (1, 50)},
    build=build,
    image_slots=SLOTS,
)
