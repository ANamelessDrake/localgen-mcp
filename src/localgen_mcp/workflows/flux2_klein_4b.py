"""FLUX.2 Klein 4B: text to image, or editing guided by up to four reference images.

Graph from ComfyUI's template image_flux2_klein_image_edit_4b_distilled.json
(flattened from its subgraphs): Qwen3 4B text encoder (CLIPLoader type flux2),
each reference image scaled to about 1 megapixel, VAE encoded and attached to
both the positive and the zeroed negative conditioning with ReferenceLatent,
Flux2Scheduler 4 steps, euler, CFGGuider cfg 1. The output takes the first
reference image's size unless width and height are given.

With an input image (slot "image", optionally "mask") it edits that image: it
is reference image 1 and also the start latent (SplitSigmasDenoise runs the
last `denoise` part of the schedule), any other references follow it, and the
output keeps the input's size (see image_edit.py for the mask and composite).
"""

from __future__ import annotations

from . import Workflow, image_edit

SLOTS = ("ref1", "ref2", "ref3", "ref4")
MODELS = ("diffusion_models/flux-2-klein-4b-fp8.safetensors", "text_encoders/qwen_3_4b.safetensors",
          "vae/flux2-vae.safetensors")
LOADERS = ("k_unet", "k_clip", "k_vae")


def add_loaders(g: dict) -> None:
    unet, clip, vae = LOADERS
    g[unet] = {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b-fp8.safetensors", "weight_dtype": "default"}}
    g[clip] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2", "device": "default"}}
    g[vae] = {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}}


def add_edit(g: dict, ns: str, text: str, refs: list[list], seed: int, steps: int, width, height,
             init: list | None = None, denoise: float = 1.0, upscale: str = "nearest-exact",
             mask: list | None = None) -> list:
    """Add one Klein sampling chain to g (loaders from add_loaders) and return its decoded IMAGE link.

    refs are IMAGE links, each scaled to about 1 megapixel, VAE encoded and attached to both the
    positive and the zeroed negative conditioning with ReferenceLatent. With init (an IMAGE link),
    sampling starts from that image's latent and runs only the last `denoise` part of the
    schedule (image to image), which keeps its composition; mask (a MASK link) then limits the
    noise to that region with SetLatentNoiseMask.
    """
    unet, clip, vae = LOADERS
    g[ns + "text"] = {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": [clip, 0]}}
    g[ns + "zero"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": [ns + "text", 0]}}
    pos, neg = [ns + "text", 0], [ns + "zero", 0]
    for i, ref in enumerate(refs):
        scale, enc, rpos, rneg = (f"{ns}ref{i}_{k}" for k in ("scale", "enc", "pos", "neg"))
        g[scale] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": ref, "upscale_method": upscale, "megapixels": 1.0, "resolution_steps": 1}}
        g[enc] = {"class_type": "VAEEncode", "inputs": {"pixels": [scale, 0], "vae": [vae, 0]}}
        g[rpos] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": [enc, 0]}}
        g[rneg] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": neg, "latent": [enc, 0]}}
        pos, neg = [rpos, 0], [rneg, 0]
    g[ns + "sched"] = {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}}
    sigmas = [ns + "sched", 0]
    if init is not None:
        g[ns + "init"] = {"class_type": "VAEEncode", "inputs": {"pixels": init, "vae": [vae, 0]}}
        g[ns + "split"] = {"class_type": "SplitSigmasDenoise", "inputs": {"sigmas": sigmas, "denoise": denoise}}
        sigmas, latent = [ns + "split", 1], [ns + "init", 0]
        if mask is not None:
            g[ns + "noise_mask"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": latent, "mask": mask}}
            latent = [ns + "noise_mask", 0]
    else:
        g[ns + "empty"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
        latent = [ns + "empty", 0]
    g.update({
        ns + "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        ns + "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        ns + "guider": {"class_type": "CFGGuider", "inputs": {"model": [unet, 0], "positive": pos, "negative": neg, "cfg": 1}},
        ns + "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": [ns + "noise", 0], "guider": [ns + "guider", 0], "sampler": [ns + "sampler", 0],
            "sigmas": sigmas, "latent_image": latent}},
        ns + "decode": {"class_type": "VAEDecode", "inputs": {"samples": [ns + "sample", 0], "vae": [vae, 0]}},
    })
    return [ns + "decode", 0]


def build(p: dict) -> dict:
    g: dict = {}
    add_loaders(g)
    refs = []
    for i, name in enumerate(r for r in (p.get("images", {}).get(s) for s in SLOTS) if r):
        g[f"load{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        refs.append([f"load{i}", 0])
    src = image_edit.add_source(g, p)
    if src:
        # The edited image is image 1 and the start latent; the output keeps its size.
        g["size"] = {"class_type": "GetImageSize", "inputs": {"image": src["work"]}}
        steps, denoise = image_edit.schedule_steps(int(p["steps"]), float(p["denoise"]))
        image = add_edit(g, "", p["prompt"], [src["work"]] + refs, p["seed"], steps, ["size", 0], ["size", 1],
                         init=src["work"], denoise=denoise, mask=src["mask"])
        g["save"] = {"class_type": "SaveImage", "inputs": {"images": image_edit.finish(g, image, src),
                                                           "filename_prefix": p["prefix"]}}
        return g
    if p.get("width") and p.get("height"):
        width, height = int(p["width"]), int(p["height"])
    elif refs:
        # The output takes the first reference's size after its 1 megapixel scaling.
        g["size_scale"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": refs[0], "upscale_method": "nearest-exact", "megapixels": 1.0, "resolution_steps": 1}}
        g["size"] = {"class_type": "GetImageSize", "inputs": {"image": ["size_scale", 0]}}
        width, height = ["size", 0], ["size", 1]
    else:
        width, height = 1024, 1024
    image = add_edit(g, "", p["prompt"], refs, p["seed"], p["steps"], width, height)
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": image, "filename_prefix": p["prefix"]}}
    return g


WORKFLOW = Workflow(
    name="flux2-klein-4b",
    modality="image",
    description=("FLUX.2 Klein 4B fp8 (Apache 2.0): text to image, or edit and combine up to 4 reference "
                 "images guided by the prompt (for example 'put the person from image 1 in the room from image 2')"),
    models=list(MODELS),
    defaults={"width": None, "height": None, "steps": 4, "denoise": 1.0},
    limits={"width": (256, 2048), "height": (256, 2048), "steps": (1, 50)},
    build=build,
    image_slots=SLOTS + image_edit.EDIT_SLOTS,
)
