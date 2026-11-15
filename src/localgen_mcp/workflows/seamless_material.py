"""Seamless (tileable) base color for a PBR surface material.

Image models do not tile, so the graph makes the image tile afterwards, all with
native ComfyUI nodes:
  base       Z-Image Turbo text to image; or FLUX.2 Klein following a reference
             image (add_edit from flux2_klein_4b); or an existing image (`source`)
  roll       offset by half the size with wrap around (ImageCrop pieces pasted back
             shifted with ImageCompositeMasked). The old edges now meet in a cross
             through the middle; the new edges are the old centre lines, which are
             continuous, so the image wraps.
  pass 1     repaint the cross: Z-Image image to image with a soft, ragged cross mask
             as the latent noise mask (SetLatentNoiseMask), then ImageCompositeMasked
             puts only the masked part back. The arms stop short of the image edges:
             repainting up to an edge would paint the two sides of that edge
             independently and break the wrap there.
  pass 2     what is left of the seam is four short pieces where the arms stopped,
             which on the torus are two spots, at the middle of the top edge and of
             the left edge. Roll by a quarter more, which puts both spots inside the
             image a quarter of the size from every edge, repaint a patch over each,
             and roll back.
No pass touches an image edge, so every edge pixel is a pixel of the original image
next to its original neighbour: the result wraps exactly. The masks are built on a
coarse grid (nested shapes summed into a ramp, noise added so the edges wander
instead of running straight, which shows as a grid when tiled), then upscaled
bilinearly. materials.py measures the result and can run both passes again on it
(`source`, roll=False).
"""

from __future__ import annotations

from . import Workflow
from .flux2_klein_4b import MODELS as KLEIN_MODELS, add_edit, add_loaders as add_klein_loaders

ZIMAGE_MODELS = ("diffusion_models/z_image_turbo_bf16.safetensors", "text_encoders/qwen_3_4b_fp8_mixed.safetensors",
                 "vae/ae.safetensors")

TEXTURE_SUFFIX = (", seamless tileable texture, flat top-down orthographic view of the surface filling the entire "
                  "frame, even diffuse lighting, no shadows, no perspective, no horizon, no objects, no border, "
                  "highly detailed PBR albedo texture")


def texture_prompt(prompt: str) -> str:
    return prompt.strip().rstrip(".") + TEXTURE_SUFFIX


def reference_prompt(prompt: str) -> str:
    return (f"Turn image 1 into a flat surface texture of {prompt.strip().rstrip('.')}, keeping its colours and "
            "material. Top-down orthographic view of the surface filling the entire frame, even diffuse lighting, "
            "no shadows, no perspective, no objects, no border.")


def add_zimage(g: dict, prompt: str) -> tuple[list, list, list, list]:
    """Z-Image Turbo loaders and conditioning (as zimage_turbo.py); returns model, positive, negative, vae."""
    g["z_unet"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors",
                                                         "weight_dtype": "default"}}
    g["z_clip"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b_fp8_mixed.safetensors",
                                                         "type": "lumina2", "device": "default"}}
    g["z_vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}}
    g["z_text"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["z_clip", 0]}}
    g["z_zero"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["z_text", 0]}}
    g["z_model"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["z_unet", 0], "shift": 3}}
    return ["z_model", 0], ["z_text", 0], ["z_zero", 0], ["z_vae", 0]


def add_roll(g: dict, ns: str, image: list, size: int, dx: int, dy: int) -> list:
    """Offset a size x size image by (dx, dy) with wrap around (a torus roll), 0 < dx, dy < size."""
    if not (0 < dx < size and 0 < dy < size):
        raise ValueError("roll offsets must be inside the image")
    dest = image
    # The piece starting at (x, y) moves to ((x + dx) % size, (y + dy) % size).
    pieces = [(x, y, w, h) for x, w in ((0, size - dx), (size - dx, dx)) for y, h in ((0, size - dy), (size - dy, dy))]
    for i, (x, y, w, h) in enumerate(pieces):
        crop, paste = f"{ns}crop{i}", f"{ns}paste{i}"
        g[crop] = {"class_type": "ImageCrop", "inputs": {"image": image, "width": w, "height": h, "x": x, "y": y}}
        g[paste] = {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": dest, "source": [crop, 0], "x": (x + dx) % size, "y": (y + dy) % size,
            "resize_source": False}}
        dest = [paste, 0]
    return dest


MASK_CELLS = 64  # masks are drawn on a 64 x 64 grid, then upscaled bilinearly

Rect = tuple[int, int, int, int]  # x, y, width, height in mask cells


def cross_layers(width_fraction: float) -> list[list[Rect]]:
    """Nested crosses for pass 1, outermost first: each layer two cells narrower than the last
    and its arms one cell shorter every other layer; the outermost keeps one empty cell at every
    image edge, so the edges get no repaint at all."""
    c = MASK_CELLS
    outer = max(4, 2 * int(round(width_fraction * c / 2)))
    layers = []
    for i, width in enumerate(range(outer, 0, -2)):
        inset = 1 + i // 2
        lo = c // 2 - width // 2
        layers.append([(lo, inset, width, c - 2 * inset), (inset, lo, c - 2 * inset, width)])
    return layers


def cross_gap(width_fraction: float) -> int:
    """How many cells from each edge pass 1's core stops: the seam left for pass 2."""
    return 1 + (len(cross_layers(width_fraction)) - 1) // 2


def patch_layers(width_fraction: float) -> list[list[Rect]]:
    """Nested squares for pass 2, one per leftover seam spot. After the quarter roll the spots sit
    at (3/4, 1/4) and (1/4, 3/4) of the image; each core square covers the seam left within
    cross_gap cells of the spot, and the outermost square stays clear of the edges."""
    c = MASK_CELLS
    core = cross_gap(width_fraction) + 1
    outer = min(c // 4 - 2, core + 4)
    layers = []
    for half in range(outer, core - 1, -1):
        layers.append([(cx - half, cy - half, 2 * half, 2 * half) for cx, cy in ((3 * c // 4, c // 4), (c // 4, 3 * c // 4))])
    return layers


def add_soft_mask(g: dict, ns: str, size: int, layers: list[list[Rect]], seed: int, noise: float) -> list:
    """A soft, ragged mask: the layers' rectangles summed into a ramp, noise added, clipped to the
    outermost layer, the innermost layer forced to 1, upscaled to size x size."""
    c = MASK_CELLS
    step = 1.0 / len(layers)
    g[ns + "zero"] = {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": c, "height": c}}

    def paint(name: str, rects: list[Rect], value: float, dest: list) -> list:
        for j, (x, y, w, h) in enumerate(rects):
            solid, comp = f"{ns}{name}_{j}", f"{ns}{name}_{j}_add"
            g[solid] = {"class_type": "SolidMask", "inputs": {"value": value, "width": w, "height": h}}
            g[comp] = {"class_type": "MaskComposite", "inputs": {
                "destination": dest, "source": [solid, 0], "x": x, "y": y, "operation": "add"}}
            dest = [comp, 0]
        return dest

    ramp = [ns + "zero", 0]
    for i, rects in enumerate(layers):
        ramp = paint(f"l{i}", rects, step, ramp)
    g[ns + "ramp_img"] = {"class_type": "MaskToImage", "inputs": {"mask": ramp}}
    g[ns + "noisy"] = {"class_type": "ImageAddNoise", "inputs": {"image": [ns + "ramp_img", 0], "seed": seed,
                                                                 "strength": noise}}
    g[ns + "noisy_mask"] = {"class_type": "ImageToMask", "inputs": {"image": [ns + "noisy", 0], "channel": "red"}}
    envelope = paint("env", layers[0], 1.0, [ns + "zero", 0])
    g[ns + "clipped"] = {"class_type": "MaskComposite", "inputs": {
        "destination": [ns + "noisy_mask", 0], "source": envelope, "x": 0, "y": 0, "operation": "multiply"}}
    core = paint("core", layers[-1], 1.0, [ns + "clipped", 0])
    g[ns + "img"] = {"class_type": "MaskToImage", "inputs": {"mask": core}}
    g[ns + "up"] = {"class_type": "ImageScale", "inputs": {
        "image": [ns + "img", 0], "upscale_method": "bilinear", "width": size, "height": size, "crop": "disabled"}}
    g[ns + "mask"] = {"class_type": "ImageToMask", "inputs": {"image": [ns + "up", 0], "channel": "red"}}
    return [ns + "mask", 0]


def add_seam_pass(g: dict, ns: str, image: list, mask: list, z: tuple, seed: int, steps: int,
                  denoise: float) -> list:
    """Repaint the masked band of image with Z-Image image to image and composite only the band back."""
    model, pos, neg, vae = z
    g[ns + "enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": image, "vae": vae}}
    g[ns + "noise_mask"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": [ns + "enc", 0], "mask": mask}}
    g[ns + "sample"] = {"class_type": "KSampler", "inputs": {
        "model": model, "positive": pos, "negative": neg, "latent_image": [ns + "noise_mask", 0],
        "seed": seed, "steps": steps, "cfg": 1, "sampler_name": "res_multistep", "scheduler": "simple",
        "denoise": denoise}}
    g[ns + "dec"] = {"class_type": "VAEDecode", "inputs": {"samples": [ns + "sample", 0], "vae": vae}}
    g[ns + "out"] = {"class_type": "ImageCompositeMasked", "inputs": {
        "destination": image, "source": [ns + "dec", 0], "x": 0, "y": 0, "resize_source": False, "mask": mask}}
    return [ns + "out", 0]


def build(p: dict) -> dict:
    size, seed, steps = int(p["size"]), int(p["seed"]), int(p["steps"])
    if size % 64:
        raise ValueError(f"size must be a multiple of 64, got {size}")
    images = p.get("images") or {}
    g: dict = {}
    z = add_zimage(g, texture_prompt(p["prompt"]))
    if images.get("source"):
        g["source"] = {"class_type": "LoadImage", "inputs": {"image": images["source"]}}
        g["base"] = {"class_type": "ImageScale", "inputs": {
            "image": ["source", 0], "upscale_method": "lanczos", "width": size, "height": size, "crop": "center"}}
        base = ["base", 0]
    elif images.get("reference"):
        add_klein_loaders(g)
        g["reference"] = {"class_type": "LoadImage", "inputs": {"image": images["reference"]}}
        base = add_edit(g, "base_", reference_prompt(p["prompt"]), [["reference", 0]], seed,
                        int(p["reference_steps"]), size, size, upscale="lanczos")
    else:
        model, pos, neg, vae = z
        g["base_latent"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": size, "height": size,
                                                                           "batch_size": 1}}
        g["base_sample"] = {"class_type": "KSampler", "inputs": {
            "model": model, "positive": pos, "negative": neg, "latent_image": ["base_latent", 0], "seed": seed,
            "steps": steps, "cfg": 1, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1}}
        g["base_dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["base_sample", 0], "vae": vae}}
        base = ["base_dec", 0]
    half, quarter = size // 2, size // 4
    image = add_roll(g, "roll_", base, size, half, half) if p.get("roll", True) else base
    width, noise, denoise = float(p["seam_width"]), float(p["seam_noise"]), float(p["seam_denoise"])
    mask = add_soft_mask(g, "seam_", size, cross_layers(width), seed + 2, noise)
    image = add_seam_pass(g, "pass_", image, mask, z, seed + 1, steps, denoise)
    image = add_roll(g, "quarter_", image, size, quarter, quarter)
    mask = add_soft_mask(g, "spot_", size, patch_layers(width), seed + 4, noise)
    image = add_seam_pass(g, "pass2_", image, mask, z, seed + 3, steps, denoise)
    image = add_roll(g, "back_", image, size, size - quarter, size - quarter)
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": image, "filename_prefix": p["prefix"]}}
    if p.get("save_untiled"):
        g["save_untiled"] = {"class_type": "SaveImage", "inputs": {"images": base,
                                                                   "filename_prefix": p["prefix"] + "_untiled"}}
    return g


WORKFLOW = Workflow(
    name="seamless-zimage",
    modality="material",
    description=("Seamless base color for a PBR material: Z-Image Turbo (Apache 2.0) texture, or FLUX.2 Klein 4B "
                 "from a reference image, offset by half with wrap around and the seam cross repainted under a mask"),
    models=list(ZIMAGE_MODELS) + [m for m in KLEIN_MODELS if m not in ZIMAGE_MODELS],
    defaults={"size": 1024, "steps": 8, "reference_steps": 4, "seam_width": 0.25, "seam_noise": 0.25, "seam_denoise": 0.7,
              "roll": True, "save_untiled": True},
    limits={"size": (512, 2048), "steps": (1, 50), "reference_steps": (1, 50)},
    build=build,
    image_slots=("reference", "source"),
    is_default=True,
)
