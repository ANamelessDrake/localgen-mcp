"""Seamless (tileable) base color for a PBR surface material.

Image models do not tile, so the graph makes the image tile afterwards, all with
native ComfyUI nodes:
  base       Z-Image Turbo text to image; or FLUX.2 Klein following a reference
             image (add_edit from flux2_klein_4b); or an existing image (`source`)
  roll       offset by half the size with wrap around (four ImageCrop quadrants
             pasted back swapped with ImageCompositeMasked). The old edges now meet
             in a cross through the middle and the new edges are the old centre
             lines, which are continuous, so the image wraps without a seam.
  seam mask  a cross shaped band over that seam (SolidMask + MaskComposite), its
             edges softened by an area downscale and bilinear upscale
  repaint    Z-Image image to image under that mask as the latent noise mask
             (SetLatentNoiseMask), so the sampler repaints only the band and keeps
             the rest as context; then ImageCompositeMasked puts only the band back
             onto the rolled image, so pixels outside it are unchanged
The edges are never repainted, so the result tiles by construction; the repaint
only hides the cross. materials.py checks both numerically and can run the
repaint again on the result (`source`, roll=False).
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


def add_roll(g: dict, ns: str, image: list, size: int) -> list:
    """Offset a size x size image by half its size in x and y with wrap around (a torus roll)."""
    h = size // 2
    # Quadrant at (x, y) of the source moves to ((x + h) % size, (y + h) % size).
    dest = image
    for i, (x, y) in enumerate(((0, 0), (h, 0), (0, h), (h, h))):
        crop, paste = f"{ns}crop{i}", f"{ns}paste{i}"
        g[crop] = {"class_type": "ImageCrop", "inputs": {"image": image, "width": h, "height": h, "x": x, "y": y}}
        g[paste] = {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": dest, "source": [crop, 0], "x": (x + h) % size, "y": (y + h) % size,
            "resize_source": False}}
        dest = [paste, 0]
    return dest


MASK_CELLS = 32  # the seam mask is drawn on a 32 x 32 grid, then smoothly upscaled


def seam_layers(width_fraction: float) -> list[int]:
    """Widths, in mask cells, of the nested crosses that build the band: outermost first, two cells
    narrower each step, down to the two cell core over the old edges."""
    outer = max(4, 2 * int(round(width_fraction * MASK_CELLS / 2)))
    return list(range(outer, 0, -2))


def add_seam_mask(g: dict, ns: str, size: int, width_fraction: float, seed: int, noise: float = 0.25) -> list:
    """A soft, ragged cross through the centre: 1 over the old edges, falling to 0 at the band's edge.

    Built on a coarse grid: nested crosses summed into a ramp, noise added so the band's edge
    wanders instead of running straight (straight edges show as a grid when tiled), clipped to the
    outer band, the core forced back to 1, then upscaled bilinearly to the image size.
    """
    c = MASK_CELLS
    layers = seam_layers(width_fraction)
    step = 1.0 / len(layers)

    def cross(name: str, width: int, value: float, dest: list) -> list:
        lo = c // 2 - width // 2
        g[f"{ns}{name}_v"] = {"class_type": "SolidMask", "inputs": {"value": value, "width": width, "height": c}}
        g[f"{ns}{name}_h"] = {"class_type": "SolidMask", "inputs": {"value": value, "width": c, "height": width}}
        g[f"{ns}{name}_cv"] = {"class_type": "MaskComposite", "inputs": {
            "destination": dest, "source": [f"{ns}{name}_v", 0], "x": lo, "y": 0, "operation": "add"}}
        # Where the bars cross the values add up (clamped to 1), so the centre square is a little denser.
        g[f"{ns}{name}_ch"] = {"class_type": "MaskComposite", "inputs": {
            "destination": [f"{ns}{name}_cv", 0], "source": [f"{ns}{name}_h", 0], "x": 0, "y": lo,
            "operation": "add"}}
        return [f"{ns}{name}_ch", 0]

    g[ns + "zero"] = {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": c, "height": c}}
    ramp = [ns + "zero", 0]
    for i, width in enumerate(layers):
        ramp = cross(f"l{i}", width, step, ramp)
    g[ns + "ramp_img"] = {"class_type": "MaskToImage", "inputs": {"mask": ramp}}
    g[ns + "noisy"] = {"class_type": "ImageAddNoise", "inputs": {"image": [ns + "ramp_img", 0], "seed": seed,
                                                                 "strength": noise}}
    g[ns + "noisy_mask"] = {"class_type": "ImageToMask", "inputs": {"image": [ns + "noisy", 0], "channel": "red"}}
    envelope = cross("env", layers[0], 1.0, [ns + "zero", 0])
    g[ns + "clipped"] = {"class_type": "MaskComposite", "inputs": {
        "destination": [ns + "noisy_mask", 0], "source": envelope, "x": 0, "y": 0, "operation": "multiply"}}
    core = cross("core", layers[-1], 1.0, [ns + "clipped", 0])
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
    image = add_roll(g, "roll_", base, size) if p.get("roll", True) else base
    mask = add_seam_mask(g, "seam_", size, float(p["seam_width"]), seed + 2, float(p["seam_noise"]))
    image = add_seam_pass(g, "pass_", image, mask, z, seed + 1, steps, float(p["seam_denoise"]))
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
