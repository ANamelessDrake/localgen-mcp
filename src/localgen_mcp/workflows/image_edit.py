"""Image to image and masked editing, shared by the image workflows.

An image workflow that lists EDIT_SLOTS in its image_slots calls add_source()
before sampling and finish() after decoding:

  LoadImage                      the caller's image at its own size (the original)
  ImageScaleToTotalPixels        a working copy at about 1 megapixel, both sides a
                                 multiple of 16 (the VAE downsamples by 8 and the
                                 models patch 2x2), so any input size works
  LoadImage + ImageToMask(red)   the mask, read from the red channel: white = change,
                                 black = keep. (LoadImage's own mask output is
                                 1 - alpha, which is not what a black and white
                                 mask image means.)
  ResizeImageMaskNode            the mask resized to the original's size
  GrowMask + ImageBlur           grown and feathered by FEATHER pixels, so the edit
                                 blends in and the sampler sees a little context

The workflow VAE encodes the working image as its start latent, limits the noise
to the mask with SetLatentNoiseMask, and samples with partial denoise. finish()
then pastes the decoded result back over the original with ImageCompositeMasked
(resized to the original's size), so every pixel outside the feathered mask is
the original's, byte for byte. Without a mask it only resizes the result back.
"""

from __future__ import annotations

EDIT_SLOTS = ("image", "mask")
WORK_MEGAPIXELS = 1.0
MULTIPLE = 16
FEATHER = 8          # pixels the mask grows by, and the blur radius of its edge
MAX_REFERENCES = 4   # Klein's limit, the edited image counting as one


def edit_images(input_image: str | None, mask_image: str | None, reference_images: list[str],
                denoise: float | None, width: int | None, height: int | None) -> dict[str, str]:
    """Validate how the editing parameters combine; return the image slots to fill."""
    if mask_image and not input_image:
        raise ValueError("mask_image needs input_image (the image the mask applies to)")
    if denoise is not None and not input_image:
        raise ValueError("denoise needs input_image (without one there is nothing to start from)")
    if denoise is not None and not 0.05 <= denoise <= 1.0:
        raise ValueError("denoise must be between 0.05 and 1.0")
    if input_image and (width or height):
        raise ValueError("width and height cannot be set with input_image: the output keeps its size")
    limit = MAX_REFERENCES - (1 if input_image else 0)
    if len(reference_images) > limit:
        raise ValueError(f"at most {limit} reference_images" + (" with input_image" if input_image else ""))
    images = {f"ref{i + 1}": path for i, path in enumerate(reference_images)}
    if input_image:
        images["image"] = input_image
    if mask_image:
        images["mask"] = mask_image
    return images


def ksampler_steps(steps: int, denoise: float) -> tuple[int, float]:
    """KSampler (steps, denoise) inputs that make its partial denoise land close to the request.

    KSampler runs `steps` steps from the end of an int(steps / denoise) step schedule, so the
    denoise it really applies is steps / int(steps / denoise). With few steps that is coarse: at
    8 steps any denoise from 0.89 up becomes a full schedule from pure noise, which throws the
    input image away. This picks k steps (steps to 2 x steps) out of an n step schedule with k / n
    closest to the request and n > k (so below 1 the input always counts), and returns k and a
    denoise input that makes KSampler use exactly n.
    """
    if denoise >= 0.9999:
        return steps, 1.0
    best = min(((k, max(k + 1, round(k / denoise))) for k in range(steps, 2 * steps + 1)),
               key=lambda kn: (abs(kn[0] / kn[1] - denoise), kn[0]))
    k, n = best
    return k, k / (n + 0.5)


def schedule_steps(steps: int, denoise: float) -> tuple[int, float]:
    """Schedule length and denoise for SplitSigmasDenoise (custom samplers), which runs the last
    round(length * denoise) steps: the length (steps to 4 x steps) whose real denoise is closest,
    running at least `steps` steps where possible and never the full schedule below 1."""
    if denoise >= 0.9999:
        return steps, 1.0

    def run(t: int) -> int:
        return min(t - 1, max(1, round(t * denoise)))
    t = min(range(steps, 4 * steps + 1), key=lambda t: (abs(run(t) / t - denoise), run(t) < steps, t))
    return t, run(t) / t


def add_source(g: dict, p: dict) -> dict | None:
    """Add the input image (and mask) nodes; None when params carry no input image.

    Returns links: "original" (IMAGE at the input's size), "work" (IMAGE, about 1 MP,
    multiple of 16) and "mask" (MASK at the original's size, grown and feathered) or None.
    """
    images = p.get("images") or {}
    if not images.get("image"):
        return None
    g["edit_src"] = {"class_type": "LoadImage", "inputs": {"image": images["image"]}}
    g["edit_work"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["edit_src", 0], "upscale_method": "lanczos", "megapixels": WORK_MEGAPIXELS,
        "resolution_steps": MULTIPLE}}
    src = {"original": ["edit_src", 0], "work": ["edit_work", 0], "mask": None}
    if images.get("mask"):
        g["edit_mask_load"] = {"class_type": "LoadImage", "inputs": {"image": images["mask"]}}
        g["edit_mask_red"] = {"class_type": "ImageToMask", "inputs": {"image": ["edit_mask_load", 0], "channel": "red"}}
        g["edit_mask_fit"] = {"class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["edit_mask_red", 0], "resize_type": "match size", "resize_type.match": ["edit_src", 0],
            "resize_type.crop": "disabled", "scale_method": "bilinear"}}
        g["edit_mask_grow"] = {"class_type": "GrowMask", "inputs": {
            "mask": ["edit_mask_fit", 0], "expand": FEATHER, "tapered_corners": True}}
        g["edit_mask_img"] = {"class_type": "MaskToImage", "inputs": {"mask": ["edit_mask_grow", 0]}}
        g["edit_mask_blur"] = {"class_type": "ImageBlur", "inputs": {
            "image": ["edit_mask_img", 0], "blur_radius": FEATHER, "sigma": FEATHER / 2}}
        g["edit_mask"] = {"class_type": "ImageToMask", "inputs": {"image": ["edit_mask_blur", 0], "channel": "red"}}
        src["mask"] = ["edit_mask", 0]
    return src


def noise_mask(g: dict, latent: list, src: dict | None) -> list:
    """Limit sampling to the mask (SetLatentNoiseMask), or pass the latent through."""
    if not src or not src["mask"]:
        return latent
    g["edit_noise_mask"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": latent, "mask": src["mask"]}}
    return ["edit_noise_mask", 0]


def finish(g: dict, decoded: list, src: dict | None) -> list:
    """The image to save: decoded as is without an input image; otherwise back at the original's
    size, and with a mask pasted over the original so pixels outside the mask are untouched."""
    if not src:
        return decoded
    if src["mask"]:
        g["edit_composite"] = {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": src["original"], "source": decoded, "x": 0, "y": 0, "resize_source": True,
            "mask": src["mask"]}}
        return ["edit_composite", 0]
    g["edit_restore"] = {"class_type": "ResizeImageMaskNode", "inputs": {
        "input": decoded, "resize_type": "match size", "resize_type.match": src["original"],
        "resize_type.crop": "disabled", "scale_method": "lanczos"}}
    return ["edit_restore", 0]
