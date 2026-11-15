"""Background removal: RGBA cut-outs with BiRefNet, through native ComfyUI nodes.

Two graphs, both batched (one chain per input image in a single prompt, so
BiRefNet loads once for the whole batch):

cut (WORKFLOW): LoadImage, RemoveBackground, then JoinImageWithAlpha into a
full size RGBA PNG. Mask conventions, which bit this repo before:
RemoveBackground outputs the FOREGROUND (subject = 1); LoadImage's mask output
is 1 - alpha (background = 1); JoinImageWithAlpha inverts its alpha input (it
expects the LoadImage convention), so it is fed InvertMask(foreground). With
trim, a small copy of the foreground mask is saved too, from which the caller
reads the subject's bounding box (no native node returns one as numbers).

finish (FINISH): lays out each cut-out from integers the caller computed with
layout(): crop to the box, paste centred on a transparent canvas (padding,
square), optional resize, optional solid background. The RGBA image goes
through ImageCrop, ImageCompositeMasked and ImageScale as one 4 channel image,
so colour and alpha cannot drift apart.
"""

from __future__ import annotations

import re

from . import Workflow

MAX_IMAGES = 64
SLOTS = tuple(f"image{i:02d}" for i in range(MAX_IMAGES))
PREVIEW_SIZE = 512          # long edge of the saved foreground mask used to find the box
SIZE_RANGE = (16, 4096)
_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")


def color_int(background: str) -> int:
    """'#rrggbb' (or 'rrggbb') as the integer EmptyImage takes."""
    m = _HEX.match(background.strip())
    if not m:
        raise ValueError(f"background must be a colour like '#ffffff', got {background!r}")
    return int(m.group(1), 16)


def filled(p: dict) -> list[tuple[int, str]]:
    """(slot index, uploaded name) for every filled slot, in slot order."""
    images = p.get("images", {})
    return [(i, images[s]) for i, s in enumerate(SLOTS) if images.get(s)]


def build_cut(p: dict) -> dict:
    g: dict = {"bg_model": {"class_type": "LoadBackgroundRemovalModel",
                            "inputs": {"bg_removal_name": "birefnet.safetensors"}}}
    for i, name in filled(p):
        load, frame, fg, inv, cut = (f"{k}{i:02d}" for k in ("load", "frame", "fg", "inv", "cut"))
        g[load] = {"class_type": "LoadImage", "inputs": {"image": name}}
        # LoadImage returns every frame of a GIF or animated WebP; use the first only.
        g[frame] = {"class_type": "ImageFromBatch", "inputs": {"image": [load, 0], "batch_index": 0, "length": 1}}
        g[fg] = {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["bg_model", 0], "image": [frame, 0]}}
        # JoinImageWithAlpha computes alpha = 1 - input, so give it the background (1 - foreground).
        g[inv] = {"class_type": "InvertMask", "inputs": {"mask": [fg, 0]}}
        g[cut] = {"class_type": "JoinImageWithAlpha", "inputs": {"image": [frame, 0], "alpha": [inv, 0]}}
        g[f"save{i:02d}"] = {"class_type": "SaveImage",
                             "inputs": {"images": [cut, 0], "filename_prefix": f"{p['prefix']}_{i:02d}c"}}
        if p.get("trim"):
            g[f"mimg{i:02d}"] = {"class_type": "MaskToImage", "inputs": {"mask": [fg, 0]}}
            g[f"msmall{i:02d}"] = {"class_type": "ImageScaleToMaxDimension", "inputs": {
                "image": [f"mimg{i:02d}", 0], "upscale_method": "area", "largest_size": PREVIEW_SIZE}}
            g[f"msave{i:02d}"] = {"class_type": "SaveImage", "inputs": {
                "images": [f"msmall{i:02d}", 0], "filename_prefix": f"{p['prefix']}_{i:02d}m"}}
    return g


def layout(width: int, height: int, box: tuple[int, int, int, int] | None, trim: bool, padding: float,
           square: bool, size: int | None) -> dict:
    """Crop rectangle, canvas and output size for one cut-out of width x height.

    box is the subject's bounding box (x0, y0, x1, y1, exclusive ends) or None for the whole image.
    With trim the crop is the box, plus padding (a fraction of its longer side) on every side; without,
    the whole image. square centres it on a square canvas. size scales the canvas so its longer side
    is size pixels.
    """
    x0, y0, x1, y1 = box if (trim and box) else (0, 0, width, height)
    rw, rh = x1 - x0, y1 - y0
    pad = round(padding * max(rw, rh)) if trim else 0
    cw, ch = rw + 2 * pad, rh + 2 * pad
    if square:
        cw = ch = max(cw, ch)
    out = {"x": x0, "y": y0, "w": rw, "h": rh, "canvas_w": cw, "canvas_h": ch,
           "ox": (cw - rw) // 2, "oy": (ch - rh) // 2, "out_w": cw, "out_h": ch}
    if size:
        scale = size / max(cw, ch)
        out["out_w"], out["out_h"] = max(1, round(cw * scale)), max(1, round(ch * scale))
    return out


def build_finish(p: dict) -> dict:
    g: dict = {}
    background = p.get("background")
    for i, name in filled(p):
        L = p["layout"][SLOTS[i]]
        n = lambda k: f"{k}{i:02d}"  # noqa: E731
        g[n("load")] = {"class_type": "LoadImage", "inputs": {"image": name}}
        # The cut-out is an RGBA PNG; LoadImage splits it into RGB and 1 - alpha, the convention
        # JoinImageWithAlpha expects, so the mask goes straight back in (no inversion here).
        g[n("rgba")] = {"class_type": "JoinImageWithAlpha", "inputs": {"image": [n("load"), 0], "alpha": [n("load"), 1]}}
        g[n("crop")] = {"class_type": "ImageCrop", "inputs": {
            "image": [n("rgba"), 0], "x": L["x"], "y": L["y"], "width": L["w"], "height": L["h"]}}
        # A transparent canvas: black RGB with alpha 1 - 1 = 0.
        g[n("blank")] = {"class_type": "EmptyImage", "inputs": {
            "width": L["canvas_w"], "height": L["canvas_h"], "batch_size": 1, "color": 0}}
        g[n("clear")] = {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": L["canvas_w"], "height": L["canvas_h"]}}
        g[n("canvas")] = {"class_type": "JoinImageWithAlpha", "inputs": {"image": [n("blank"), 0], "alpha": [n("clear"), 0]}}
        g[n("place")] = {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": [n("canvas"), 0], "source": [n("crop"), 0], "x": L["ox"], "y": L["oy"],
            "resize_source": False}}
        image = [n("place"), 0]
        if (L["out_w"], L["out_h"]) != (L["canvas_w"], L["canvas_h"]):
            # Lanczos goes through PIL as RGBA, which resamples with premultiplied alpha.
            g[n("scale")] = {"class_type": "ImageScale", "inputs": {
                "image": image, "upscale_method": "lanczos", "width": L["out_w"], "height": L["out_h"], "crop": "disabled"}}
            image = [n("scale"), 0]
        if background:
            g[n("bgfill")] = {"class_type": "EmptyImage", "inputs": {
                "width": L["out_w"], "height": L["out_h"], "batch_size": 1, "color": color_int(background)}}
            g[n("alpha")] = {"class_type": "ImageToMask", "inputs": {"image": image, "channel": "alpha"}}
            g[n("flat")] = {"class_type": "ImageCompositeMasked", "inputs": {
                "destination": [n("bgfill"), 0], "source": image, "x": 0, "y": 0, "resize_source": False,
                "mask": [n("alpha"), 0]}}
            image = [n("flat"), 0]
        g[n("save")] = {"class_type": "SaveImage", "inputs": {"images": image, "filename_prefix": f"{p['prefix']}_{i:02d}f"}}
    return g


WORKFLOW = Workflow(
    name="birefnet-cutout",
    modality="cutout",
    description="BiRefNet background removal (MIT), RGBA cut-outs, native ComfyUI nodes",
    models=["background_removal/birefnet.safetensors"],
    defaults={"trim": False},
    build=build_cut,
    image_slots=SLOTS,
    min_images=1,
    is_default=True,
)

# Second pass, no model: not in the registry, run directly by localgen_mcp.cutout.
FINISH = Workflow(
    name="cutout-layout",
    modality="cutout",
    description="Crop, pad, resize and flatten RGBA cut-outs",
    models=[],
    defaults={"layout": {}, "background": None},
    build=build_finish,
    image_slots=SLOTS,
    min_images=1,
)
