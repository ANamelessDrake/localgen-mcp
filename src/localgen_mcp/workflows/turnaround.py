"""Turnaround: consistent front, left, back and right views of one object from a concept image.

One ComfyUI graph, native nodes only:

  FLUX.2 Klein (add_edit)   a clean front view from the concept (flat even lighting, no cast
                            shadows, plain background), then every other view with references
                            [clean front, concept] and a prompt naming the view and which way
                            the object faces in it.
  RemoveBackground          a foreground mask per view (BiRefNet).
  ImageCropToMask (x2)      framing for Pixal3D's fixed rig, below.
  SaveImage                 each framed view (RGBA), each raw Klein view, and a sheet.

Framing. Pixal3D's rig (nodes_trellis2._orbit_camera_to_world) wants square views with the
object centred and one scale in every view, the object's widest extent at 1/1.1 of the frame.
Generated views come at arbitrary scales, so each one is scaled by its silhouette height,
which is the same in every view of a standing object seen level from the side (the
assumption this rests on):
  1. Crop to the mask with a wide aspect (CANVAS_ASPECT): ImageCropToMask grows the box to that
     aspect anchored on the height whenever the silhouette is narrower than CANVAS_ASPECT times
     its height, so every view comes out with its silhouette CANVAS_HEIGHT / HEIGHT_PAD tall,
     centred on a CANVAS_ASPECT x CANVAS_HEIGHT canvas.
  2. Add the canvases' masks together. The union's box is as tall as any silhouette and as wide
     as the widest one, centred on the canvas.
  3. Crop every view to that one union box, square, with the rig's pad of 1.1. One crop for all
     views means one scale for all views, and the widest silhouette spans 1/1.1 of the frame.
plan() computes the same numbers from silhouette boxes; it documents the maths and the tests
check the graph against it.
"""

from __future__ import annotations

import re

from . import Workflow
from .flux2_klein_4b import MODELS as KLEIN_MODELS, add_edit, add_loaders

# View name -> azimuth in the rig's convention: Z up, azimuth 0 is the front, and the left view
# (90) sees the object's left side from +X, so in it the object faces image-left.
RIG_VIEWS = {"front": 0, "left": 90, "back": 180, "right": 270}
THREE_QUARTER_VIEWS = {"front_left": 45, "back_left": 135, "back_right": 225, "front_right": 315}
ALL_VIEWS = {**RIG_VIEWS, **THREE_QUARTER_VIEWS}

# Mirror pairs: for a left-right symmetric object, a view is its partner flipped horizontally.
MIRROR_OF = {"right": "left", "back_right": "back_left", "front_right": "front_left"}

CANVAS_HEIGHT = 1024
CANVAS_ASPECT = 4          # step 1 keeps the height anchor for silhouettes up to 4 times wider than tall
HEIGHT_PAD = 1.04          # a little room above and below so soft mask edges are not cut
RIG_PAD = 1.1              # the rig's frame is 1.1 times the object's widest extent
UNION_OUT = 4096           # ImageCropToMask shrinks inputs larger than its output first; keep the canvas whole

# Added to a prompt when generate_turnaround makes the concept itself.
CONCEPT_SUFFIX = (", the whole object in full view, standing upright, centered, flat even lighting, no cast "
                  "shadows, plain light grey background")

LIGHTING = ("Flat even lighting with no cast shadows, no shading gradients, no highlights, reflections or "
            "rim light: show the true surface colours. Plain light grey background, nothing else in the image.")

WHERE = {
    "front": "seen from directly in front: it faces the viewer",
    "left": ("seen from its left side, in profile: it faces the left edge of the image, its front is at the "
             "image's left and its back at the image's right, and we see its own left side"),
    "back": ("seen from directly behind: its back faces the viewer and its front and face are hidden; its own "
             "left side is on the image's left and its own right side on the image's right"),
    "right": ("seen from its right side, in profile: it faces the right edge of the image, its front is at the "
              "image's right and its back at the image's left, and we see its own right side"),
    "front_left": ("in a three-quarter view from its front left: it is turned 45 degrees to face the lower left "
                   "of the image, so we see its front and its own left side"),
    "front_right": ("in a three-quarter view from its front right: it is turned 45 degrees to face the lower right "
                    "of the image, so we see its front and its own right side"),
    "back_left": ("in a three-quarter view from behind on its left: it faces away from the viewer, turned towards "
                  "the left of the image, so we see its back and its own left side and not its front"),
    "back_right": ("in a three-quarter view from behind on its right: it faces away from the viewer, turned "
                   "towards the right of the image, so we see its back and its own right side and not its front"),
}


def front_prompt(subject: str) -> str:
    return (f"Image 1 is concept art of {subject}. Redraw exactly the same {subject} as a clean orthographic "
            "front view for a 3D modelling turnaround: the whole object, standing upright, centred, seen level "
            "from directly in front. Keep its design, proportions, parts, colours and materials unchanged. "
            + LIGHTING)


def view_prompt(view: str, subject: str) -> str:
    return (f"Image 1 is the front view of {subject}; image 2 is its concept art. Draw the same {subject}, "
            f"{WHERE[view]}. This is the {view.replace('_', ' ')} view of an orthographic 3D modelling "
            "turnaround: the whole object, standing upright, seen level, at the same size as in image 1. It "
            "has exactly the same parts, proportions, colours and materials as image 1, each part in its true "
            "place as the object turns. " + LIGHTING)


def views_for(three_quarter: bool) -> list[str]:
    return list(ALL_VIEWS if three_quarter else RIG_VIEWS)


def ordered(views: list[str]) -> list[str]:
    """Views by azimuth, the order of the sheet."""
    return sorted(views, key=lambda v: ALL_VIEWS[v])


def plan(boxes: dict[str, tuple[float, float]], size: int = 1024) -> dict:
    """The framing the graph applies, from each view's silhouette (width, height) in its own pixels.

    Returns each view's scale (output pixels per input pixel) and its silhouette size in the
    output, plus the frame's side in canvas pixels. Every silhouette ends up the same height,
    centred, and the widest one spans size / RIG_PAD.
    """
    out = {}
    for view, (w, h) in boxes.items():
        if h <= 0 or w <= 0:
            raise ValueError(f"{view}: empty silhouette")
        if w / h > CANVAS_ASPECT:
            raise ValueError(f"{view}: silhouette wider than {CANVAS_ASPECT} times its height")
        out[view] = CANVAS_HEIGHT / (h * HEIGHT_PAD)  # step 1: canvas pixels per input pixel
    union_w = max(boxes[v][0] * s for v, s in out.items())
    union_h = max(boxes[v][1] * s for v, s in out.items())
    side = max(union_w, union_h) * RIG_PAD  # step 3: one square crop for every view
    return {"frame_side_canvas_px": side,
            "views": {v: {"scale": s * size / side, "width": boxes[v][0] * s * size / side,
                          "height": boxes[v][1] * s * size / side} for v, s in out.items()}}


def build(p: dict) -> dict:
    views = ordered(views_for(bool(p.get("three_quarter"))))
    size, steps, seed = int(p["view_size"]), int(p["steps"]), int(p["seed"])
    subject = p["subject"]
    symmetric = bool(p.get("symmetric"))
    g: dict = {"concept": {"class_type": "LoadImage", "inputs": {"image": p["images"]["concept"]}}}
    add_loaders(g)
    concept = ["concept", 0]
    front = add_edit(g, "front_", front_prompt(subject), [concept], seed, steps, size, size, upscale="lanczos")
    raw = {"front": front}
    for i, view in enumerate(views):
        if view == "front" or (symmetric and view in MIRROR_OF):
            continue
        refs = [front, concept] if p.get("ref_order", "front_first") == "front_first" else [concept, front]
        init, denoise = None, 1.0
        if view == "back" and p.get("back_start") == "mirror":
            # A rough start: the front flipped has the back's silhouette, left on the image's left.
            g["back_flip"] = {"class_type": "ImageFlip", "inputs": {"image": front, "flip_method": "y-axis: horizontally"}}
            init, denoise = ["back_flip", 0], float(p["back_denoise"])
        raw[view] = add_edit(g, f"{view}_", view_prompt(view, subject), refs, seed + 1 + i, steps, size, size,
                             init=init, denoise=denoise, upscale="lanczos")
    for view, partner in MIRROR_OF.items():
        if symmetric and view in views:
            g[f"{view}_mirror"] = {"class_type": "ImageFlip", "inputs": {
                "image": raw[partner], "flip_method": "y-axis: horizontally"}}
            raw[view] = [f"{view}_mirror", 0]

    g["bg_model"] = {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": "birefnet.safetensors"}}
    bg = p["background"]
    union = None
    canvas = {}
    for view in views:
        g[f"{view}_mask"] = {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["bg_model", 0],
                                                                          "image": raw[view]}}
        g[f"{view}_mask_img"] = {"class_type": "MaskToImage", "inputs": {"mask": [f"{view}_mask", 0]}}
        # Step 1: height-anchored crop onto a wide canvas, for the colour image and its mask alike.
        for kind, image, fill in (("canvas", raw[view], bg), ("canvas_mask", [f"{view}_mask_img", 0], "#000000")):
            g[f"{view}_{kind}"] = {"class_type": "ImageCropToMask", "inputs": {
                "images": image, "masks": [f"{view}_mask", 0], "width": CANVAS_ASPECT * CANVAS_HEIGHT,
                "height": CANVAS_HEIGHT, "pad_factor": HEIGHT_PAD, "grow_mask": 0, "background": fill}}
        g[f"{view}_canvas_m"] = {"class_type": "ImageToMask", "inputs": {"image": [f"{view}_canvas_mask", 0],
                                                                         "channel": "red"}}
        canvas[view] = [f"{view}_canvas_m", 0]
        # Step 2: the union of every view's canvas mask.
        if union is None:
            union = canvas[view]
        else:
            g[f"{view}_union"] = {"class_type": "MaskComposite", "inputs": {
                "destination": union, "source": canvas[view], "x": 0, "y": 0, "operation": "add"}}
            union = [f"{view}_union", 0]
    framed = {}
    for view in views:
        # Step 3: the same square crop of every canvas, the widest silhouette at 1/RIG_PAD.
        for kind, image, fill in (("frame", [f"{view}_canvas", 0], bg), ("frame_mask", [f"{view}_canvas_mask", 0], "#000000")):
            g[f"{view}_{kind}"] = {"class_type": "ImageCropToMask", "inputs": {
                "images": image, "masks": union, "width": UNION_OUT, "height": UNION_OUT,
                "pad_factor": RIG_PAD, "grow_mask": 0, "background": fill}}
            g[f"{view}_{kind}_s"] = {"class_type": "ImageScale", "inputs": {
                "image": [f"{view}_{kind}", 0], "upscale_method": "area", "width": size, "height": size,
                "crop": "disabled"}}
        g[f"{view}_alpha"] = {"class_type": "ImageToMask", "inputs": {"image": [f"{view}_frame_mask_s", 0],
                                                                      "channel": "red"}}
        # JoinImageWithAlpha inverts its alpha input (LoadImage's convention), so give it the background.
        g[f"{view}_alpha_inv"] = {"class_type": "InvertMask", "inputs": {"mask": [f"{view}_alpha", 0]}}
        g[f"{view}_rgba"] = {"class_type": "JoinImageWithAlpha", "inputs": {
            "image": [f"{view}_frame_s", 0], "alpha": [f"{view}_alpha_inv", 0]}}
        framed[view] = [f"{view}_frame_s", 0]
        g[f"{view}_save"] = {"class_type": "SaveImage", "inputs": {
            "images": [f"{view}_rgba", 0], "filename_prefix": f"{p['prefix']}_view_{view}"}}
        g[f"{view}_save_raw"] = {"class_type": "SaveImage", "inputs": {
            "images": raw[view], "filename_prefix": f"{p['prefix']}_raw_{view}"}}
    sheet = framed[views[0]]
    for i, view in enumerate(views[1:]):
        g[f"sheet{i}"] = {"class_type": "ImageStitch", "inputs": {
            "image1": sheet, "image2": framed[view], "direction": "right", "match_image_size": True,
            "spacing_width": 0, "spacing_color": "white"}}
        sheet = [f"sheet{i}", 0]
    g["sheet_save"] = {"class_type": "SaveImage", "inputs": {"images": sheet, "filename_prefix": f"{p['prefix']}_sheet"}}
    return g


_NAME = re.compile(r"_(view|raw)_([a-z_]+?)_\d+_\.png$|_(sheet)_\d+_\.png$")


def classify(paths: list[str]) -> dict:
    """Sort a run's downloaded files into framed views, raw views and the sheet, by file name."""
    out: dict = {"views": {}, "raw": {}, "sheet": None}
    for path in paths:
        m = _NAME.search(path)
        if not m:
            continue
        if m.group(3):
            out["sheet"] = path
        elif m.group(2) in ALL_VIEWS:
            out["views" if m.group(1) == "view" else "raw"][m.group(2)] = path
    for key in ("views", "raw"):
        out[key] = {v: out[key][v] for v in ordered(list(out[key]))}
    return out


_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def check(p: dict) -> None:
    if not _HEX.match(str(p.get("background", ""))):
        raise ValueError("background must be a colour like #d8d8d8")
    if p.get("back_start") not in ("none", "mirror"):
        raise ValueError("back_start must be 'none' or 'mirror'")
    if p.get("ref_order") not in ("front_first", "concept_first"):
        raise ValueError("ref_order must be 'front_first' or 'concept_first'")
    if not 0.5 <= float(p.get("back_denoise", 0.9)) <= 1.0:
        raise ValueError("back_denoise must be between 0.5 and 1.0")
    if not str(p.get("subject", "")).strip():
        raise ValueError("subject must name the object")


def _build(p: dict) -> dict:
    check(p)
    return build(p)


WORKFLOW = Workflow(
    name="turnaround-klein",
    modality="turnaround",
    description=("Consistent front, left, back and right views (optionally three-quarter views) of one object "
                 "from a concept image, flat lit, framed for Pixal3D's rig: FLUX.2 Klein 4B (Apache 2.0) "
                 "plus BiRefNet masks"),
    models=list(KLEIN_MODELS) + ["background_removal/birefnet.safetensors"],
    defaults={"subject": "the object", "three_quarter": False, "symmetric": False, "view_size": 1024, "steps": 4,
              "background": "#d8d8d8", "ref_order": "front_first", "back_start": "none", "back_denoise": 0.9},
    limits={"view_size": (512, 1536), "steps": (2, 20)},
    build=_build,
    image_slots=("concept",),
    min_images=1,
    is_default=True,
)
