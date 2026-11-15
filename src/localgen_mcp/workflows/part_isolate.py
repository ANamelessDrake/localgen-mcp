"""Isolate one part of a concept image as a clean standalone picture, ready to mesh.

ImageCrop cuts the part's box out of the concept (the box pins down which part
and keeps its pose), and FLUX.2 Klein redraws only that part: the same angle,
proportions and colours, hidden areas completed, neighbouring parts and the
background removed. Asking Klein to isolate or recolour a part of the whole
image was unreliable (it moved, merged or missed parts); the crop fixes that.
Used by generate_parts, which meshes each result and assembles them.
"""

from __future__ import annotations

from . import Workflow
from .flux2_klein_4b import MODELS as KLEIN_MODELS, add_edit, add_loaders


def part_prompt(label: str, subject: str, description: str = "") -> str:
    what = f"{label} ({description})" if description else label
    return (f"Image 1 is a crop of a {subject}'s {what}, partly overlapped by neighbouring parts. Redraw only the "
            f"{label} as one complete standalone part: same angle, proportions, materials, colours and style, with "
            f"any hidden areas completed. Remove everything that is not part of the {label} (the body, other limbs, "
            "edges of other parts). Centered, plain light grey background, soft even lighting.")


def build(p: dict) -> dict:
    g: dict = {"concept": {"class_type": "LoadImage", "inputs": {"image": p["images"]["concept"]}},
               "crop": {"class_type": "ImageCrop", "inputs": {
                   "image": ["concept", 0], "x": int(p["x"]), "y": int(p["y"]),
                   "width": int(p["w"]), "height": int(p["h"])}}}
    add_loaders(g)
    size = int(p["size"])
    image = add_edit(g, "", p["prompt"], [["crop", 0]], int(p["seed"]), int(p["steps"]), size, size,
                     upscale="lanczos")
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": image, "filename_prefix": p["prefix"]}}
    return g


WORKFLOW = Workflow(
    name="part-isolate-klein",
    modality="part",
    description="Isolate one part of a concept image (crop plus FLUX.2 Klein 4B cleanup) for meshing",
    models=list(KLEIN_MODELS),
    defaults={"x": 0, "y": 0, "w": 512, "h": 512, "size": 1024, "steps": 4},
    limits={"size": (512, 1536), "steps": (2, 20)},
    build=build,
    image_slots=("concept",),
    min_images=1,
    is_default=True,
)
