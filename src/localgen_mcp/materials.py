"""generate_material: a tileable PBR surface material from a prompt.

1. ComfyUI (workflows/seamless_material.py): base color, offset by half with wrap
   around, the seam cross repainted under a mask.
2. Blender check (blender/material_maps.py, mode "check"): seam metrics of the result.
   If the repainted centre still shows a line, the repaint runs again on the result
   (no offset, a fresh seed, a little more denoise), up to `seam_passes` times in all,
   keeping a pass only if it weakens the centre line.
3. Blender maps: height, normal, roughness, AO, the 3x3 tile and a lit preview.

This module only orchestrates; it imports no imaging libraries.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from . import blender_run, jobs, workflows
from .config import Settings

MARKER = "MATERIAL_MAPS_RESULT"
PREVIEWS = ("sphere", "plane", "none")


def slug(text: str, limit: int = 4) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())[:limit]
    return "_".join(words) or "material"


def validate(size: int, normal_strength: float, roughness: float, roughness_variation: float, ao_strength: float,
             seam_passes: int, seam_denoise: float, seam_width: float, preview: str) -> None:
    if not 512 <= size <= 2048 or size % 64:
        raise ValueError("size must be 512 to 2048 and a multiple of 64 (1024 or 2048 are typical)")
    if not 0.0 <= normal_strength <= 10.0:
        raise ValueError("normal_strength must be between 0 and 10")
    if not (0.0 <= roughness <= 1.0 and 0.0 <= roughness_variation <= 1.0 and 0.0 <= ao_strength <= 1.0):
        raise ValueError("roughness, roughness_variation and ao_strength must be between 0 and 1")
    if not 1 <= seam_passes <= 4:
        raise ValueError("seam_passes must be between 1 and 4")
    if not 0.3 <= seam_denoise <= 1.0:
        raise ValueError("seam_denoise must be between 0.3 and 1.0")
    if not 0.05 <= seam_width <= 0.3:
        raise ValueError("seam_width must be between 0.05 and 0.3 (a fraction of the size)")
    if preview not in PREVIEWS:
        raise ValueError(f"preview must be one of {PREVIEWS}")


def centre_score(seams: dict) -> float:
    """The stronger of the two centre lines, relative to the image's own neighbour differences."""
    return max(seams["x"]["centre_ratio"] or 0.0, seams["y"]["centre_ratio"] or 0.0)


async def _blender(settings: Settings, spec: dict, expect: Path) -> dict:
    return await blender_run.run(settings, "material_maps.py", [json.dumps(spec)], MARKER, expect)


async def generate(settings: Settings, prompt: str, *, size: int = 1024, reference_image: str | None = None,
                   tile_image: str | None = None, name: str | None = None, normal_strength: float = 1.0,
                   directx_normal: bool = False, invert_height: bool = False, roughness: float = 0.7,
                   roughness_variation: float = 0.25, ao_strength: float = 0.6, seam_passes: int = 2,
                   seam_denoise: float = 0.7, seam_width: float = 0.25, preview: str = "sphere",
                   seed: int | None = None, output_dir: str | None = None, idle_timeout_seconds: int | None = None,
                   timeout: float = 900, progress=None) -> dict:
    validate(size, normal_strength, roughness, roughness_variation, ao_strength, seam_passes, seam_denoise,
             seam_width, preview)
    if reference_image and tile_image:
        raise ValueError("give reference_image or tile_image, not both")
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    stem = f"{slug(name or prompt)}_{seed}"
    outdir = (Path(output_dir).expanduser() if output_dir else jobs.default_output_dir(settings, "material")) / stem
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = outdir / stem
    wf = workflows.get("material")
    images = {"reference": reference_image} if reference_image else {}
    if tile_image:
        images = {"source": tile_image}
    overrides = {"prompt": prompt, "size": size, "seed": seed, "seam_denoise": seam_denoise,
                 "seam_width": seam_width, "save_untiled": True}
    gen = await jobs.run_workflow(settings, wf, [overrides], str(outdir), idle_timeout_seconds, images=images,
                                  timeout=timeout, progress=progress)
    untiled = next(Path(p) for p in gen["paths"] if "_untiled_" in Path(p).name)
    tiled = next(Path(p) for p in gen["paths"] if "_untiled_" not in Path(p).name)
    base = prefix.with_name(stem + "_basecolor.png")
    tiled.replace(base)

    first = await _blender(settings, {"mode": "check", "base": str(base), "untiled": str(untiled)}, base)
    untiled.unlink(missing_ok=True)
    before, check = first["seams"]["untiled"], first["seams"]["base_color"]
    passes = [{"pass": 1, "seed": seed + 1, "denoise": seam_denoise, "seams": check, "kept": True}]
    denoise = seam_denoise
    # The edges are never repainted, so only the centre can need another pass.
    while not check["centre_ok"] and len(passes) < seam_passes:
        # The repainted cross still shows a line: repaint it again on the result, no new offset,
        # and keep the new image only if its centre line is weaker.
        denoise = round(min(1.0, denoise + 0.1), 2)
        pass_seed = seed + 1000 * len(passes)
        again = dict(overrides, roll=False, seed=pass_seed, seam_denoise=denoise, save_untiled=False)
        rerun = await jobs.run_workflow(settings, wf, [again], str(outdir), idle_timeout_seconds,
                                        images={"source": str(base)}, timeout=timeout, progress=progress)
        candidate = Path(rerun["paths"][0])
        new = (await _blender(settings, {"mode": "check", "base": str(candidate)}, candidate))["seams"]["base_color"]
        kept = centre_score(new) < centre_score(check)
        passes.append({"pass": len(passes) + 1, "seed": pass_seed + 1, "denoise": denoise, "seams": new,
                       "kept": kept})
        if kept:
            candidate.replace(base)
            check = new
        else:
            candidate.unlink(missing_ok=True)

    spec = {"mode": "maps", "base": str(base), "prefix": str(prefix), "normal_strength": normal_strength,
            "directx": directx_normal, "invert_height": invert_height, "roughness": roughness,
            "roughness_variation": roughness_variation, "ao_strength": ao_strength, "preview": preview}
    expect = Path(str(prefix) + "_tile3x3.png")
    maps = await _blender(settings, spec, expect)
    files = {"base_color": str(base), **maps["files"]}
    return {
        "workflow": gen["workflow"],
        "prompt": prompt,
        "seed": seed,
        "size": size,
        "files": files,
        "paths": list(files.values()),
        "output_dir": str(outdir),
        "normal_convention": maps["normal_convention"],
        "tiles": all(m["ok"] for m in maps["seams"].values()),
        "seams": {"untiled_base_color": before, **maps["seams"]},
        "seam_passes": passes,
        "stats": maps["stats"],
        "input_images": gen.get("input_images"),
        "generate_seconds": gen["total_seconds"],
        "maps_seconds": maps["seconds"],
        "gpu_memory_used_peak_mb": gen.get("gpu_memory_used_peak_mb"),
        "idle_timeout_seconds": gen.get("idle_timeout_seconds"),
    }
