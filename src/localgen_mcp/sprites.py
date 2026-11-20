"""Pixel art sprite tools: generate_sprite, generate_animation, generate_tileset, pack_sheet, pixelize.

Built on the existing pipeline, no second one:
  drawing      the image workflows (Z-Image Turbo; FLUX.2 Klein with style references or an edit),
               the seamless material workflow for ground tiles, and generate_turnaround for views
  cut-out      cutout.remove_background (BiRefNet)
  pixel pass   pixel_ops.py, a numpy and Pillow helper run as a child process with this
               interpreter; run_ops reads its final "PIXEL_OPS_RESULT {json}" line and logs (never
               returns) anything else it prints

This module only orchestrates; like the rest of the MCP server process it imports no imaging
library. Nothing here is specific to one game: style, palette, cell size, outline, shading steps
and the tile kinds are parameters, and STYLES and TILE_PRESETS only hold defaults.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import blender_run, cutout, jobs, workflows
from .config import Settings
from .workflows.image_edit import edit_images

logger = logging.getLogger("localgen_mcp.sprites")

MARKER = "PIXEL_OPS_RESULT"
ERROR_MARKER = "PIXEL_OPS_ERROR"
HELPER = Path(__file__).parent / "pixel_ops.py"
PALETTE_DIR = Path(__file__).parent / "palettes"
TIMEOUT = int(os.environ.get("LOCALGEN_TIMEOUT_PIXEL", 600))
DRAW_SIZE = 1024
SIZE_RANGE = (8, 512)
REF_BACKGROUND = "#d8d8d8"
_HEX = re.compile(r"^#?[0-9a-fA-F]{6}$")


class PixelError(RuntimeError):
    pass


# -- styles ------------------------------------------------------------------

# A style is prompt wording plus the pixel pass defaults that suit it. Any parameter given to a tool
# overrides the preset; a style that is not a preset name is used as free text prompt wording.
STYLES: dict[str, dict[str, Any]] = {
    "cartoon": {
        "prompt": ("warm cartoon game art, chunky thick dark outlines, flat colours with simple cel shading, "
                   "bold saturated palette, simple readable shapes with slightly oversized heads, hands and "
                   "tools, seen from a slightly raised three-quarter view as in a 2D strategy game"),
        "outline": True, "outline_color": "auto", "shading_steps": 3, "saturation": 1.15},
    "grim": {
        "prompt": ("dark gritty roguelike game art, muted desaturated earthy colours, heavy black outlines, "
                   "harsh two tone shading, strong simple silhouette, seen from the side"),
        "outline": True, "outline_color": "darkest", "shading_steps": 2, "saturation": 0.75},
    "painterly": {
        "prompt": ("soft painterly game art, visible brush strokes, rich colours, gentle light and shadow, "
                   "no outlines, seen from a slightly raised three-quarter view"),
        "outline": False, "outline_color": "auto", "shading_steps": None, "saturation": 1.0},
    "flat": {
        "prompt": "minimal flat vector game art, flat colours with no shading, clean geometric shapes, no outlines",
        "outline": False, "outline_color": "auto", "shading_steps": 1, "saturation": 1.0},
}
DEFAULT_STYLE = "cartoon"
FREE_TEXT_STYLE = {"outline": True, "outline_color": "auto", "shading_steps": None, "saturation": 1.0}


def resolve_style(style: str | None, outline: bool | None = None, outline_color: str | None = None,
                  shading_steps: int | None = None, saturation: float | None = None) -> dict:
    """The style preset (or free text) with any explicit parameter applied over it."""
    name = (style or DEFAULT_STYLE).strip()
    preset = STYLES.get(name.lower())
    out = dict(preset) if preset else {**FREE_TEXT_STYLE, "prompt": name}
    out["name"] = name.lower() if preset else "custom"
    for key, value in (("outline", outline), ("outline_color", outline_color), ("shading_steps", shading_steps),
                       ("saturation", saturation)):
        if value is not None:
            out[key] = value
    check_pixel_options(outline_color=out["outline_color"], shading_steps=out["shading_steps"],
                        saturation=out["saturation"])
    return out


def shading_words(steps: int | None) -> str:
    if not steps:
        return ""
    if steps == 1:
        return " Flat colours, no shading."
    if steps == 3:
        return " Each colour has exactly three tones: a shadow, a base and one highlight."
    return f" Each colour has exactly {steps} flat tones, no gradients."


def sprite_prompt(subject: str, style: dict) -> str:
    return (f"{subject.strip().rstrip('.')}. {style['prompt']}.{shading_words(style['shading_steps'])} A single "
            "game sprite showing the whole subject, centred with space around it, isolated on a plain flat light "
            "grey background, no ground, no cast shadow, no text, no border, no frame.")


def style_ref_prompt(subject: str, style: dict, count: int) -> str:
    imgs = "image 1" if count == 1 else f"images 1 to {count}"
    return (f"Use {imgs} only as a style reference: match their art style, line weight, shading and palette, but "
            f"draw a new subject. {sprite_prompt(subject, style)}")


def tile_prompt(subject: str, style: dict) -> str:
    return (f"{subject.strip().rstrip('.')}. {style['prompt']}.{shading_words(style['shading_steps'])} A single "
            "square game map tile seen from directly above, filling the whole frame edge to edge, no border, "
            "no frame, no text, no perspective.")


# -- options -----------------------------------------------------------------


def check_pixel_options(size: int | None = None, max_colors: int | None = None, outline_color: str | None = None,
                        shading_steps: int | None = None, saturation: float | None = None,
                        blur: float | None = None, fit: str | None = None, anchor: str | None = None) -> None:
    if size is not None and not SIZE_RANGE[0] <= int(size) <= SIZE_RANGE[1]:
        raise ValueError(f"size must be between {SIZE_RANGE[0]} and {SIZE_RANGE[1]} pixels")
    if max_colors is not None and not 2 <= int(max_colors) <= 256:
        raise ValueError("max_colors must be between 2 and 256")
    if outline_color is not None and outline_color not in ("auto", "darkest") and not _HEX.match(outline_color):
        raise ValueError('outline_color must be "auto", "darkest" or a colour like "#1f1720"')
    if shading_steps is not None and not 1 <= int(shading_steps) <= 8:
        raise ValueError("shading_steps must be between 1 and 8")
    if saturation is not None and not 0.0 <= float(saturation) <= 3.0:
        raise ValueError("saturation must be between 0 and 3 (1 keeps it)")
    if blur is not None and not 0.0 <= float(blur) <= 3.0:
        raise ValueError("blur must be between 0 and 3 (0.5 is the default; 0 is plain nearest neighbour)")
    if fit is not None and fit not in ("trim", "frame", "fill"):
        raise ValueError('fit must be "trim", "frame" or "fill"')
    if anchor is not None and anchor not in ("center", "bottom"):
        raise ValueError('anchor must be "center" or "bottom"')


def shipped_palettes() -> dict[str, Path]:
    return {p.stem: p for p in sorted(PALETTE_DIR.glob("*.gpl"))}


PALETTE_SUFFIXES = (".gpl", ".hex", ".txt", ".png")
PALETTE_MAX_BYTES = 1 << 20
_PNG = b"\x89PNG\r\n\x1a\n"


def palette_roots(settings: Settings) -> list[Path]:
    """Where a palette file may be read from: the shipped palettes, the output folder and the
    install's tmp folder (inputs staged for tools). A palette path is never read from anywhere
    else, so the parameter cannot be pointed at arbitrary files."""
    return [PALETTE_DIR.resolve(), settings.output_dir.resolve(), (settings.root / "tmp").resolve()]


def read_text_palette(path: Path) -> list[str]:
    """Colours of a GPL file or a hex list (one #rrggbb per line, ";" comments), parsed strictly:
    every line must be a colour, a comment, blank, or a GPL header line, else the file is rejected.
    Errors give only a line number, never the line."""
    try:
        lines = path.read_bytes().decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise ValueError("palette file is not UTF-8 text") from None
    gpl = bool(lines) and lines[0].strip() == "GIMP Palette"
    cols = []
    for n, line in enumerate(lines[1:] if gpl else lines, start=2 if gpl else 1):
        text = line.strip()
        if not text or text.startswith("#" if gpl else ";"):
            continue
        if gpl:
            if re.match(r"^(Name|Columns):", text):
                continue
            parts = text.split()
            if len(parts) >= 3 and all(p.isdigit() and int(p) <= 255 for p in parts[:3]):
                cols.append("#%02x%02x%02x" % tuple(int(p) for p in parts[:3]))
                continue
        else:
            word = text.split()[0]
            if _HEX.match(word):
                cols.append("#" + word.lstrip("#").lower())
                continue
        raise ValueError(f"palette file line {n} is not a colour")
    if not 2 <= len(cols) <= 256:
        raise ValueError(f"a palette file needs 2 to 256 colours, found {len(cols)}")
    return cols


def resolve_palette(settings: Settings, palette: str | list[str] | None) -> str | list[str] | None:
    """None for an auto palette; else a list of "#rrggbb", or the path of a PNG whose colours the
    helper reads. palette: a shipped palette name ("moriveil"), a list of colours, or a .gpl, .hex,
    .txt or .png file under palette_roots(). Text files are parsed here, so only colours reach the
    helper; errors never quote file contents."""
    if palette is None or palette == "auto" or palette == []:
        return None
    if isinstance(palette, list):
        if not 2 <= len(palette) <= 256 or not all(isinstance(c, str) and _HEX.match(c.strip()) for c in palette):
            raise ValueError("a palette list must hold 2 to 256 colours like '#a83232'")
        return ["#" + c.strip().lstrip("#") for c in palette]
    shipped = shipped_palettes()
    if palette in shipped:
        return read_text_palette(shipped[palette])
    path = Path(palette).expanduser().resolve()
    if path.suffix.lower() not in PALETTE_SUFFIXES:
        raise ValueError(f"palette must be a shipped name ({sorted(shipped)}) or a "
                         f"{', '.join(PALETTE_SUFFIXES)} file")
    if not any(path.is_relative_to(root) for root in palette_roots(settings)):
        raise ValueError(f"a palette file must be under {settings.output_dir} or {settings.root / 'tmp'} "
                         "(copy it there), or be a shipped palette")
    if not path.is_file():
        raise ValueError(f"palette file not found: {path}")
    if path.stat().st_size > PALETTE_MAX_BYTES:
        raise ValueError("palette file too large")
    if path.suffix.lower() == ".png":
        if path.read_bytes()[:8] != _PNG:
            raise ValueError("palette .png is not a PNG image")
        return str(path)
    return read_text_palette(path)


def image_input(raw: str, what: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{what} not found: {path}")
    if not jobs.is_image_file(path):  # checked by content
        raise ValueError(f"{what} is not a PNG, JPEG, WebP, GIF, BMP or TIFF image: {path}")
    return path


def slug(text: str, limit: int = 5) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())[:limit]
    return "_".join(words) or "sprite"


def unique_stem(outdir: Path, stem: str) -> str:
    """stem, or stem_2, stem_3, ... so earlier results in the folder are never overwritten."""
    n, out = 1, stem
    while any(outdir.glob(out + ".*")) or any(outdir.glob(out + "_*.png")):
        n += 1
        out = f"{stem}_{n}"
    return out


def out_dir(settings: Settings, output_dir: str | None) -> Path:
    d = Path(output_dir).expanduser() if output_dir else jobs.default_output_dir(settings, "sprite")
    d.mkdir(parents=True, exist_ok=True)
    return d


# -- helper process ----------------------------------------------------------


async def run_ops(settings: Settings, spec: dict, expect: Path) -> dict:
    """Run one pixel_ops operation in a child process (this interpreter, -I) and return its report.

    The child's output is logged here, never returned: a failure reaches the client only as the
    helper's own PIXEL_OPS_ERROR message (a ValueError raised in pixel_ops, which never quotes file
    contents) or a short generic error."""
    work = settings.root / "tmp"
    work.mkdir(parents=True, exist_ok=True)
    spec_path = work / f"pixel-{uuid.uuid4().hex[:10]}.json"
    spec_path.write_text(json.dumps(spec))
    cmd = [blender_run.PYTHON, "-I", str(HELPER), str(spec_path)]
    op = spec.get("op")
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE,
                                                    env={**os.environ, "TMPDIR": str(work)})
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise PixelError(f"pixel pass ({op}) timed out after {TIMEOUT} s") from None
    finally:
        spec_path.unlink(missing_ok=True)
    stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")
    lines = stdout.splitlines()
    for line in reversed(lines):
        if line.startswith(ERROR_MARKER + " "):
            raise PixelError(f"pixel pass ({op}): {json.loads(line[len(ERROR_MARKER) + 1:])}")
    if proc.returncode == 0 and expect.is_file():
        for line in reversed(lines):
            if line.startswith(MARKER + " "):
                return json.loads(line[len(MARKER) + 1:])
    logger.error("pixel_ops %s failed (exit %s):\n%s", op, proc.returncode,
                 "\n".join((stdout + stderr).splitlines()[-40:]))
    raise PixelError(f"pixel pass ({op}) failed (exit {proc.returncode}); details are in the server log")


def pixel_spec(size: int, palette, max_colors: int, style: dict, blur: float, items: list[dict],
               palette_out: Path | None = None, **extra) -> dict:
    return {"op": "pixelize", "size": size, "palette": palette, "max_colors": max_colors,
            "outline": bool(style["outline"]), "outline_color": style["outline_color"],
            "shading_steps": style["shading_steps"], "saturation": style["saturation"], "blur": blur,
            "items": items, "palette_out": str(palette_out) if palette_out else None, **extra}


# -- drawing -----------------------------------------------------------------

Progress = Callable[[float, float, str], Awaitable[None]] | None


KLEIN = "flux2-klein-4b"


async def draw(settings: Settings, prompts: list[str], seeds: list[int], outdir: Path, idle: int | None,
               timeout: float, progress: Progress, refs: list[str] | None = None,
               input_image: str | None = None) -> dict:
    """One image per prompt, all queued at once (one model load), with the image workflows
    generate_image uses: Z-Image Turbo, or FLUX.2 Klein with reference images or an image to edit
    (input_image is then image 1 and the output keeps its size)."""
    images = edit_images(input_image, None, refs or [], None, None, None)
    wf = workflows.pick("image", KLEIN if input_image else None, set(images))
    size = {} if input_image else {"width": DRAW_SIZE, "height": DRAW_SIZE}
    job_list = [{"prompt": p, "seed": s, **size} for p, s in zip(prompts, seeds)]
    return await jobs.run_workflow(settings, wf, job_list, str(outdir), idle, images=images, timeout=timeout,
                                   progress=progress)


async def cut(settings: Settings, paths: list[str], outdir: Path, idle: int | None, timeout: float,
              progress: Progress) -> list[str]:
    """RGBA cut-outs (BiRefNet), in input order, written to outdir."""
    res = await cutout.remove_background(settings, paths, output_dir=str(outdir), idle_timeout_seconds=idle,
                                         timeout=timeout, progress=progress)
    return [o["path"] for o in res["outputs"]]


def _move(src: str, dst: Path) -> str:
    shutil.move(src, dst)
    return str(dst)


# -- generate_sprite ---------------------------------------------------------


async def generate_sprite(settings: Settings, prompt: str, *, size: int = 64, palette=None, max_colors: int = 32,
                          style: str | None = None, outline: bool | None = None, outline_color: str | None = None,
                          shading_steps: int | None = None, saturation: float | None = None, blur: float = 0.5,
                          style_images: list[str] | None = None, views: str = "front", anchor: str = "bottom",
                          seed: int | None = None, name: str | None = None, output_dir: str | None = None,
                          idle_timeout_seconds: int | None = None, timeout: float = 900, progress: Progress = None,
                          turnaround: Callable[..., Awaitable[dict]] | None = None) -> dict:
    if not prompt or not prompt.strip():
        raise ValueError("give a prompt describing the sprite")
    if views not in ("front", "turnaround"):
        raise ValueError('views must be "front" or "turnaround"')
    check_pixel_options(size=size, max_colors=max_colors, blur=blur, anchor=anchor)
    st = resolve_style(style, outline, outline_color, shading_steps, saturation)
    pal = resolve_palette(settings, palette)
    refs = [str(image_input(p, "style image")) for p in (style_images or []) if p]
    if len(refs) > 4:
        raise ValueError("at most 4 style_images")
    seed = random.randint(0, 2**31 - 1) if seed is None else int(seed)
    outdir = out_dir(settings, output_dir)
    stem = unique_stem(outdir, f"{slug(name or prompt)}_{seed}")
    t0 = time.time()
    text = style_ref_prompt(prompt, st, len(refs)) if refs else sprite_prompt(prompt, st)
    drawn = await draw(settings, [text], [seed], outdir, idle_timeout_seconds, timeout, progress, refs=refs)
    raw = _move(drawn["paths"][0], outdir / f"{stem}_raw.png")
    result: dict[str, Any] = {"prompt": prompt, "full_prompt": text, "seed": seed, "style": st, "size": size,
                              "output_dir": str(outdir), "raw": raw, "draw_workflow": drawn["workflow"]}
    if views == "turnaround":
        if turnaround is None:
            raise ValueError("views='turnaround' is not available here")
        ta = await turnaround(concept_image=raw, subject=prompt, seed=seed, output_dir=str(outdir / f"{stem}_views"),
                              idle_timeout_seconds=idle_timeout_seconds)
        sources = dict(ta["views"])
        # One framing for all views (generate_turnaround frames them at one scale), so "frame" keeps it.
        items = [{"input": p, "output": str(outdir / f"{stem}_{v}.png"), "preview": str(outdir / f"{stem}_{v}_x8.png"),
                  "fit": "frame"} for v, p in sources.items()]
        result["turnaround"] = {k: ta.get(k) for k in ("sheet", "clean_front", "concept_image")}
    else:
        [source] = await cut(settings, [raw], outdir, idle_timeout_seconds, timeout, progress)
        source = _move(source, outdir / f"{stem}_source.png")
        sources = {"front": source}
        items = [{"input": source, "output": str(outdir / f"{stem}.png"), "preview": str(outdir / f"{stem}_x8.png"),
                  "fit": "trim", "anchor": anchor}]
    palette_out = outdir / f"{stem}_palette.gpl" if pal is None else None
    px = await run_ops(settings, pixel_spec(size, pal, max_colors, st, blur, items, palette_out),
                       Path(items[-1]["output"]))
    outs = {v: o for v, o in zip(sources, px["outputs"])}
    result.update({
        "sprite": outs["front"]["path"] if "front" in outs else px["outputs"][0]["path"],
        "preview": (outs.get("front") or px["outputs"][0]).get("preview"),
        "source": sources.get("front"),
        "views": {v: o["path"] for v, o in outs.items()} if views == "turnaround" else None,
        "view_sources": sources if views == "turnaround" else None,
        "palette": px["palette"], "palette_file": px.get("palette_file") or (pal if isinstance(pal, str) else None),
        "pixel": [{k: o[k] for k in ("path", "scale", "colours_used", "opaque_pixels", "clipped", "outline_color")}
                  for o in px["outputs"]],
        "paths": [o["path"] for o in px["outputs"]] + [raw] + list(sources.values()),
        "total_seconds": round(time.time() - t0, 2),
    })
    return result


# -- generate_animation ------------------------------------------------------

# Pose wording per action, played in order; frame i of n uses phase floor(i * len / n). iou_min is the
# silhouette overlap with the base sprite a frame must reach (a walk moves the legs; a fall moves
# everything), hist_max the largest colour histogram distance.
ACTIONS: dict[str, dict[str, Any]] = {
    "idle": {"fps": 6, "loop": True, "iou_min": 0.7, "hist_max": 0.3, "phases": [
        "standing relaxed in its normal pose, breathing in, chest and shoulders slightly raised",
        "standing relaxed in its normal pose, fully breathed in, head a little higher",
        "standing relaxed in its normal pose, breathing out, shoulders slightly lowered",
        "standing relaxed in its normal pose, fully breathed out, body a little lower"]},
    "walk": {"fps": 10, "loop": True, "iou_min": 0.5, "hist_max": 0.35, "phases": [
        "walking: right foot forward touching the ground, left foot behind, arms swinging opposite",
        "walking: weight down on the right foot, left foot lifting off behind",
        "walking: passing position, left leg swinging forward past the right leg, body at its highest",
        "walking: left foot forward touching the ground, right foot behind, arms swinging opposite",
        "walking: weight down on the left foot, right foot lifting off behind",
        "walking: passing position, right leg swinging forward past the left leg, body at its highest"]},
    "chop": {"fps": 10, "loop": True, "iou_min": 0.45, "hist_max": 0.35, "phases": [
        "ready stance, holding its tool in both hands at waist height",
        "winding up, lifting its tool back over its shoulder",
        "tool raised high overhead, body leaning back",
        "swinging the tool down and forward hard",
        "the tool striking low in front of it at full extension",
        "pulling the tool back up, returning to the ready stance"]},
    "attack": {"fps": 12, "loop": False, "iou_min": 0.45, "hist_max": 0.35, "phases": [
        "battle ready stance, weapon held in front",
        "drawing the weapon back to strike, weight on the back foot",
        "lunging forward, weapon swinging through",
        "full extension of the strike, weapon far forward",
        "follow through, weapon low after the strike",
        "recovering back into the battle ready stance"]},
    "hurt": {"fps": 10, "loop": False, "iou_min": 0.5, "hist_max": 0.35, "phases": [
        "flinching from a hit, head snapped back, eyes shut",
        "recoiling backwards off balance, arms up",
        "staggering, regaining balance",
        "back in its normal pose, wary"]},
    "die": {"fps": 8, "loop": False, "iou_min": 0.15, "hist_max": 0.4, "phases": [
        "hit hard, recoiling backwards",
        "staggering and losing balance, weapon or tool slipping",
        "falling to its knees",
        "toppling over sideways",
        "lying on the ground",
        "lying still on the ground, motionless"]},
}
GENERIC_FPS = 8


def frame_poses(action: str, frames: int, frame_prompts: list[str] | None) -> tuple[list[str], dict]:
    """The pose text for each frame and the action's settings (fps, loop, thresholds)."""
    preset = ACTIONS.get(action.lower().strip())
    info = {k: v for k, v in (preset or {"fps": GENERIC_FPS, "loop": True, "iou_min": 0.45,
                                         "hist_max": 0.35}).items() if k != "phases"}
    if frame_prompts:
        if len(frame_prompts) != frames:
            raise ValueError(f"frame_prompts must have one entry per frame ({frames})")
        return [str(p) for p in frame_prompts], info
    if preset:
        phases = preset["phases"]
        return [phases[(i * len(phases)) // frames] for i in range(frames)], info
    return [f"in the pose of frame {i + 1} of {frames} of a smooth {action} animation cycle" for i in range(frames)], info


def frame_prompt(subject: str, action: str, pose: str, index: int, frames: int, has_prev: bool) -> str:
    prev = " Image 2 is the previous frame of the same animation." if has_prev else ""
    return (f"Image 1 shows {subject}.{prev} Draw exactly the same character with the same design, proportions, "
            f"colours, clothing and equipment, at the same size, framing and camera angle as image 1, facing the "
            f"same way, now {pose}. This is frame {index + 1} of {frames} of a {action} animation for a 2D game. "
            "Whole body in view, plain flat light grey background, no ground, no cast shadow, no text.")


def score(m: dict) -> float:
    return m["iou"] - m["hist_distance"]


def passes(m: dict, iou_min: float, hist_max: float) -> bool:
    return m["iou"] >= iou_min and m["hist_distance"] <= hist_max


async def generate_animation(settings: Settings, sprite_or_prompt: str, action: str, *, frames: int = 6,
                             size: int = 64, palette=None, max_colors: int = 32, style: str | None = None,
                             outline: bool | None = None, outline_color: str | None = None,
                             shading_steps: int | None = None, saturation: float | None = None, blur: float = 0.5,
                             subject: str | None = None, frame_prompts: list[str] | None = None,
                             retries: int = 2, min_iou: float | None = None, max_hist_distance: float | None = None,
                             fps: int | None = None, seed: int | None = None, name: str | None = None,
                             output_dir: str | None = None, idle_timeout_seconds: int | None = None,
                             timeout: float = 900, progress: Progress = None) -> dict:
    if not 2 <= frames <= 16:
        raise ValueError("frames must be between 2 and 16")
    if not 0 <= retries <= 5:
        raise ValueError("retries must be between 0 and 5")
    if not action or not action.strip():
        raise ValueError("give an action, for example idle, walk, chop, attack, hurt or die")
    check_pixel_options(size=size, max_colors=max_colors, blur=blur)
    st = resolve_style(style, outline, outline_color, shading_steps, saturation)
    pal = resolve_palette(settings, palette)
    poses, info = frame_poses(action, frames, frame_prompts)
    iou_min = info["iou_min"] if min_iou is None else float(min_iou)
    hist_max = info["hist_max"] if max_hist_distance is None else float(max_hist_distance)
    seed = random.randint(0, 2**31 - 1) if seed is None else int(seed)
    outdir = out_dir(settings, output_dir)
    t0 = time.time()
    raw_input = Path(sprite_or_prompt).expanduser()
    is_image = raw_input.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff") \
        or raw_input.is_file()
    common = dict(size=size, max_colors=max_colors, blur=blur, idle_timeout_seconds=idle_timeout_seconds,
                  timeout=timeout, progress=progress)
    if is_image:
        base = await _base_from_image(settings, image_input(sprite_or_prompt, "sprite_or_prompt"), outdir, st, pal,
                                      **common)
        what = subject or "the character"
        label = subject or raw_input.stem
    else:
        sp = await generate_sprite(settings, sprite_or_prompt, palette=palette, style=style, outline=outline,
                                   outline_color=outline_color, shading_steps=shading_steps, saturation=saturation,
                                   seed=seed, name=name, output_dir=str(outdir), **common)
        base = {"reference": sp["raw"], "source": sp["source"], "sprite": sp["sprite"], "palette": sp["palette"],
                "scale": sp["pixel"][0]["scale"], "generated": sp}
        what = subject or sp["prompt"]
        label = what
    stem = unique_stem(outdir, f"{slug(name or label)}_{slug(action, 2)}_{seed}")
    shared = base["palette"]  # every frame uses the base sprite's colours, so colours cannot drift
    pixel_style = dict(st)

    async def pixelize_frames(srcs: list[str], outs: list[Path]) -> list[dict]:
        items = [{"input": s, "output": str(o), "fit": "trim", "anchor": "bottom", "scale": base["scale"]}
                 for s, o in zip(srcs, outs)]
        res = await run_ops(settings, pixel_spec(size, shared, len(shared), pixel_style, blur, items), outs[-1])
        return res["outputs"]

    async def metrics(paths: list[str]) -> list[dict]:
        res = await run_ops(settings, {"op": "metrics", "base": base["sprite"], "frames": paths}, Path(paths[-1]))
        return res["frames"]

    # 1. Draw every frame in order: Klein edits the base reference, with the previous frame as image 2.
    raws: list[str] = []
    for i, pose in enumerate(poses):
        refs = [raws[-1]] if raws else []
        d = await draw(settings, [frame_prompt(what, action, pose, i, frames, bool(raws))], [seed + 1 + i], outdir,
                       idle_timeout_seconds, timeout, progress, refs=refs, input_image=base["reference"])
        raws.append(_move(d["paths"][0], outdir / f"{stem}_f{i:02d}_raw.png"))
    # 2. Cut out, pixelize with the shared palette and the base's scale, measure.
    sources = [_move(p, outdir / f"{stem}_f{i:02d}_source.png")
               for i, p in enumerate(await cut(settings, raws, outdir, idle_timeout_seconds, timeout, progress))]
    finals = [outdir / f"{stem}_f{i:02d}.png" for i in range(frames)]
    pix = await pixelize_frames(sources, finals)
    found = await metrics([str(f) for f in finals])
    frames_report = [{"frame": i, "seed": seed + 1 + i, "pose": poses[i], **_m(found[i]), "clipped": pix[i]["clipped"],
                      "attempts": 1, "passed": passes(found[i], iou_min, hist_max)} for i in range(frames)]
    # 3. Retry the frames that fall under the thresholds with new seeds; keep the best attempt.
    for i in range(frames):
        for attempt in range(retries):
            if frames_report[i]["passed"]:
                break
            s = seed + 1 + i + 1000 * (attempt + 1)
            refs = [raws[i - 1]] if i else []
            d = await draw(settings, [frame_prompt(what, action, poses[i], i, frames, bool(i))], [s], outdir,
                           idle_timeout_seconds, timeout, progress, refs=refs, input_image=base["reference"])
            [src] = await cut(settings, d["paths"], outdir, idle_timeout_seconds, timeout, progress)
            trial = outdir / f"{stem}_f{i:02d}_try{attempt + 1}.png"
            tp = (await pixelize_frames([src], [trial]))[0]
            [m] = await metrics([str(trial)])
            frames_report[i]["attempts"] += 1
            if score(m) > score(frames_report[i]):
                _move(d["paths"][0], Path(raws[i]))
                _move(src, Path(sources[i]))
                _move(str(trial), finals[i])
                frames_report[i].update({"seed": s, **_m(m), "clipped": tp["clipped"],
                                         "passed": passes(m, iou_min, hist_max)})
            else:
                for p in (d["paths"][0], src, str(trial)):
                    Path(p).unlink(missing_ok=True)
    # 4. Neighbouring frame metrics after the retries, the strip, a GIF preview and the JSON.
    final_m = await metrics([str(f) for f in finals])
    for r, m in zip(frames_report, final_m):
        r.update({"hist_distance_prev": m["hist_distance_prev"], "iou_prev": m["iou_prev"]})
    rate = int(fps or info["fps"])
    strip = outdir / f"{stem}_strip.png"
    packed = await run_ops(settings, {"op": "pack", "images": [{"name": f"{stem}_{i:02d}", "path": str(f)}
                                                              for i, f in enumerate(finals)],
                                      "cell": [size, size], "columns": frames, "power_of_two": False,
                                      "sheet": str(strip), "atlas": str(outdir / f"{stem}_strip_atlas.json"),
                                      "contact": str(outdir / f"{stem}_contact.png")}, strip)
    gif = outdir / f"{stem}_preview.gif"
    await run_ops(settings, {"op": "animate", "frames": [str(f) for f in finals], "output": str(gif), "fps": rate,
                             "zoom": max(1, 256 // size), "loop": info["loop"]}, gif)
    summary = _summary(frames_report)
    data = {"name": stem, "action": action, "frame_count": frames, "frame_size": [size, size], "fps": rate,
            "loop": info["loop"], "strip": strip.name, "frames": [f.name for f in finals],
            "palette": shared, "base_sprite": base["sprite"], "seed": seed,
            "thresholds": {"iou_min": iou_min, "hist_distance_max": hist_max}, "consistency": summary,
            "per_frame": frames_report}
    json_path = outdir / f"{stem}.json"
    json_path.write_text(json.dumps(data, indent=1))
    return {
        "strip": str(strip), "frames": [str(f) for f in finals], "json": str(json_path), "preview_gif": str(gif),
        "contact_sheet": packed["contact_sheet"], "atlas": packed["atlas"], "base_sprite": base["sprite"],
        "base_reference": base["reference"], "frame_sources": sources, "frame_raws": raws, "palette": shared,
        "action": action, "fps": rate, "loop": info["loop"], "seed": seed, "style": st,
        "thresholds": {"iou_min": iou_min, "hist_distance_max": hist_max}, "consistency": summary,
        "per_frame": frames_report, "output_dir": str(outdir),
        "paths": [str(strip), str(json_path), str(gif), *map(str, finals)],
        "base": base.get("generated"), "total_seconds": round(time.time() - t0, 2),
    }


def _m(m: dict) -> dict:
    return {"hist_distance": m["hist_distance"], "iou": m["iou"]}


def _summary(frames: list[dict]) -> dict:
    ious = [f["iou"] for f in frames]
    hists = [f["hist_distance"] for f in frames]
    return {"frames_passed": sum(f["passed"] for f in frames), "frames": len(frames),
            "iou_mean": round(sum(ious) / len(ious), 4), "iou_min": min(ious),
            "hist_distance_mean": round(sum(hists) / len(hists), 4), "hist_distance_max": max(hists),
            "retried_frames": sum(f["attempts"] > 1 for f in frames)}


async def _base_from_image(settings: Settings, src: Path, outdir: Path, st: dict, pal, *, size: int, max_colors: int,
                           blur: float, idle_timeout_seconds, timeout, progress) -> dict:
    """The base for an animation from an existing image: a flat reference at least DRAW_SIZE on its
    longer side for Klein (a small sprite is scaled up with nearest neighbour), its cut-out, and the
    base sprite pixelized from that cut-out, whose palette and scale every frame then uses."""
    stem = unique_stem(outdir, f"{src.stem}_base")
    ref = outdir / f"{stem}_reference.png"
    await run_ops(settings, {"op": "flatten", "input": str(src), "output": str(ref), "background": REF_BACKGROUND,
                             "min_size": DRAW_SIZE}, ref)
    [source] = await cut(settings, [str(ref)], outdir, idle_timeout_seconds, timeout, progress)
    source = _move(source, outdir / f"{stem}_source.png")
    final = outdir / f"{stem}.png"
    px = await run_ops(settings, pixel_spec(size, pal, max_colors, st, blur,
                                            [{"input": source, "output": str(final), "fit": "trim",
                                              "anchor": "bottom"}]), final)
    o = px["outputs"][0]
    return {"reference": str(ref), "source": source, "sprite": o["path"], "palette": px["palette"],
            "scale": o["scale"]}


# -- generate_tileset --------------------------------------------------------

# A preset is only a list of kinds; callers usually pass their own. Each kind: name, prompt (what to
# draw), seamless (a ground texture that wraps; generated with the seamless material workflow) and
# transparent (an object cut out on a transparent cell; else an opaque tile filling the cell).
TILE_PRESETS: dict[str, list[dict[str, Any]]] = {
    "strategy": [
        {"name": "grass", "prompt": "short green meadow grass with a few tiny flowers", "seamless": True},
        {"name": "forest", "prompt": "a dense cluster of three leafy trees seen from above at an angle"},
        {"name": "stump", "prompt": "a chopped tree stump with an axe cut top"},
        {"name": "rock", "prompt": "a large grey boulder with a few smaller stones"},
        {"name": "gravel", "prompt": "loose grey gravel and small pebbles", "seamless": True},
        {"name": "water_edge", "prompt": "a grassy bank along the top edge meeting calm blue water below",
         "transparent": False},
        {"name": "wood_wall", "prompt": "a short section of sharpened wooden palisade wall"},
        {"name": "stone_wall", "prompt": "a short section of thick grey stone block wall"},
        {"name": "gate_open", "prompt": "a wooden fortress gate standing open between two posts"},
        {"name": "gate_closed", "prompt": "a closed wooden fortress gate between two posts"},
        {"name": "tower", "prompt": "a small round stone watch tower with a wooden roof"},
        {"name": "home", "prompt": "a small cosy cottage with a thatched roof and a chimney"},
        {"name": "hearth", "prompt": "an open stone hearth with a crackling fire"},
        {"name": "farm_empty", "prompt": "a tilled farm field of brown soil furrows, nothing planted",
         "transparent": False},
        {"name": "farm_ripe", "prompt": "a farm field full of ripe golden wheat in rows", "transparent": False},
        {"name": "raider_camp", "prompt": "a ramshackle raider camp: a crude hide tent, a spiked fence and a campfire"},
    ],
}


def normalise_kinds(kinds: list | None, preset: str | None) -> list[dict]:
    """Tile kinds as dicts {name, prompt, seamless, transparent}; names made file safe and unique."""
    if kinds is None:
        key = preset or "strategy"
        if key not in TILE_PRESETS:
            raise ValueError(f"unknown tile preset {key!r}; presets: {sorted(TILE_PRESETS)} (or pass kinds)")
        kinds = TILE_PRESETS[key]
    if not kinds or len(kinds) > 64:
        raise ValueError("give 1 to 64 tile kinds")
    out, seen = [], set()
    for i, k in enumerate(kinds):
        if isinstance(k, str):
            k = {"name": k}
        if not isinstance(k, dict) or not str(k.get("name") or k.get("prompt") or "").strip():
            raise ValueError(f"tile kind {i} needs a name (a string, or a dict with name and optional prompt, "
                             "seamless and transparent)")
        label = str(k.get("name") or k.get("prompt"))
        name = re.sub(r"[^A-Za-z0-9_]+", "_", label.strip().lower()).strip("_")[:40] or f"tile{i}"
        if name in seen:
            raise ValueError(f"duplicate tile kind {name!r}")
        seen.add(name)
        seamless = bool(k.get("seamless", False))
        out.append({"name": name, "prompt": str(k.get("prompt") or label.replace("_", " ")),
                    "seamless": seamless, "transparent": bool(k.get("transparent", not seamless)) and not seamless})
    return out


async def generate_tileset(settings: Settings, prompt: str | None = None, *, kinds: list | None = None,
                           preset: str | None = None, size: int = 64, palette=None, max_colors: int = 32,
                           style: str | None = None, outline: bool | None = None, outline_color: str | None = None,
                           shading_steps: int | None = None, saturation: float | None = None, blur: float = 0.5,
                           columns: int | None = None, seed: int | None = None, name: str | None = None,
                           output_dir: str | None = None, idle_timeout_seconds: int | None = None,
                           timeout: float = 900, progress: Progress = None) -> dict:
    tiles = normalise_kinds(kinds, preset)
    check_pixel_options(size=size, max_colors=max_colors, blur=blur)
    st = resolve_style(style, outline, outline_color, shading_steps, saturation)
    pal = resolve_palette(settings, palette)
    seed = random.randint(0, 2**31 - 1) if seed is None else int(seed)
    outdir = out_dir(settings, output_dir)
    stem = unique_stem(outdir, f"{slug(name or prompt or 'tileset', 4)}_{seed}")
    t0 = time.time()
    theme = f", {prompt.strip().rstrip('.')}" if prompt and prompt.strip() else ""
    for i, t in enumerate(tiles):
        t["seed"] = seed + i
        t["raw"] = str(outdir / f"{stem}_{t['name']}_raw.png")
    # Ground tiles: the seamless material workflow (tiling base color only), all queued at once.
    ground = [t for t in tiles if t["seamless"]]
    if ground:
        wf = workflows.get("material")
        overrides = [{"prompt": f"{t['prompt']}{theme}, {st['prompt']}", "size": DRAW_SIZE, "seed": t["seed"],
                      "save_untiled": False} for t in ground]
        res = await jobs.run_workflow(settings, wf, overrides, str(outdir), idle_timeout_seconds,
                                      timeout=timeout, progress=progress)
        for t, o in zip(ground, res["outputs"]):
            _move(o["paths"][0], Path(t["raw"]))
            t["source"] = t["raw"]
    # Other tiles and objects: one batch of the image workflow, then one cut-out batch for objects.
    drawn = [t for t in tiles if not t["seamless"]]
    if drawn:
        texts = [sprite_prompt(t["prompt"] + theme, st) if t["transparent"] else tile_prompt(t["prompt"] + theme, st)
                 for t in drawn]
        res = await draw(settings, texts, [t["seed"] for t in drawn], outdir, idle_timeout_seconds, timeout, progress)
        for t, o in zip(drawn, res["outputs"]):
            _move(o["paths"][0], Path(t["raw"]))
            t["source"] = t["raw"]
    objects = [t for t in drawn if t["transparent"]]
    if objects:
        cuts = await cut(settings, [t["raw"] for t in objects], outdir, idle_timeout_seconds, timeout, progress)
        for t, c in zip(objects, cuts):
            t["source"] = _move(c, outdir / f"{stem}_{t['name']}_source.png")
    items = []
    for t in tiles:
        t["path"] = str(outdir / f"{stem}_{t['name']}.png")
        item = {"input": t["source"], "output": t["path"]}
        if t["transparent"]:
            item.update(fit="trim", anchor="bottom")
        else:
            item.update(fit="fill", outline=False, wrap=t["seamless"])
        items.append(item)
    palette_out = outdir / f"{stem}_palette.gpl" if pal is None else None
    px = await run_ops(settings, pixel_spec(size, pal, max_colors, st, blur, items, palette_out), Path(items[-1]["output"]))
    sheet = outdir / f"{stem}_sheet.png"
    packed = await run_ops(settings, {"op": "pack", "images": [{"name": t["name"], "path": t["path"]} for t in tiles],
                                      "cell": [size, size], "columns": columns or min(8, len(tiles)),
                                      "power_of_two": True, "sheet": str(sheet), "atlas": str(outdir / f"{stem}_atlas.json"),
                                      "contact": str(outdir / f"{stem}_contact.png")}, sheet)
    return {
        "tiles": [{k: t[k] for k in ("name", "prompt", "seamless", "transparent", "seed", "path", "source", "raw")}
                  | {"colours_used": o["colours_used"]} for t, o in zip(tiles, px["outputs"])],
        "sheet": str(sheet), "atlas": packed["atlas"], "contact_sheet": packed["contact_sheet"],
        "palette": px["palette"], "palette_file": px.get("palette_file") or (pal if isinstance(pal, str) else None),
        "style": st, "seed": seed, "size": size, "output_dir": str(outdir),
        "paths": [str(sheet), packed["atlas"], packed["contact_sheet"]] + [t["path"] for t in tiles],
        "skipped": ["auto tiling transition sets (47 tile blob) are not generated"],
        "total_seconds": round(time.time() - t0, 2),
    }


# -- pack_sheet and pixelize -------------------------------------------------


def pack_entries(images: list) -> list[dict]:
    if not images or len(images) > 1024:
        raise ValueError("give 1 to 1024 images")
    out, seen = [], set()
    for i, e in enumerate(images):
        if isinstance(e, str):
            e = {"path": e}
        if not isinstance(e, dict) or not e.get("path"):
            raise ValueError(f"image {i} must be a path or {{name, path}}")
        path = image_input(e["path"], f"image {i}")
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(e.get("name") or path.stem)).strip("_") or f"cell{i}"
        base, n = name, 1
        while name in seen:
            n += 1
            name = f"{base}_{n}"
        seen.add(name)
        out.append({"name": name, "path": str(path)})
    return out


async def pack_sheet(settings: Settings, images: list, *, cell: int | list[int] | None = None,
                     columns: int | None = None, name: str | None = None, padding: int = 0, align: str = "bottom",
                     power_of_two: bool = True, output_dir: str | None = None) -> dict:
    entries = pack_entries(images)
    if cell is not None:
        cw, ch = (cell, cell) if isinstance(cell, int) else (int(cell[0]), int(cell[1]))
        if not (1 <= cw <= 4096 and 1 <= ch <= 4096):
            raise ValueError("cell must be 1 to 4096 pixels")
        cell = [cw, ch]
    if columns is not None and not 1 <= columns <= 256:
        raise ValueError("columns must be between 1 and 256")
    if not 0 <= padding <= 64:
        raise ValueError("padding must be between 0 and 64")
    if align not in ("bottom", "center"):
        raise ValueError('align must be "bottom" or "center"')
    outdir = out_dir(settings, output_dir)
    stem = unique_stem(outdir, slug(name or "sheet", 6))
    sheet = outdir / f"{stem}.png"
    res = await run_ops(settings, {"op": "pack", "images": entries, "cell": cell, "columns": columns,
                                   "padding": padding, "align": align, "power_of_two": power_of_two,
                                   "name": stem, "sheet": str(sheet), "atlas": str(outdir / f"{stem}.json"),
                                   "contact": str(outdir / f"{stem}_contact.png")}, sheet)
    atlas = res["atlas_data"]
    return {"sheet": res["sheet"], "atlas": res["atlas"], "contact_sheet": res["contact_sheet"],
            "width": atlas["width"], "height": atlas["height"], "cell": [atlas["cell_width"], atlas["cell_height"]],
            "columns": atlas["columns"], "rows": atlas["rows"], "cells": atlas["cells"], "unity": atlas["unity"],
            "paths": [res["sheet"], res["atlas"], res["contact_sheet"]], "output_dir": str(outdir)}


async def pixelize(settings: Settings, image: str, *, size: int = 64, palette=None, max_colors: int = 32,
                   outline: bool = True, outline_color: str = "auto", fit: str = "trim", anchor: str = "center",
                   blur: float = 0.5, saturation: float = 1.0, shading_steps: int | None = None, wrap: bool = False,
                   remove_background: bool = False, name: str | None = None, output_dir: str | None = None,
                   idle_timeout_seconds: int | None = None, timeout: float = 900, progress: Progress = None) -> dict:
    src = image_input(image, "image")
    check_pixel_options(size=size, max_colors=max_colors, outline_color=outline_color, shading_steps=shading_steps,
                        saturation=saturation, blur=blur, fit=fit, anchor=anchor)
    pal = resolve_palette(settings, palette)
    outdir = out_dir(settings, output_dir)
    stem = unique_stem(outdir, f"{slug(name or src.stem, 6)}_px{size}")
    t0 = time.time()
    warnings, source = [], str(src)
    if remove_background:
        [c] = await cut(settings, [str(src)], outdir, idle_timeout_seconds, timeout, progress)
        source = _move(c, outdir / f"{stem}_source.png")
    elif fit != "fill" and not jobs.png_has_alpha(src):
        warnings.append("the image has no transparency, so the whole image is the subject; pass "
                        "remove_background=True to cut the subject out first")
    final = outdir / f"{stem}.png"
    st = {"outline": outline, "outline_color": outline_color, "shading_steps": shading_steps, "saturation": saturation}
    item = {"input": source, "output": str(final), "preview": str(outdir / f"{stem}_x8.png"), "fit": fit,
            "anchor": anchor, "wrap": wrap}
    palette_out = outdir / f"{stem}_palette.gpl" if pal is None else None
    px = await run_ops(settings, pixel_spec(size, pal, max_colors, st, blur, [item], palette_out), final)
    o = px["outputs"][0]
    if o["clipped"]:
        warnings.append("part of the subject fell outside the cell")
    return {"sprite": o["path"], "preview": o.get("preview"), "input": str(src),
            "source": source if remove_background else None, "size": [size, size], "palette": px["palette"],
            "palette_file": px.get("palette_file") or (pal if isinstance(pal, str) else None),
            "colours_used": o["colours_used"], "outline_color": o["outline_color"], "scale": o["scale"],
            "settings": {"fit": fit, "anchor": anchor, "blur": blur, "max_colors": max_colors, "outline": outline,
                         "outline_color": outline_color, "saturation": saturation, "shading_steps": shading_steps,
                         "wrap": wrap},
            "warnings": warnings, "paths": [p for p in (o["path"], o.get("preview"), px.get("palette_file")) if p],
            "output_dir": str(outdir), "total_seconds": round(time.time() - t0, 2)}
