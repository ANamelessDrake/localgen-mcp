"""Pixel art post processing: pixelize, palettes, outlines, consistency metrics and sprite sheets.

    python -I pixel_ops.py SPEC.json

Run as a child process with the MCP venv's interpreter (numpy and Pillow); the MCP server never
imports this module, so it stays free of imaging libraries. The spec's "op" picks the operation
(pixelize, metrics, pack, flatten, animate, pose_guides, import_sheet, grid); the last stdout line is
"PIXEL_OPS_RESULT {json}".
The file imports nothing from the package, so it runs by path with -I (no script folder or
current folder on sys.path). The tests import its functions directly.

pixelize, per image:
  1. place: the subject (alpha above ALPHA_CUT) is mapped onto a size x size cell by one scale
     (source pixels per cell pixel): "trim" fits the subject's box, "frame" the whole image (one
     scale for views that share a framing), "fill" stretches the whole image to the cell (tiles).
     A fixed scale can be given, so animation frames keep the base sprite's scale.
  2. downsample: a Gaussian blur (sigma = blur x scale / 2, blur 0.5 by default, premultiplied
     alpha, wrapping for seamless tiles) evaluated only at the cell pixel centres, which is a blur then nearest
     neighbour sampling; alpha is then cut to 0 or 1.
  3. colour: optional saturation change, then every opaque pixel goes to the nearest palette
     colour in OKLab. An "auto" palette is extracted from the cell (k-means in OKLab; with
     shading_steps, hue families times that many tones). max_colors keeps the most used entries.
  4. clean up single stray pixels and pinholes, then a 1 pixel outline (4-neighbour ring) round
     the silhouette in a palette colour: the darkest, a fixed colour, or per pixel "auto" (the
     palette colour nearest a darkened copy of the colour it borders). One palette slot is kept
     for the outline, so the colour count never exceeds max_colors.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MARKER = "PIXEL_OPS_RESULT"
ERROR_MARKER = "PIXEL_OPS_ERROR"  # "PIXEL_OPS_ERROR {json message}": a ValueError raised here, safe to show
ALPHA_CUT = 0.5
FITS = ("trim", "frame", "fill")
ANCHORS = ("center", "bottom")
OUTLINE_DARKEN = 0.45          # an auto outline aims at this share of the bordered colour's lightness
SPECKLE_STEP = 0.15             # lone pixels closer than this (OKLab) to the colour round them are noise
IMAGE_MAGIC = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"BM", b"II*\x00", b"MM\x00*")


# -- colour ------------------------------------------------------------------

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def oklab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0..255 or 0..1 floats, last axis 3) to OKLab."""
    c = np.asarray(rgb, dtype=np.float64)
    if c.size and c.max() > 1.0:
        c = c / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.cbrt(lin @ _M1.T) @ _M2.T


def hex_colour(text: str) -> tuple[int, int, int]:
    t = str(text).strip().lstrip("#")
    if len(t) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in t):
        raise ValueError("not a colour like #rrggbb")  # never echo the text: it may come from any file
    return int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16)


def to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def read_palette(source) -> np.ndarray:
    """Palette colours (k x 3, uint8) from a list of "#rrggbb" strings or a PNG (its distinct opaque
    colours, at most 256). Text palette files (GPL, hex lists) are parsed by the server
    (sprites.read_text_palette) and arrive here as lists. Errors never quote the source."""
    if isinstance(source, (list, tuple)):
        cols = []
        for i, c in enumerate(source):
            try:
                cols.append(hex_colour(c))
            except ValueError:
                raise ValueError(f"palette entry {i + 1} is not a colour like #rrggbb") from None
    else:
        path = Path(source)
        if not path.read_bytes()[:8] == IMAGE_MAGIC[0]:
            raise ValueError("a palette image must be a PNG")
        try:
            img = np.asarray(_open(path))
        except (OSError, SyntaxError):
            raise ValueError("the palette PNG could not be read") from None
        px = img.reshape(-1, 4)
        px = px[px[:, 3] >= 128][:, :3]
        cols = [tuple(c) for c in np.unique(px, axis=0)]
        if len(cols) > 256:
            raise ValueError(f"palette image has {len(cols)} colours; at most 256")
    seen, out = set(), []
    for c in cols:
        if tuple(c) not in seen:
            seen.add(tuple(c))
            out.append(c)
    if not 2 <= len(out) <= 256:
        raise ValueError(f"a palette needs 2 to 256 colours, got {len(out)}")
    return np.array(out, dtype=np.uint8)


def write_gpl(path: Path, colours: np.ndarray, name: str) -> None:
    lines = ["GIMP Palette", f"Name: {name}", "Columns: 8", "#"]
    lines += [f"{int(r):3d} {int(g):3d} {int(b):3d}\t{to_hex((r, g, b))}" for r, g, b in colours]
    Path(path).write_text("\n".join(lines) + "\n")


def nearest(lab_px: np.ndarray, lab_pal: np.ndarray) -> np.ndarray:
    """Index of the nearest palette colour (OKLab distance) for each pixel."""
    out = np.empty(len(lab_px), dtype=np.int64)
    for i in range(0, len(lab_px), 65536):
        chunk = lab_px[i:i + 65536]
        out[i:i + 65536] = ((chunk[:, None, :] - lab_pal[None, :, :]) ** 2).sum(-1).argmin(1)
    return out


def kmeans(points: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> np.ndarray:
    """Deterministic k-means (k-means++ start); returns k or fewer centres."""
    uniq = np.unique(points, axis=0)
    if len(uniq) <= k:
        return uniq
    rng = np.random.default_rng(seed)
    centres = [points[rng.integers(len(points))]]
    d2 = ((points - centres[0]) ** 2).sum(1)
    for _ in range(1, k):
        if d2.sum() <= 0:
            break
        centres.append(points[rng.choice(len(points), p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((points - centres[-1]) ** 2).sum(1))
    c = np.array(centres)
    for _ in range(iters):
        lab = nearest(points, c)
        new = np.array([points[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(len(c))])
        if np.allclose(new, c, atol=1e-6):
            break
        c = new
    return c


def auto_palette(rgb: np.ndarray, max_colors: int, shading_steps: int | None) -> np.ndarray:
    """Extract a palette (uint8, at most max_colors) from opaque pixels (N x 3, 0..255).

    Without shading_steps: k-means in OKLab. With it: hue families (k-means on the colour
    channels with lightness weighted down), then shading_steps tones per family (1-D k-means
    on lightness), so every colour gets the same number of shading steps."""
    if len(rgb) == 0:
        return np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    lab = oklab(rgb.astype(np.float64))
    if not shading_steps:
        centres = kmeans(lab, max_colors)
    else:
        families = max(1, max_colors // shading_steps)
        feat = lab * np.array([0.5, 1.0, 1.0])
        fam = nearest(feat, kmeans(feat, families))
        centres = []
        for j in np.unique(fam):
            members = lab[fam == j]
            tones = kmeans(members[:, :1], shading_steps)
            idx = nearest(members[:, :1], tones)
            centres += [members[idx == t].mean(0) for t in range(len(tones)) if np.any(idx == t)]
        centres = np.array(centres)
    # Back to sRGB through the pixels themselves: each centre becomes the mean colour of its members.
    idx = nearest(lab, centres)
    cols = [rgb[idx == j].mean(0) for j in range(len(centres)) if np.any(idx == j)]
    return np.unique(np.clip(np.rint(cols), 0, 255).astype(np.uint8), axis=0)


def adjust_saturation(rgb: np.ndarray, factor: float) -> np.ndarray:
    """Scale saturation (rgb N x 3 in 0..1) by factor around each pixel's own grey."""
    if abs(factor - 1.0) < 1e-6:
        return rgb
    grey = (rgb * np.array([0.299, 0.587, 0.114])).sum(-1, keepdims=True)
    return np.clip(grey + (rgb - grey) * factor, 0.0, 1.0)


# -- geometry ----------------------------------------------------------------


MAX_PIXELS = 4096 * 4096


def _open(path) -> Image.Image:
    """The first frame of an image as RGBA, refusing anything over MAX_PIXELS."""
    with Image.open(path) as im:
        if im.width * im.height > MAX_PIXELS:
            raise ValueError(f"{path} is {im.width}x{im.height}; at most {MAX_PIXELS} pixels")
        im.seek(0)
        return im.convert("RGBA")


def load_rgba(path) -> np.ndarray:
    """An image as float RGBA in 0..1 (H x W x 4)."""
    return np.asarray(_open(path), dtype=np.float64) / 255.0


def subject_box(alpha: np.ndarray, cut: float = ALPHA_CUT):
    ys, xs = np.nonzero(alpha >= cut)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _gauss(sigma: float) -> np.ndarray:
    if sigma < 0.3:
        return np.array([1.0])
    r = int(math.ceil(3 * sigma))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return w / w.sum()


def _gather(img: np.ndarray, centres: np.ndarray, axis: int, kernel: np.ndarray, mode: str) -> np.ndarray:
    """Blurred values of img along axis, evaluated only at integer positions centres."""
    n = img.shape[axis]
    r = len(kernel) // 2
    acc = 0.0
    for t, w in zip(range(-r, r + 1), kernel):
        pos = centres + t
        if mode == "wrap":
            taken = np.take(img, pos % n, axis=axis)
        else:
            taken = np.take(img, np.clip(pos, 0, n - 1), axis=axis)
            if mode == "zero":
                outside = (pos < 0) | (pos >= n)
                if outside.any():
                    shape = [1] * img.ndim
                    shape[axis] = len(pos)
                    taken = taken * (~outside).reshape(shape)
        acc = acc + w * taken
    return acc


def place(img: np.ndarray, size: int, fit: str, anchor: str, margin: int, scale: float | None = None) -> dict:
    """Where the cell samples the source: origin (x0, y0) and scale (sx, sy) in source pixels per
    cell pixel, so cell pixel (u, v) samples the source at (x0 + (u + 0.5) sx, y0 + (v + 0.5) sy)."""
    h, w = img.shape[:2]
    if fit == "fill":
        return {"x0": 0.0, "y0": 0.0, "sx": w / size, "sy": h / size, "box": [0, 0, w, h]}
    inner = size - 2 * margin
    if fit == "frame":
        box = (0, 0, w, h)
    else:
        box = subject_box(img[..., 3])
        if box is None:
            box = (0, 0, w, h)
    bx0, by0, bx1, by1 = box
    s = scale if scale else max(bx1 - bx0, by1 - by0) / inner
    cw, ch = (bx1 - bx0) / s, (by1 - by0) / s
    if anchor == "bottom" and fit == "trim":
        alpha = img[by0:by1, bx0:bx1, 3]
        mass = alpha.sum(0)
        cx = bx0 + (float((mass * (np.arange(len(mass)) + 0.5)).sum() / mass.sum()) if mass.sum() else (bx1 - bx0) / 2)
        x0 = cx - (size / 2) * s
        y0 = by1 - (size - margin) * s
    else:
        x0 = bx0 - (size - cw) / 2 * s
        y0 = by0 - (size - ch) / 2 * s
    return {"x0": x0, "y0": y0, "sx": s, "sy": s, "box": list(box)}


def downsample(img: np.ndarray, size: int, where: dict, blur: float, wrap: bool) -> np.ndarray:
    """size x size float RGBA: the source blurred (premultiplied) and sampled at the cell centres."""
    pre = img.copy()
    pre[..., :3] *= pre[..., 3:4]
    opaque = bool((img[..., 3] >= 0.999).all())
    mode = "wrap" if wrap else ("clamp" if opaque else "zero")
    cx = np.floor(where["x0"] + (np.arange(size) + 0.5) * where["sx"]).astype(np.int64)
    cy = np.floor(where["y0"] + (np.arange(size) + 0.5) * where["sy"]).astype(np.int64)
    rows = _gather(pre, cy, 0, _gauss(blur * where["sy"] / 2), mode)
    out = _gather(rows, cx, 1, _gauss(blur * where["sx"] / 2), mode)
    a = out[..., 3:4]
    rgb = np.where(a > 1e-6, out[..., :3] / np.maximum(a, 1e-6), 0.0)
    return np.concatenate([np.clip(rgb, 0, 1), np.clip(a, 0, 1)], axis=-1)


def _neighbours4(mask: np.ndarray) -> np.ndarray:
    """Count of 4-neighbours set, for a boolean mask (outside counts as unset)."""
    m = np.pad(mask.astype(np.int16), 1)
    return m[:-2, 1:-1] + m[2:, 1:-1] + m[1:-1, :-2] + m[1:-1, 2:]


def cleanup(mask: np.ndarray) -> np.ndarray:
    """Drop opaque pixels with no opaque 4-neighbour, fill transparent pixels with all four set."""
    n = _neighbours4(mask)
    return (mask & (n > 0)) | (~mask & (n == 4))


def despeckle(rgb: np.ndarray, mask: np.ndarray, max_step: float = SPECKLE_STEP) -> np.ndarray:
    """Recolour lone pixels: an opaque pixel that shares its colour with none of its 4-neighbours,
    while at least 3 of them share one colour, takes that colour, unless it differs from it by more
    than max_step in OKLab (a deliberate detail such as an eye or a highlight stays)."""
    h, w = mask.shape
    out = rgb.copy()
    lab = oklab(rgb.astype(np.float64))
    pad = np.pad(rgb.astype(np.int64), ((1, 1), (1, 1), (0, 0)), constant_values=-1)
    keys = pad[..., 0] * 65536 + pad[..., 1] * 256 + pad[..., 2]
    pm = np.pad(mask, 1)
    nb_keys = np.stack([keys[:-2, 1:-1], keys[2:, 1:-1], keys[1:-1, :-2], keys[1:-1, 2:]], -1)
    nb_mask = np.stack([pm[:-2, 1:-1], pm[2:, 1:-1], pm[1:-1, :-2], pm[1:-1, 2:]], -1)
    centre = keys[1:-1, 1:-1]
    lone = mask & ~((nb_keys == centre[..., None]) & nb_mask).any(-1)
    for y, x in zip(*np.nonzero(lone)):
        ks = nb_keys[y, x][nb_mask[y, x]]
        if len(ks) < 3:
            continue
        vals, counts = np.unique(ks, return_counts=True)
        if counts.max() < 3:
            continue
        k = int(vals[counts.argmax()])
        colour = np.array([k >> 16, (k >> 8) & 255, k & 255])
        if np.sqrt(((oklab(colour.astype(np.float64)) - lab[y, x]) ** 2).sum()) <= max_step:
            out[y, x] = colour
    return out


def outline_ring(mask: np.ndarray) -> np.ndarray:
    """Transparent pixels 4-adjacent to the silhouette."""
    return ~mask & (_neighbours4(mask) > 0)


# -- pixelize ----------------------------------------------------------------


def choose_palette(cell_rgb: np.ndarray, mask: np.ndarray, palette, max_colors: int, shading_steps: int | None,
                   outline: bool, outline_color: str) -> tuple[np.ndarray, np.ndarray | None]:
    """The colours the cell may use, and the outline colour reserved among them (or None)."""
    reserve = 1 if outline else 0
    px = (cell_rgb[mask] * 255.0)
    if palette is None:
        cols = auto_palette(px, max(1, max_colors - reserve), shading_steps)
        if not outline:
            return cols, None
        if outline_color in ("auto", "darkest"):
            lab = oklab(cols.astype(np.float64))
            dark = cols[lab[:, 0].argmin()].astype(np.float64) * 0.45  # darker than anything in the cell
            ol = np.clip(np.rint(dark), 0, 255).astype(np.uint8)
        else:
            ol = np.array(hex_colour(outline_color), dtype=np.uint8)
        return np.unique(np.vstack([cols, ol[None]]), axis=0), ol
    pal = np.asarray(palette, dtype=np.uint8)
    lab_pal = oklab(pal.astype(np.float64))
    ol = None
    if outline:
        ol = (np.array(hex_colour(outline_color), dtype=np.uint8) if outline_color not in ("auto", "darkest")
              else pal[lab_pal[:, 0].argmin()])
    if len(pal) + (0 if ol is None or any((pal == ol).all(1)) else 1) <= max_colors:
        cols = pal
    else:
        # Keep the entries the cell uses most, leaving one slot for the outline colour.
        idx = nearest(oklab(px), lab_pal) if len(px) else np.zeros(0, dtype=np.int64)
        counts = np.bincount(idx, minlength=len(pal))
        order = [i for i in np.argsort(-counts, kind="stable") if ol is None or not (pal[i] == ol).all()]
        cols = pal[order[:max_colors - reserve]]
    if ol is not None:
        cols = np.unique(np.vstack([cols, ol[None]]), axis=0)
    return cols, ol


def quantise(cell_rgb: np.ndarray, mask: np.ndarray, cols: np.ndarray, exclude: np.ndarray | None) -> np.ndarray:
    """uint8 RGB cell with every opaque pixel on its nearest palette colour (never the reserved
    outline colour unless it is the only one)."""
    use = cols
    if exclude is not None and len(cols) > 1:
        keep = ~(cols == exclude).all(1)
        if keep.any():
            use = cols[keep]
    out = np.zeros(cell_rgb.shape, dtype=np.uint8)
    if mask.any():
        out[mask] = use[nearest(oklab(cell_rgb[mask]), oklab(use.astype(np.float64)))]
    return out


def add_outline(rgb: np.ndarray, mask: np.ndarray, cols: np.ndarray, ol: np.ndarray, mode: str) -> tuple:
    """Paint the outline ring; returns the new rgb and mask."""
    ring = outline_ring(mask)
    if not ring.any():
        return rgb, mask
    rgb = rgb.copy()
    if mode != "auto":
        rgb[ring] = ol
        return rgb, mask | ring
    lab_cols = oklab(cols.astype(np.float64))
    lab_img = oklab(rgb.astype(np.float64))
    h, w = mask.shape
    for y, x in zip(*np.nonzero(ring)):
        nb = [(y + dy, x + dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
              if 0 <= y + dy < h and 0 <= x + dx < w and mask[y + dy, x + dx]]
        src = min((lab_img[p] for p in nb), key=lambda c: c[0])  # darker than the darkest colour it borders
        target = src * np.array([OUTLINE_DARKEN, 0.8, 0.8])
        darker = lab_cols[:, 0] < src[0] - 0.12
        if darker.any():
            cand = np.nonzero(darker)[0]
            rgb[y, x] = cols[cand[((lab_cols[cand] - target) ** 2).sum(1).argmin()]]
        else:
            rgb[y, x] = ol
    return rgb, mask | ring


def cell_colours(img: np.ndarray, size: int, *, outline: bool = True, fit: str = "trim", anchor: str = "center",
                 blur: float = 0.5, saturation: float = 1.0, wrap: bool = False, scale: float | None = None,
                 clean: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """Steps 1 and 2 and the saturation change: float RGB cell, opaque mask, placement."""
    if fit not in FITS:
        raise ValueError(f"fit must be one of {FITS}")
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {ANCHORS}")
    margin = 1 if outline and fit != "fill" else 0
    where = place(img, size, fit, anchor, margin, scale)
    cell = downsample(img, size, where, blur, wrap)
    mask = cell[..., 3] >= ALPHA_CUT
    if clean and fit != "fill":
        mask = cleanup(mask)
    rgb = cell[..., :3]
    flat = adjust_saturation(rgb.reshape(-1, 3), saturation).reshape(rgb.shape)
    return flat, mask, where


def joint_palette(cells: list[tuple[np.ndarray, np.ndarray]], max_colors: int, shading_steps: int | None,
                  outline: bool, outline_color: str) -> np.ndarray:
    """One auto palette for several cells (a tileset, the views of a turnaround), so they share colours."""
    px = [rgb[mask] for rgb, mask in cells if mask.any()]
    pixels = np.concatenate(px) if px else np.zeros((0, 3))
    cols, _ = choose_palette(pixels[:, None, :], np.ones((len(pixels), 1), dtype=bool), None, max_colors,
                             shading_steps, outline, outline_color)
    return cols


def pixelize_array(img: np.ndarray, size: int, *, palette=None, max_colors: int = 32, outline: bool = True,
                   outline_color: str = "auto", fit: str = "trim", anchor: str = "center", blur: float = 0.5,
                   saturation: float = 1.0, shading_steps: int | None = None, wrap: bool = False,
                   scale: float | None = None, clean: bool = True) -> tuple[np.ndarray, dict]:
    """The whole pixelize pass on one float RGBA image; returns uint8 RGBA (size x size) and a report."""
    flat, mask, where = cell_colours(img, size, outline=outline, fit=fit, anchor=anchor, blur=blur,
                                     saturation=saturation, wrap=wrap, scale=scale, clean=clean)
    cols, ol = choose_palette(flat, mask, palette, max_colors, shading_steps, outline, outline_color)
    out_rgb = quantise(flat, mask, cols, ol)
    if clean:
        out_rgb = despeckle(out_rgb, mask)
    if outline:
        out_rgb, mask = add_outline(out_rgb, mask, cols, ol, "auto" if outline_color == "auto" else "fixed")
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., :3] = np.where(mask[..., None], out_rgb, 0)
    out[..., 3] = mask * 255
    used = np.unique(out[mask][:, :3], axis=0) if mask.any() else np.zeros((0, 3), dtype=np.uint8)
    report = {"size": [size, size], "scale": round(float(where["sx"]), 4), "subject_box": where["box"],
              "colours_used": int(len(used)), "palette": [to_hex(c) for c in cols],
              "outline_color": to_hex(ol) if ol is not None else None,
              "opaque_pixels": int(mask.sum()), "clipped": _clipped(img, where, size)}
    return out, report


def _clipped(img: np.ndarray, where: dict, size: int) -> bool:
    """True when part of the subject falls outside the cell (a fixed scale that is too large)."""
    box = subject_box(img[..., 3])
    if box is None:
        return False
    x0, y0, x1, y1 = box
    return bool(x0 < where["x0"] - 0.5 or y0 < where["y0"] - 0.5
                or x1 > where["x0"] + size * where["sx"] + 0.5 or y1 > where["y0"] + size * where["sy"] + 0.5)


def save_png(arr: np.ndarray, path, scale: int = 1) -> str:
    im = Image.fromarray(arr, "RGBA" if arr.shape[-1] == 4 else "RGB")
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    return str(path)


def op_pixelize(spec: dict) -> dict:
    """spec: items [{input, output, preview?, scale?, fit?, anchor?, outline?, wrap?}], size, palette
    (hex list, file path or None for auto), max_colors, outline, outline_color, blur, saturation,
    shading_steps, fit, anchor, wrap, preview_scale, palette_out (write the palette used as GPL).

    With an auto palette and several items, the palette is extracted from all of them together,
    so every item uses the same colours."""
    palette = read_palette(spec["palette"]) if spec.get("palette") else None
    max_colors = int(spec.get("max_colors", 32))
    outline_color = spec.get("outline_color", "auto")
    items = spec["items"]

    def opts(item: dict) -> dict:
        return {"outline": bool(item.get("outline", spec.get("outline", True))),
                "fit": item.get("fit", spec.get("fit", "trim")), "anchor": item.get("anchor", spec.get("anchor", "center")),
                "blur": float(spec.get("blur", 0.5)), "saturation": float(spec.get("saturation", 1.0)),
                "wrap": bool(item.get("wrap", spec.get("wrap", False))), "scale": item.get("scale")}

    images = [load_rgba(item["input"]) for item in items]
    if palette is None and len(items) > 1:
        cells = [cell_colours(img, int(spec["size"]), **opts(item))[:2] for img, item in zip(images, items)]
        palette = joint_palette(cells, max_colors, spec.get("shading_steps"),
                                any(opts(item)["outline"] for item in items), outline_color)
        max_colors = max(max_colors, len(palette))
    outputs = []
    for img, item in zip(images, items):
        arr, rep = pixelize_array(img, int(spec["size"]), palette=palette, max_colors=max_colors,
                                  outline_color=outline_color, shading_steps=spec.get("shading_steps"), **opts(item))
        rep["input"] = item["input"]
        rep["path"] = save_png(arr, item["output"])
        if item.get("preview"):
            rep["preview"] = save_png(arr, item["preview"], int(spec.get("preview_scale", 8)))
        outputs.append(rep)
    result = {"outputs": outputs,
              "palette": outputs[0]["palette"] if outputs else []}
    if spec.get("palette_out") and outputs:
        cols = np.array([hex_colour(c) for c in result["palette"]], dtype=np.uint8)
        write_gpl(Path(spec["palette_out"]), cols, Path(spec["palette_out"]).stem)
        result["palette_file"] = spec["palette_out"]
    return result


# -- consistency metrics -----------------------------------------------------


def colour_histogram(arr: np.ndarray) -> dict:
    """Share of the opaque pixels per exact colour."""
    mask = arr[..., 3] >= 128
    if not mask.any():
        return {}
    cols, counts = np.unique(arr[mask][:, :3], axis=0, return_counts=True)
    total = counts.sum()
    return {to_hex(c): n / total for c, n in zip(cols, counts)}


def histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Total variation distance of two colour histograms: 0 same colours in the same shares, 1 no
    colour in common."""
    ha, hb = colour_histogram(a), colour_histogram(b)
    keys = set(ha) | set(hb)
    if not keys:
        return 0.0
    return round(0.5 * sum(abs(ha.get(k, 0.0) - hb.get(k, 0.0)) for k in keys), 4)


def silhouette_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of the opaque pixels of two same size cells."""
    ma, mb = a[..., 3] >= 128, b[..., 3] >= 128
    union = (ma | mb).sum()
    return round(float((ma & mb).sum() / union), 4) if union else 1.0


def load_u8(path) -> np.ndarray:
    return np.asarray(_open(path))


def op_metrics(spec: dict) -> dict:
    """spec: base (path), frames [paths], prev (optional path the first frame follows, default the
    base); each frame against the base and against the frame before it."""
    base = load_u8(spec["base"])
    out, prev = [], load_u8(spec["prev"]) if spec.get("prev") else base
    for path in spec["frames"]:
        f = load_u8(path)
        out.append({"path": path, "hist_distance": histogram_distance(base, f), "iou": silhouette_iou(base, f),
                    "hist_distance_prev": histogram_distance(prev, f), "iou_prev": silhouette_iou(prev, f)})
        prev = f
    return {"frames": out}


# -- sheets ------------------------------------------------------------------


def next_pow2(n: int) -> int:
    return 1 << max(0, int(n - 1).bit_length())


def sheet_layout(n: int, cell_w: int, cell_h: int, columns: int | None, padding: int, power_of_two: bool) -> dict:
    """Grid geometry: columns, rows, sheet size and each cell's top-left corner (x, y)."""
    if n < 1:
        raise ValueError("no images to pack")
    cols = columns or max(1, math.ceil(math.sqrt(n)))
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    w = cols * cell_w + (cols - 1) * padding
    h = rows * cell_h + (rows - 1) * padding
    if power_of_two:
        w, h = next_pow2(w), next_pow2(h)
    cells = [((i % cols) * (cell_w + padding), (i // cols) * (cell_h + padding)) for i in range(n)]
    return {"columns": cols, "rows": rows, "width": w, "height": h, "cells": cells}


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except (OSError, TypeError):
        return ImageFont.load_default()


def contact_sheet(images: list[np.ndarray], names: list[str], cell_w: int, cell_h: int, path, columns: int) -> str:
    """Review sheet: every image scaled up by a whole factor on a checkerboard, its name below."""
    zoom = max(1, min(8, 192 // max(cell_w, cell_h)))
    tw, th = cell_w * zoom, cell_h * zoom
    label_h, gap = 22, 8
    cols = min(columns, len(images))
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * (tw + gap) + gap, rows * (th + label_h + gap) + gap), (40, 40, 44))
    tile = 8
    yy, xx = np.mgrid[0:th, 0:tw]
    checker = np.where(((yy // tile + xx // tile) % 2)[..., None] == 0, 200, 160).astype(np.uint8).repeat(3, -1)
    draw = ImageDraw.Draw(sheet)
    font = _font(13)
    for i, (arr, name) in enumerate(zip(images, names)):
        x = gap + (i % cols) * (tw + gap)
        y = gap + (i // cols) * (th + label_h + gap)
        bg = Image.fromarray(checker, "RGB")
        im = Image.fromarray(arr, "RGBA").resize((arr.shape[1] * zoom, arr.shape[0] * zoom), Image.NEAREST)
        bg.paste(im, ((tw - im.width) // 2, th - im.height), im)
        sheet.paste(bg, (x, y))
        label = name if len(name) <= 28 else name[:27] + "~"
        draw.text((x + 2, y + th + 4), label, fill=(235, 235, 235), font=font)
    sheet.save(path)
    return str(path)


def op_pack(spec: dict) -> dict:
    """spec: images [{name, path}], cell [w, h] or None (largest image), columns, padding, align
    ("bottom" or "center"), power_of_two, sheet, atlas, contact (paths)."""
    entries = spec["images"]
    if not entries:
        raise ValueError("no images to pack")
    arrs = [load_u8(e["path"]) for e in entries]
    if spec.get("cell"):
        cw, ch = (int(v) for v in spec["cell"])
    else:
        cw, ch = max(a.shape[1] for a in arrs), max(a.shape[0] for a in arrs)
    for e, a in zip(entries, arrs):
        if a.shape[1] > cw or a.shape[0] > ch:
            raise ValueError(f"{e['name']} is {a.shape[1]}x{a.shape[0]}, larger than the {cw}x{ch} cell")
    pad = int(spec.get("padding", 0))
    lay = sheet_layout(len(arrs), cw, ch, spec.get("columns"), pad, bool(spec.get("power_of_two", True)))
    sheet = np.zeros((lay["height"], lay["width"], 4), dtype=np.uint8)
    cells = []
    for e, a, (x, y) in zip(entries, arrs, lay["cells"]):
        ox = (cw - a.shape[1]) // 2
        oy = ch - a.shape[0] if spec.get("align", "bottom") == "bottom" else (ch - a.shape[0]) // 2
        sheet[y + oy:y + oy + a.shape[0], x + ox:x + ox + a.shape[1]] = a
        cells.append({"name": e["name"], "x": x, "y": y, "w": cw, "h": ch,
                      "col": x // (cw + pad), "row": y // (ch + pad),
                      "unity_y": lay["height"] - y - ch, "source": e["path"]})
    save_png(sheet, spec["sheet"])
    atlas = {"name": spec.get("name", Path(spec["sheet"]).stem), "image": Path(spec["sheet"]).name,
             "width": lay["width"], "height": lay["height"], "cell_width": cw, "cell_height": ch,
             "columns": lay["columns"], "rows": lay["rows"], "padding": pad, "cells": cells,
             "unity": {"sprite_mode": "Multiple", "slice": "Grid By Cell Size", "pixel_size": [cw, ch],
                       "offset": [0, 0], "padding": [pad, pad], "filter_mode": "Point (no filter)",
                       "compression": "None", "pixels_per_unit": cw,
                       "note": "cells run left to right, top to bottom; unity_y is the cell's bottom edge "
                               "measured from the sheet's bottom, as Unity's sprite rects use"}}
    Path(spec["atlas"]).write_text(json.dumps(atlas, indent=1))
    contact = contact_sheet(arrs, [e["name"] for e in entries], cw, ch, spec["contact"],
                            int(spec.get("contact_columns") or lay["columns"]))
    return {"sheet": spec["sheet"], "atlas": spec["atlas"], "contact_sheet": contact, "atlas_data": atlas}


# -- references --------------------------------------------------------------


def op_flatten(spec: dict) -> dict:
    """spec: input, output, background (#rrggbb), min_size: composite RGBA onto a flat colour, and
    scale small images up with nearest neighbour to at least min_size on the longer side (so a
    finished 64 px sprite can serve as a diffusion reference)."""
    im = _open(spec["input"])
    zoom = max(1, math.ceil(int(spec.get("min_size", 0)) / max(im.size)))
    if zoom > 1:
        im = im.resize((im.width * zoom, im.height * zoom), Image.NEAREST)
    bg = Image.new("RGBA", im.size, hex_colour(spec.get("background", "#d8d8d8")) + (255,))
    bg.alpha_composite(im)
    bg.convert("RGB").save(spec["output"])
    return {"path": spec["output"], "size": list(bg.size), "zoom": zoom}


def op_animate(spec: dict) -> dict:
    """spec: frames [paths], output (.gif), fps or durations [ms per frame], zoom, background (#rrggbb), loop,
    cycles: a GIF preview of an animation, each frame scaled up by zoom and flattened on the
    background (GIF has no partial transparency). cycles (default 1) plays the frames that many
    times in the one file, which is easier to review in viewers that play a GIF once."""
    cycles = int(spec.get("cycles") or 1)
    if not 1 <= cycles <= 20:
        raise ValueError("gif_loops must be between 1 and 20")
    bg = hex_colour(spec.get("background", "#8a8a90")) + (255,)
    zoom = int(spec.get("zoom", 4))
    frames = []
    for path in spec["frames"]:
        im = _open(path)
        im = im.resize((im.width * zoom, im.height * zoom), Image.NEAREST)
        flat = Image.new("RGBA", im.size, bg)
        flat.alpha_composite(im)
        frames.append(flat.convert("RGB"))
    duration = max(20, round(1000 / float(spec.get("fps", 8))))
    if spec.get("durations"):  # per frame display times in ms (pose clips time their key frames)
        if len(spec["durations"]) != len(frames):
            raise ValueError("give one duration per frame")
        duration = [max(20, int(d)) for d in spec["durations"]]
    one_cycle = duration
    if cycles > 1:
        frames = frames * cycles
        if isinstance(duration, list):
            duration = duration * cycles
    frames[0].save(spec["output"], save_all=True, append_images=frames[1:], duration=duration,
                   loop=0 if spec.get("loop", True) else 1, disposal=1)
    return {"path": spec["output"], "frames": len(frames) // cycles, "cycles": cycles, "duration_ms": one_cycle}


# -- sheet import ------------------------------------------------------------
#
# A sprite sheet drawn by another image model, an artist or another tool: frames on a flat
# background, roughly in a grid but not on an exact one. The frames are found as row bands and
# then column runs of foreground pixels, read left to right and top to bottom, and each is moved
# onto one square canvas: its lowest pixel on a common ground line and the middle of its torso
# band on a common vertical, so pixelizing every canvas with fit="frame" keeps one scale and the
# motion between frames, without the jitter of each frame's own framing.

SHEET_DETECT = 60          # summed |RGB - background| above this counts as a frame when finding cells
SHEET_KEY = 45             # ... and above this is kept when the background is keyed out
SHEET_FRAGMENT = 0.4       # a run shorter than this share of the typical run is a piece of a neighbour
SHEET_SPECK = 0.02         # a run with under this share of the largest run's pixels is noise
SHEET_SPLIT = 1.6          # a column run this many times the typical width holds touching frames
TORSO_BAND = (0.35, 0.6)   # rows of a frame's own height whose median column is its anchor
SHADOW_ZONE = 0.1          # bottom share of a frame searched for a flat grey ground shadow
SHADOW_CHROMA = 14         # a shadow pixel is grey: max - min channel at most this
SHADOW_MAX = 240           # ... and differs from the background by less than this (outlines differ more)
CANVAS_PAD = 1.12          # canvas side over the larger of frame height and width round the anchor


def sheet_background(rgb: np.ndarray, border: int = 8) -> np.ndarray:
    """The flat background colour: the median of a strip round the sheet's edge."""
    b = max(1, min(border, rgb.shape[0] // 4, rgb.shape[1] // 4))
    edge = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                           rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
    return np.median(edge, axis=0)


def spans(profile: np.ndarray) -> list[list[int]]:
    """[start, end) runs where profile is True."""
    padded = np.concatenate([[False], profile.astype(bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [[int(a), int(b)] for a, b in zip(edges[::2], edges[1::2])]


def _merge(runs: list[list[int]], i: int) -> None:
    """Merge run i into the neighbour across the smaller gap."""
    left = runs[i][0] - runs[i - 1][1] if i > 0 else None
    right = runs[i + 1][0] - runs[i][1] if i + 1 < len(runs) else None
    j = i - 1 if right is None or (left is not None and left <= right) else i + 1
    a, b = sorted((i, j))
    runs[a] = [runs[a][0], runs[b][1]]
    del runs[b]


def clean_runs(mask: np.ndarray, axis: int, notes: list[str], what: str) -> list[list[int]]:
    """Runs of mask along axis (0: rows, 1: columns) with specks dropped and fragments (a detached
    hat, the far end of a tool) merged into the nearer neighbour."""
    counts = mask.sum(axis=1 - axis)
    runs = spans(counts >= 2)
    if not runs:
        return []
    mass = [int(counts[a:b].sum()) for a, b in runs]
    keep = [r for r, m in zip(runs, mass) if m >= SHEET_SPECK * max(mass)]
    if len(keep) < len(runs):
        notes.append(f"ignored {len(runs) - len(keep)} speck(s) when finding {what}")
    runs = keep
    while len(runs) > 1:
        typical = float(np.median([b - a for a, b in runs]))
        short = [i for i, (a, b) in enumerate(runs) if b - a < SHEET_FRAGMENT * typical]
        if not short:
            break
        _merge(runs, min(short, key=lambda i: runs[i][1] - runs[i][0]))
        notes.append(f"joined a narrow piece to a neighbouring {what[:-1]}")
    return runs


def split_wide(runs: list[list[int]], profile: np.ndarray, notes: list[str]) -> list[list[int]]:
    """Split column runs much wider than the typical one (frames that touch) at profile minima. A
    run with an empty column inside is a frame with a piece joined to it, and is never split."""
    if len(runs) < 2:
        return runs
    typical = float(np.median([b - a for a, b in runs]))
    out = []
    for a, b in runs:
        parts = int(round((b - a) / typical))
        if (b - a) < SHEET_SPLIT * typical or parts < 2 or (profile[a:b] < 2).any():
            out.append([a, b])
            continue
        cuts = []
        for k in range(1, parts):
            centre = a + (b - a) * k // parts
            lo, hi = max(a + 1, centre - int(typical * 0.3)), min(b - 1, centre + int(typical * 0.3))
            cuts.append(lo + int(np.argmin(profile[lo:hi])) if hi > lo else centre)
        edges = [a, *cuts, b]
        out.extend([[edges[k], edges[k + 1]] for k in range(parts)])
        notes.append(f"split a run {b - a} px wide into {parts} touching frames")
    return out


def find_cells(fg: np.ndarray, rows: int | None, notes: list[str]) -> list[list[tuple[int, int, int, int]]]:
    """Frame boxes (y0, y1, x0, x1) per row band, top to bottom and left to right."""
    bands = clean_runs(fg, 0, notes, "rows")
    if rows and len(bands) > rows:
        while len(bands) > rows:
            gaps = [bands[i + 1][0] - bands[i][1] for i in range(len(bands) - 1)]
            i = int(np.argmin(gaps))
            bands[i] = [bands[i][0], bands[i + 1][1]]
            del bands[i + 1]
        notes.append(f"joined row bands to make the {rows} rows asked for")
    grid = []
    for y0, y1 in bands:
        band = fg[y0:y1]
        cols = clean_runs(band, 1, notes, "frames")
        cols = split_wide(cols, band.sum(0), notes)
        grid.append([(y0, y1, x0, x1) for x0, x1 in cols])
    return grid


def subject_rows(alpha: np.ndarray) -> tuple[int, int] | None:
    ys = np.flatnonzero(alpha.any(1))
    return (int(ys[0]), int(ys[-1]) + 1) if len(ys) else None


def remove_ground_shadow(rgb: np.ndarray, diff: np.ndarray, alpha: np.ndarray, bg: np.ndarray) -> int:
    """Clear flat grey pixels a little darker than the background in the bottom of a frame (a drawn
    ground shadow or ground line). Returns the number of pixels cleared."""
    rows = subject_rows(alpha)
    if rows is None:
        return 0
    top, bottom = rows
    zone = np.zeros_like(alpha)
    zone[max(top, bottom - max(2, int(round(SHADOW_ZONE * (bottom - top))))):bottom] = True
    chroma = rgb.max(-1) - rgb.min(-1)
    shadow = zone & alpha & (chroma <= SHADOW_CHROMA) & (diff < SHADOW_MAX) & (rgb <= bg + 4).all(-1)
    alpha &= ~shadow
    return int(shadow.sum())


def torso_anchor(alpha: np.ndarray) -> float:
    """The median column of the frame's torso band (rows TORSO_BAND of its own height), which a
    swinging tool or a stride moves far less than the frame's box or its centre of mass."""
    top, bottom = subject_rows(alpha)
    h = bottom - top
    band = alpha[top + int(h * TORSO_BAND[0]):top + max(int(h * TORSO_BAND[1]), int(h * TORSO_BAND[0]) + 1)]
    xs = np.nonzero(band)[1]
    if len(xs) == 0:
        xs = np.nonzero(alpha)[1]
    return float(np.median(xs))


def sheet_overlay(rgb: np.ndarray, frames: list[dict], path) -> str:
    """The sheet with each frame's box and number in red, and, once aligned, its anchor column in
    blue and its ground line in green."""
    ov = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(ov)
    font = _font(max(14, rgb.shape[0] // 40))
    for i, f in enumerate(frames):
        y0, y1, x0, x1 = f["box"]
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(220, 40, 40), width=3)
        if "anchor" in f:
            ax = x0 + int(round(f["anchor"]))
            draw.line([ax, y0, ax, y1], fill=(40, 120, 220), width=2)
            draw.line([x0, y0 + f["bottom"], x1, y0 + f["bottom"]], fill=(40, 170, 60), width=2)
        draw.text((x0 + 4, y0 + 2), str(i), fill=(220, 40, 40), font=font)
    ov.save(path)
    return str(path)


def op_import_sheet(spec: dict) -> dict:
    """spec: input (sheet image), frames (expected count or None), rows (or None), columns (or None),
    background ("key": key out the flat background; "keep": cells keep it, for a BiRefNet cut-out
    after), key_threshold, shadow (clear a flat grey ground shadow), on_mismatch ("fail" or
    "keep"), output_prefix (cells are written as <prefix>_cNN.png), overlay (optional path: the
    sheet with the boxes found, for review).

    Frames are read left to right, then top to bottom. With frames given and a different count
    found, "fail" raises a ValueError that says what was found; "keep" carries on and reports it."""
    rgba = np.asarray(_open(spec["input"]), dtype=np.int32)
    if (rgba[..., 3] < 255).any():
        # A sheet with real transparency: its alpha is the mask; flatten on grey for the cells.
        bg = np.array([216.0, 216.0, 216.0])
        a = rgba[..., 3:4] / 255.0
        rgb = np.rint(rgba[..., :3] * a + bg * (1 - a)).astype(np.int32)
        diff = np.where(rgba[..., 3] >= 128, 765, 0)
    else:
        rgb = rgba[..., :3]
        bg = sheet_background(rgb)
        diff = np.abs(rgb - bg).sum(-1)
    notes: list[str] = []
    rows_req = int(spec["rows"]) if spec.get("rows") else None
    cols_req = int(spec["columns"]) if spec.get("columns") else None
    expected = int(spec["frames"]) if spec.get("frames") else (rows_req * cols_req if rows_req and cols_req else None)
    grid = find_cells(diff > SHEET_DETECT, rows_req, notes)
    per_row = [len(r) for r in grid]
    found = sum(per_row)
    problems = []
    if found == 0:
        raise ValueError("no frames found: the sheet looks like one flat colour")
    if expected is not None and found != expected:
        problems.append(f"found {found} frames but {expected} were expected")
    if rows_req and len(grid) != rows_req:
        problems.append(f"found {len(grid)} rows but {rows_req} were asked for")
    if cols_req and any(n > cols_req for n in per_row):
        problems.append(f"a row holds more than the {cols_req} columns asked for")
    if problems and spec.get("on_mismatch", "fail") == "fail":
        if spec.get("overlay"):
            sheet_overlay(rgb, [{"box": c} for row in grid for c in row], spec["overlay"])
        raise ValueError("; ".join(problems) + f" (frames per row found: {per_row}). Check the sheet and the "
                         "overlay, or pass the count found, or on_count_mismatch=\"keep\"")
    keyed = spec.get("background", "key") == "key"
    key = float(spec.get("key_threshold", SHEET_KEY))
    frames = []
    for y0, y1, x0, x1 in (cell for row in grid for cell in row):
        sub, d = rgb[y0:y1, x0:x1], diff[y0:y1, x0:x1]
        alpha = d > key
        cleared = remove_ground_shadow(sub, d, alpha, bg) if spec.get("shadow", True) else 0
        if subject_rows(alpha) is None:
            alpha = d > SHEET_DETECT
        top, bottom = subject_rows(alpha)
        xs = np.flatnonzero(alpha.any(0))
        frames.append({"box": (y0, y1, x0, x1), "alpha": alpha, "top": top, "bottom": bottom,
                       "left": int(xs[0]), "right": int(xs[-1]) + 1, "anchor": torso_anchor(alpha),
                       "shadow_pixels": cleared})
    height = max(f["bottom"] - f["top"] for f in frames)
    half = max(max(f["anchor"] - f["left"], f["right"] - f["anchor"]) for f in frames)
    side = int(math.ceil(max(height, 2 * half) * CANVAS_PAD))
    margin = max(2, int(round(0.03 * side)))
    ground, centre = side - margin, side // 2
    prefix = spec["output_prefix"]
    cells, report = [], []
    bg_u8 = np.clip(np.rint(bg), 0, 255).astype(np.uint8)
    for i, f in enumerate(frames):
        y0, y1, x0, x1 = f["box"]
        dx = centre - int(round(f["anchor"]))
        dy = ground - f["bottom"]
        canvas = np.zeros((side, side, 4), dtype=np.uint8)
        if not keyed:
            canvas[..., :3], canvas[..., 3] = bg_u8, 255
        h, w = y1 - y0, x1 - x0
        # Destination window of this cell on the canvas, clipped.
        ty0, tx0 = max(0, dy), max(0, dx)
        ty1, tx1 = min(side, h + dy), min(side, w + dx)
        if ty1 > ty0 and tx1 > tx0:
            src = (slice(ty0 - dy, ty1 - dy), slice(tx0 - dx, tx1 - dx))
            a = f["alpha"][src]
            win = canvas[ty0:ty1, tx0:tx1]
            px = rgb[y0:y1, x0:x1][src].astype(np.uint8)
            if keyed:
                win[a, :3], win[a, 3] = px[a], 255
            else:
                win[..., :3] = px
        path = f"{prefix}_c{i:02d}.png"
        save_png(canvas, path)
        cells.append(path)
        report.append({"frame": i, "row": next(r for r, row in enumerate(grid) if f["box"] in row),
                       "sheet_box": [x0, y0, x1, y1], "subject_height": f["bottom"] - f["top"],
                       "ground_shadow_pixels": f["shadow_pixels"]})
    if spec.get("overlay"):
        sheet_overlay(rgb, frames, spec["overlay"])
    return {"cells": cells, "frames_found": found, "frames_expected": expected, "rows_found": len(grid),
            "per_row": per_row, "count_ok": not problems, "problems": problems, "background": to_hex(bg_u8),
            "canvas": side, "ground_line": ground, "anchor_column": centre, "notes": notes, "frames": report,
            "overlay": spec.get("overlay")}


def op_grid(spec: dict) -> dict:
    """spec: images [paths], columns, canvas [w, h] (optional), background (#rrggbb), output: the
    images laid out left to right, top to bottom on one sheet. Without canvas every cell is the
    first image's size; with it, the canvas is split into equal cells and each image is scaled to
    fit its cell (aspect kept, centred), so the sheet matches the size another model draws at."""
    paths = spec["images"]
    if not paths:
        raise ValueError("no images to lay out")
    cols = max(1, min(int(spec.get("columns") or len(paths)), len(paths)))
    rows = math.ceil(len(paths) / cols)
    first = _open(paths[0])
    bg = hex_colour(spec.get("background", "#d8d8d8")) + (255,)
    if spec.get("canvas"):
        width, height = (int(v) for v in spec["canvas"])
        if not (64 <= width <= 4096 and 64 <= height <= 4096):
            raise ValueError("sheet_size must be 64 to 4096 pixels a side")
        cw, ch = width // cols, height // rows
    else:
        cw, ch = first.size
        width, height = cw * cols, ch * rows
    if width * height > MAX_PIXELS:
        raise ValueError(f"the sheet would be {width}x{height}; at most {MAX_PIXELS} pixels")
    sheet = Image.new("RGBA", (width, height), bg)
    boxes = []
    for i, p in enumerate(paths):
        im = _open(p)
        k = min(cw / im.width, ch / im.height)
        if abs(k - 1) > 1e-6:
            im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
        x = (i % cols) * cw + (cw - im.width) // 2
        y = (i // cols) * ch + (ch - im.height) // 2
        sheet.alpha_composite(im, (x, y))
        boxes.append([(i % cols) * cw, (i // cols) * ch, cw, ch])
    sheet.convert("RGB").save(spec["output"])
    return {"path": spec["output"], "columns": cols, "rows": rows, "size": [width, height], "cells": boxes}


# -- pose guides -------------------------------------------------------------
#
# A pose guide is a plain figure in one pose of a baked clip (pose_math.py), drawn at the size of
# the image it will be paired with, so a diffusion edit can copy the pose. pose_math does the
# retargeting and the camera; this part only draws.

_POSE_MATH = None
CLIP_MAX_BYTES = 64 << 20
GUIDE_STYLES = ("capsule", "skeleton")
WEAPONS = ("dagger", "sword", "axe", "warhammer", "spear", "shield")
GUIDE_BACKGROUND = {"capsule": (216, 216, 216), "skeleton": (0, 0, 0)}
SUPERSAMPLE = 2
# Capsule radius of the segment ending at each joint, as a share of stature (times body thickness).
SEGMENT_RADIUS = {"spine": 0.07, "chest": 0.08, "neck": 0.03, "head": 0.03, "shoulder_l": 0.03, "shoulder_r": 0.03,
                  "elbow_l": 0.033, "elbow_r": 0.033, "wrist_l": 0.027, "wrist_r": 0.027, "hand_l": 0.025,
                  "hand_r": 0.025, "hip_l": 0.048, "hip_r": 0.048, "knee_l": 0.048, "knee_r": 0.048,
                  "ankle_l": 0.036, "ankle_r": 0.036, "toe_l": 0.026, "toe_r": 0.026}
HEAD_RADIUS = 0.55            # head disc radius as a share of the head joint to crown length
GUIDE_COLOURS = {"torso": (180, 180, 188), "head": (206, 200, 194), "l": (118, 148, 206), "r": (206, 138, 108),
                 "face": (60, 60, 72), "blade": (232, 234, 240), "haft": (122, 96, 70), "shield": (156, 124, 84)}
DEPTH_SHADE = 0.6             # the farthest part is drawn at this share of its colour, the nearest at 1
# OpenPose body colours and limbs (COCO 18 order), so the skeleton style reads as an OpenPose map.
OPENPOSE_COLOURS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
                    (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
                    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]
OPENPOSE_LIMBS = [("neck_mid", "shoulder_r"), ("neck_mid", "shoulder_l"), ("shoulder_r", "elbow_r"),
                  ("elbow_r", "wrist_r"), ("shoulder_l", "elbow_l"), ("elbow_l", "wrist_l"), ("neck_mid", "hip_r"),
                  ("hip_r", "knee_r"), ("knee_r", "ankle_r"), ("neck_mid", "hip_l"), ("hip_l", "knee_l"),
                  ("knee_l", "ankle_l"), ("neck_mid", "face"), ("ankle_r", "toe_r"), ("ankle_l", "toe_l"),
                  ("wrist_r", "hand_r"), ("wrist_l", "hand_l")]
# Prop sizes in metres for a 1.75 m figure (scaled with the figure's stature).
PROP_REFERENCE_STATURE = 1.75


def pose_math():
    """pose_math.py, loaded by path from this file's folder (this helper imports nothing from the package)."""
    global _POSE_MATH
    if _POSE_MATH is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("pose_math", Path(__file__).with_name("pose_math.py"))
        _POSE_MATH = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_POSE_MATH)
    return _POSE_MATH


def load_clip(path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise ValueError("pose clip file not found")
    if p.stat().st_size > CLIP_MAX_BYTES:
        raise ValueError("pose clip file too large")
    try:
        clip = json.loads(p.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("pose clip file is not JSON") from None
    pose_math().check_clip(clip)
    return clip


def weapon_list(weapon) -> list[str]:
    if weapon in (None, "", "none"):
        return []
    items = [weapon] if isinstance(weapon, str) else list(weapon)
    for w in items:
        if w not in WEAPONS:
            raise ValueError(f"weapon must be one of {list(WEAPONS)} (or a list such as ['sword', 'shield'])")
    if len(items) > 2 or len(set(items)) != len(items) or (len(items) == 2 and "shield" not in items):
        raise ValueError("give one weapon, or one weapon and a shield")
    return items


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def prop_parts(kind: str, pos: np.ndarray, world: np.ndarray, rest_grip: dict, hand: str, size: float) -> list:
    """3D parts of one prop held in `hand` for one frame: ("capsule", [p0, p1], radius) or
    ("poly", [points]); lengths in metres times size. The grip runs along the hand's thumb side:
    the rest pose's forward direction carried by the hand segment's rotation."""
    pm = pose_math()
    J = pm.INDEX
    s = hand[0]
    h, w = pos[J[f"hand_{s}"]], pos[J[f"wrist_{s}"]]
    g = _unit(pm.qrot(world[J[f"hand_{s}"]], rest_grip[s]))
    k = _unit(h - w)                 # knuckle direction, across the grip
    grip = (h + w) / 2 + 0.35 * (h - w)
    if kind == "dagger":
        return [("capsule", [grip - 0.05 * size * g, grip + 0.3 * size * g], 0.016 * size),
                ("capsule", [grip - 0.05 * size * k, grip + 0.05 * size * k], 0.012 * size)]
    if kind == "sword":
        return [("capsule", [grip - 0.1 * size * g, grip + 0.85 * size * g], 0.02 * size),
                ("capsule", [grip + 0.06 * size * g - 0.1 * size * k, grip + 0.06 * size * g + 0.1 * size * k],
                 0.014 * size)]
    if kind == "axe":
        tip = grip + 0.62 * size * g
        head = [tip + 0.05 * size * g, tip + 0.06 * size * g + 0.17 * size * k,
                tip - 0.16 * size * g + 0.17 * size * k, tip - 0.1 * size * g]
        return [("capsule", [grip - 0.15 * size * g, tip + 0.05 * size * g], 0.017 * size), ("poly", head)]
    if kind == "warhammer":
        tip = grip + 0.62 * size * g
        return [("capsule", [grip - 0.12 * size * g, tip], 0.017 * size),
                ("capsule", [tip - 0.11 * size * k, tip + 0.11 * size * k], 0.05 * size)]
    if kind == "spear":
        tip = grip + 1.25 * size * g
        return [("capsule", [grip - 0.7 * size * g, tip], 0.014 * size),
                ("poly", [tip + 0.22 * size * g, tip + 0.045 * size * k, tip - 0.045 * size * k])]
    # shield: a disc across the forearm, facing away from the body
    e = pos[J[f"elbow_{s}"]]
    fore = _unit(w - e)
    n = _unit(np.cross(fore, g))
    if np.dot(n, w - pos[J["chest"]]) < 0:
        n = -n
    a = _unit(np.cross(n, fore))
    b = fore
    centre = (e + w) / 2 + 0.06 * size * n
    ang = np.linspace(0, 2 * np.pi, 28, endpoint=False)
    return [("poly", [centre + 0.28 * size * (math.cos(t) * a + math.sin(t) * b) for t in ang])]


def _shade(colour, near: float) -> tuple:
    f = DEPTH_SHADE + (1.0 - DEPTH_SHADE) * float(np.clip(near, 0.0, 1.0))
    return tuple(int(round(c * f)) for c in colour)


def guide_primitives(pos: np.ndarray, world: np.ndarray, *, style: str, stature: float, thickness: float,
                     weapons: list[str], weapon_hand: str, rest_grip: dict) -> list[dict]:
    """The figure for one frame as 3D primitives {kind, points, radius, colour, depth_points, order}."""
    pm = pose_math()
    J = pm.INDEX
    prims = []
    off_hand = "left" if weapon_hand == "right" else "right"
    size = stature / PROP_REFERENCE_STATURE
    for i, w in enumerate(weapons):
        hand = off_hand if (w == "shield" and len(weapons) == 2) else weapon_hand
        colour = GUIDE_COLOURS["shield" if w == "shield" else "blade"]
        for j, (kind, pts, *r) in enumerate(prop_parts(w, pos, world, rest_grip, hand, size)):
            c = GUIDE_COLOURS["haft"] if (kind == "capsule" and w in ("axe", "warhammer", "spear") and j == 0) else colour
            prims.append({"kind": kind, "points": np.array(pts), "radius": r[0] if r else 0.0, "colour": c,
                          "order": 50 + i})
    if style == "skeleton":
        pts = {n: pos[J[n]] for n in pm.NAMES}
        pts["neck_mid"] = (pos[J["shoulder_l"]] + pos[J["shoulder_r"]]) / 2
        for i, (a, b) in enumerate(OPENPOSE_LIMBS):
            prims.append({"kind": "line", "points": np.array([pts[a], pts[b]]), "radius": 0.012 * stature,
                          "colour": OPENPOSE_COLOURS[i % len(OPENPOSE_COLOURS)], "order": 10, "flat": True})
        dots = ["face", "neck_mid", "shoulder_r", "elbow_r", "wrist_r", "shoulder_l", "elbow_l", "wrist_l", "hip_r",
                "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l"]
        for i, n in enumerate(dots):
            prims.append({"kind": "disc", "points": np.array([pts[n]]), "radius": 0.016 * stature,
                          "colour": OPENPOSE_COLOURS[i], "order": 20, "flat": True})
        return prims
    for j, p in enumerate(pm.PARENTS):
        name = pm.NAMES[j]
        if p < 0 or name in pm.SYNTHETIC:
            continue
        side = name[-1] if name[-2:] in ("_l", "_r") else None
        colour = GUIDE_COLOURS[side] if side else GUIDE_COLOURS["torso"]
        prims.append({"kind": "capsule", "points": np.array([pos[p], pos[j]]),
                      "radius": SEGMENT_RADIUS[name] * stature * thickness, "colour": colour, "order": 0})
    quad = [pos[J[n]] for n in ("shoulder_l", "shoulder_r", "hip_r", "hip_l")]
    prims.append({"kind": "poly", "points": np.array(quad), "radius": 0.0, "colour": GUIDE_COLOURS["torso"],
                  "order": -1})
    head, top = pos[J["head"]], pos[J["head_top"]]
    hr = HEAD_RADIUS * float(np.linalg.norm(top - head))
    centre = (head + top) / 2
    prims.append({"kind": "disc", "points": np.array([centre]), "radius": hr, "colour": GUIDE_COLOURS["head"],
                  "order": 1})
    # The nose: a dot on the head towards the face, drawn over the head only when it faces the camera.
    face = centre + 0.85 * hr * _unit(pos[J["face"]] - centre)
    prims.append({"kind": "disc", "points": np.array([face]), "radius": 0.22 * hr, "colour": GUIDE_COLOURS["face"],
                  "order": 2})
    return prims


def draw_guide(prims: list[dict], cam: np.ndarray, scale: float, origin, size: tuple[int, int], background) -> Image.Image:
    """Project and draw primitives far to near (depth order), each shaded by its depth."""
    pm = pose_math()
    ss = SUPERSAMPLE
    im = Image.new("RGB", (size[0] * ss, size[1] * ss), tuple(background))
    draw = ImageDraw.Draw(im)
    placed = []
    for p in prims:
        proj = pm.project(p["points"], cam, scale * ss, (origin[0] * ss, origin[1] * ss))
        placed.append((float(proj[:, 2].mean()), p["order"], proj, p))
    depths = [d for d, *_ in placed]
    lo, hi = min(depths), max(depths)
    for depth, _, proj, p in sorted(placed, key=lambda t: (t[0], t[1])):
        near = (depth - lo) / (hi - lo) if hi - lo > 1e-9 else 1.0
        colour = tuple(p["colour"]) if p.get("flat") else _shade(p["colour"], near)
        r = p["radius"] * scale * ss
        xy = [(float(x), float(y)) for x, y in proj[:, :2]]
        if p["kind"] in ("capsule", "line"):
            draw.line(xy, fill=colour, width=max(1, int(round(2 * r))))
            if p["kind"] == "capsule":
                for x, y in xy:
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=colour)
        elif p["kind"] == "disc":
            x, y = xy[0]
            draw.ellipse([x - r, y - r, x + r, y + r], fill=colour)
        elif p["kind"] == "poly":
            draw.polygon(xy, fill=colour)
    return im.resize(size, Image.LANCZOS)


def _extent(prims_frames: list[list[dict]], cam: np.ndarray) -> tuple[float, float, float, float]:
    """Camera plane bounds (right min, right max, up min, up max) of every primitive, radius included."""
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for prims in prims_frames:
        for p in prims:
            q = p["points"] @ cam[:2].T
            lo = np.minimum(lo, q.min(0) - p["radius"])
            hi = np.maximum(hi, q.max(0) + p["radius"])
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def fit_shrink(extent, scale: float, origin, width: int, height: int, margin: float) -> float:
    """The factor (at most 1) to shrink the scale by, about the origin, so the camera plane extent
    (right min, right max, up min, up max) stays margin pixels inside the canvas."""
    rx0, rx1, uy0, uy1 = extent
    # pixel x = ox + r * scale, pixel y = oy - u * scale: how far each side reaches from the origin,
    # and the room there.
    sides = [(-rx0, origin[0] - margin), (rx1, width - margin - origin[0]),
             (uy1, origin[1] - margin), (-uy0, height - margin - origin[1])]
    shrink = 1.0
    for reach, room in sides:
        if reach > 0 and reach * scale > room:
            shrink = min(shrink, max(room, 1e-6) / (reach * scale))
    return shrink


def op_pose_guides(spec: dict) -> dict:
    """spec: clip (path), frames (n) or clip_frames [indices], selection ("auto", "even", "keys"),
    body (preset or dict), view, elevation, style ("capsule" or "skeleton"), weapon (name or [name,
    "shield"]), weapon_hand (default the clip's), exaggerate, size [w, h] or match_image (a cut-out
    whose size and subject box the figure takes: feet on the box bottom, centred, as tall as the
    box), outputs [paths], contact (optional path for a review sheet).

    One scale for every frame: the clip's first frame stands as tall as the box (or 80% of the
    canvas), feet on the floor line; if any selected frame would leave the canvas, the scale
    shrinks for all frames and "shrink" reports by how much."""
    pm = pose_math()
    clip = load_clip(spec["clip"])
    meta = clip.get("metadata") or {}
    style = spec.get("style", "capsule")
    if style not in GUIDE_STYLES:
        raise ValueError(f"guide style must be one of {list(GUIDE_STYLES)}")
    weapons = weapon_list(spec.get("weapon"))
    weapon_hand = spec.get("weapon_hand") or meta.get("weapon_hand") or "right"
    if weapon_hand not in ("left", "right"):
        raise ValueError('weapon_hand must be "left" or "right"')
    cam = pm.camera(spec.get("view", "front"), spec.get("elevation", 0.0))
    rt = pm.retarget(clip, spec.get("body"), float(spec.get("exaggerate", 1.0)))
    frame_count = int(clip["frame_count"])
    if spec.get("clip_frames") is not None:
        chosen = [int(f) for f in spec["clip_frames"]]
        if not chosen or any(not 0 <= f < frame_count for f in chosen):
            raise ValueError(f"clip_frames must be indices from 0 to {frame_count - 1}")
        impact = meta.get("impact_frame")
        sel = {"frames": chosen, "labels": [""] * len(chosen),
               "impact_index": chosen.index(impact) if impact in chosen else None}
    else:
        sel = pm.select_frames(meta, frame_count, int(spec["frames"]), spec.get("selection", "auto"))
    if spec.get("validate_only"):
        return {"clip": clip.get("name"), "clip_frames": sel["frames"], "impact_index": sel["impact_index"],
                "action": meta.get("action"), "loop": bool(meta.get("loop")), "weapons": weapons}
    outputs = spec["outputs"]
    if len(outputs) != len(sel["frames"]):
        raise ValueError("give one output path per frame")
    offsets = rt["offsets"]
    rest_grip = {}
    for s in "lr":
        axis = _unit(offsets[pm.INDEX[f"hand_{s}"]])
        fwd = np.array([0.0, -1.0, 0.0])
        rest_grip[s] = _unit(fwd - np.dot(fwd, axis) * axis)
    thickness = rt["body"]["thickness"]
    common = dict(style=style, stature=rt["stature"], thickness=thickness, weapons=weapons,
                  weapon_hand=weapon_hand, rest_grip=rest_grip)
    frames = [guide_primitives(rt["positions"][f], rt["world"][f], **common) for f in sel["frames"]]
    first = guide_primitives(rt["positions"][0], rt["world"][0], **{**common, "weapons": []})
    background = GUIDE_BACKGROUND[style]
    if spec.get("match_image"):
        ref = load_rgba(spec["match_image"])
        width, height = ref.shape[1], ref.shape[0]
        box = subject_box(ref[..., 3]) if ref.shape[2] == 4 else None
        if box is None or (box[2] - box[0]) * (box[3] - box[1]) >= 0.98 * width * height:
            box = (int(0.1 * width), int(0.1 * height), int(0.9 * width), int(0.9 * height))
    else:
        width, height = (int(v) for v in spec.get("size") or (1024, 1024))
        box = (int(0.1 * width), int(0.1 * height), int(0.9 * width), int(0.9 * height))
    if not (16 <= width <= 4096 and 16 <= height <= 4096):
        raise ValueError("guide size must be 16 to 4096 pixels a side")
    x0, y0, x1, y1 = box
    fx0, fx1, fy0, fy1 = _extent([first], cam)
    scale = (y1 - y0) / max(fy1 - fy0, 1e-6)
    # The first frame centred on the box, its lowest point on the box bottom.
    origin = [(x0 + x1) / 2 - (fx0 + fx1) / 2 * scale, y1 + fy0 * scale]
    shrink = fit_shrink(_extent(frames, cam), scale, origin, width, height, 0.02 * min(width, height))
    scale *= shrink
    paths = []
    for prims, out in zip(frames, outputs):
        draw_guide(prims, cam, scale, origin, (width, height), background).save(out)
        paths.append(str(out))
    fps = float(clip["fps"])
    usable = pm.cycle_length(meta, frame_count)
    picked = sel["frames"]
    steps = [(picked[i + 1] - picked[i]) for i in range(len(picked) - 1)]
    if meta.get("loop"):
        steps.append(usable - picked[-1] + picked[0])
    else:
        steps.append(round(sum(steps) / len(steps)) if steps else 1)
    durations = [max(20, int(round(1000 * max(s, 1) / fps))) for s in steps]
    if spec.get("contact"):
        thumbs = [np.asarray(Image.open(p).convert("RGBA").resize((256, int(256 * height / width)), Image.LANCZOS))
                  for p in paths]
        names = [f"f{i:02d} c{f}" + (" impact" if i == sel["impact_index"] else (f" {lab}" if lab else ""))
                 for i, (f, lab) in enumerate(zip(picked, sel["labels"]))]
        contact_sheet(thumbs, names, 256, thumbs[0].shape[0], spec["contact"], min(len(thumbs), 8))
    return {"paths": paths, "clip": clip.get("name"), "clip_frames": picked, "labels": sel["labels"],
            "impact_index": sel["impact_index"], "durations_ms": durations, "fps": fps,
            "loop": bool(meta.get("loop")), "action": meta.get("action"), "weapon_hand": weapon_hand,
            "weapons": weapons, "body": rt["body"], "scale": round(scale, 3), "shrink": round(shrink, 4),
            "size": [width, height], "box": list(box), "contact": spec.get("contact")}


OPS = {"pixelize": op_pixelize, "metrics": op_metrics, "pack": op_pack, "flatten": op_flatten,
       "animate": op_animate, "pose_guides": op_pose_guides, "import_sheet": op_import_sheet, "grid": op_grid}


def main(argv: list[str]) -> int:
    spec = json.loads(Path(argv[0]).read_text())
    try:
        result = OPS[spec["op"]](spec)
    except ValueError as e:
        # Messages raised in this module name the problem, never file contents; the server passes
        # only this line to the client. Anything else exits with a traceback the server only logs.
        print(ERROR_MARKER + " " + json.dumps(str(e)[:300]), flush=True)
        return 2
    print(MARKER + " " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
