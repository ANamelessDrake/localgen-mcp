"""Pixel art post processing: pixelize, palettes, outlines, consistency metrics and sprite sheets.

    python -I pixel_ops.py SPEC.json

Run as a child process with the MCP venv's interpreter (numpy and Pillow); the MCP server never
imports this module, so it stays free of imaging libraries. The spec's "op" picks the operation
(pixelize, metrics, pack, flatten); the last stdout line is "PIXEL_OPS_RESULT {json}".
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
        raise ValueError(f"not a colour like #rrggbb: {text!r}")
    return int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16)


def to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def read_palette(source) -> np.ndarray:
    """Palette colours (k x 3, uint8) from a list of hex strings, a GPL file, a hex list file
    (one #rrggbb per line) or an image (its distinct opaque colours, at most 256)."""
    if isinstance(source, (list, tuple)):
        cols = [hex_colour(c) for c in source]
    else:
        path = Path(source)
        head = path.read_bytes()[:16]
        if head.startswith(b"GIMP Palette"):
            cols = []
            for line in path.read_text(errors="replace").splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
                    cols.append(tuple(int(p) for p in parts[:3]))
        elif not head.startswith(IMAGE_MAGIC) and not (head[:4] == b"RIFF" and head[8:12] == b"WEBP"):
            # A text list of colours, one #rrggbb (or rrggbb) per line; ";" starts a comment line.
            cols = [hex_colour(s.split()[0]) for s in path.read_text().splitlines()
                    if s.strip() and not s.lstrip().startswith(";")]
        else:
            img = np.asarray(_open(path))
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
    """spec: base (path), frames [paths]; each frame against the base and the frame before it."""
    base = load_u8(spec["base"])
    out, prev = [], base
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
    """spec: frames [paths], output (.gif), fps, zoom, background (#rrggbb), loop: a looping GIF
    preview of an animation, each frame scaled up by zoom and flattened on the background (GIF has
    no partial transparency)."""
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
    frames[0].save(spec["output"], save_all=True, append_images=frames[1:], duration=duration,
                   loop=0 if spec.get("loop", True) else 1, disposal=1)
    return {"path": spec["output"], "frames": len(frames), "duration_ms": duration}


OPS = {"pixelize": op_pixelize, "metrics": op_metrics, "pack": op_pack, "flatten": op_flatten,
       "animate": op_animate}


def main(argv: list[str]) -> int:
    spec = json.loads(Path(argv[0]).read_text())
    result = OPS[spec["op"]](spec)
    print(MARKER + " " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
