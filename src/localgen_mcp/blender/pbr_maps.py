"""PBR map maths for tileable materials, numpy only (no bpy).

Imported by material_maps.py inside Blender's Python and by the tests; the MCP
server never imports it. Arrays are float in 0..1, rows top to bottom (image
order). Every filter is periodic (FFT blur, np.roll derivatives), so a map made
from a tileable image tiles too.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722])


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Relative luminance of sRGB encoded values, returned sRGB encoded again (perceptual 0..1)."""
    y = srgb_to_linear(rgb[..., :3]) @ LUMA
    return np.where(y <= 0.0031308, y * 12.92, 1.055 * np.power(y, 1 / 2.4) - 0.055)


def blur_wrap(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with wrap around edges (multiplication in the Fourier domain)."""
    if sigma <= 0:
        return a.copy()
    h, w = a.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    kernel = np.exp(-2 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    return np.fft.irfft2(np.fft.rfft2(a) * kernel, s=(h, w))


def normalise(a: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    if hi - lo < 1e-8:
        return np.full_like(a, 0.5)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def height_from_albedo(rgb: np.ndarray, invert: bool = False, detail: float = 1.0) -> np.ndarray:
    """Height (0 low, 1 high) from the base color: luminance, high pass, then a light blur.

    Brighter reads as higher (lit ridges, dark crevices), invert for materials where it is the
    other way round. The high pass (removing a blur of size/8) drops large lighting and colour
    gradients that are not relief; detail scales the small final blur down (more fine detail).
    """
    size = max(rgb.shape[:2])
    lum = luminance(rgb)
    high = lum - blur_wrap(lum, size / 8.0)
    high = blur_wrap(high, max(0.0, 1.0 * (size / 1024.0) / max(detail, 0.1)))
    h = normalise(high)
    return 1.0 - h if invert else h


def gradients_wrap(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central differences with wrap around: d/dx (columns, left to right) and d/drow (top to bottom)."""
    dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5
    drow = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5
    return dx, drow


def normal_from_height(h: np.ndarray, strength: float = 1.0, directx: bool = False) -> np.ndarray:
    """Tangent space normal map (RGB 0..1) from height, derivatives taken with wrap around.

    OpenGL convention by default (green is +Y, the surface facing the top of the image; Unity
    and Blender). directx=True flips green (-Y; Unreal, DirectX). strength 1 treats the full
    height range as a depth of size/128 pixels, so the look does not depend on the resolution.
    """
    size = max(h.shape)
    depth = strength * size / 128.0
    dx, drow = gradients_wrap(h)
    # Height increases along +x by dx; up the image (+Y) is decreasing row, so dh/dY = -drow.
    nx, ny = -depth * dx, depth * drow
    nz = np.ones_like(h)
    n = np.stack([nx, ny, nz], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    if directx:
        n[..., 1] *= -1
    return n * 0.5 + 0.5


def cavity(h: np.ndarray, size: int | None = None) -> np.ndarray:
    """How far each texel sits below its surroundings, over three scales (0 none, 1 deepest)."""
    scale = (size or max(h.shape)) / 1024.0
    c = np.zeros_like(h)
    for sigma, weight in ((2.0, 0.5), (8.0, 0.3), (24.0, 0.2)):
        c += weight * np.clip(blur_wrap(h, sigma * scale) - h, 0.0, None)
    top = np.percentile(c, 99.5)
    return np.clip(c / top, 0.0, 1.0) if top > 1e-8 else c


def ambient_occlusion(h: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """AO (1 open, darker in cavities of the height), 1 - strength at the deepest."""
    return 1.0 - strength * cavity(h)


def roughness_map(h: np.ndarray, rgb: np.ndarray, base: float = 0.7, variation: float = 0.25) -> np.ndarray:
    """Heuristic roughness: base, rougher in cavities and darker spots (dust, grime, corrosion),
    smoother on raised and bright areas (worn and polished)."""
    lum = luminance(rgb)
    lum_detail = normalise(lum - blur_wrap(lum, max(rgb.shape[:2]) / 8.0))
    r = base + variation * (0.6 * (0.5 - h) + 0.4 * (0.5 - lum_detail)) * 2.0
    return np.clip(r, 0.02, 1.0)


def seam_metrics(a: np.ndarray) -> dict:
    """How well an image wraps: the difference across each wrap edge against the differences
    between neighbouring pixels inside it.

    For each axis: wrap is the mean absolute difference between the last and the first column (or
    row), interior_mean the same for every neighbouring pair inside, ratio = wrap / interior_mean,
    and wrap_percentile the share of interior pairs whose difference is smaller than the wrap's.
    ok means the wrap is no larger than the largest interior pair: the edge is no stronger than
    a transition the image already has. (A stricter test, such as the 99th percentile, flags
    textures with real straight features, for example panel seams on a grid, which models like
    to place exactly at the centre lines a half offset moves to the edges; wrap_percentile
    shows how the wrap ranks.) centre_* do the same for the middle column and row, where a half
    offset puts the original edges.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 2:
        a = a[..., None]
    out = {}
    for axis, name in ((1, "x"), (0, "y")):
        b = np.moveaxis(a, axis, 0)  # pairs along the first axis
        n = b.shape[0]
        diffs = np.abs(b[1:] - b[:-1]).reshape(n - 1, -1).mean(axis=1)  # one value per interior pair
        wrap = float(np.abs(b[0] - b[-1]).mean())
        mid = n // 2
        centre = float(diffs[mid - 1])  # pair (mid - 1, mid)
        inner = np.delete(diffs, mid - 1)
        mean, p99, top = float(inner.mean()), float(np.percentile(inner, 99)), float(inner.max())
        out[name] = {
            "wrap": round(wrap, 5), "interior_mean": round(mean, 5), "interior_p99": round(p99, 5),
            "interior_max": round(top, 5),
            "ratio": round(wrap / mean, 3) if mean > 0 else None,
            "wrap_percentile": round(float((inner < wrap).mean() * 100), 1),
            "ok": wrap <= top,
            "centre": round(centre, 5), "centre_ratio": round(centre / mean, 3) if mean > 0 else None,
            "centre_ok": centre <= top,
        }
    out["ok"] = out["x"]["ok"] and out["y"]["ok"]
    out["centre_ok"] = out["x"]["centre_ok"] and out["y"]["centre_ok"]
    return out


def tile(a: np.ndarray, n: int = 3, max_size: int = 2048) -> np.ndarray:
    """n x n copies side by side, subsampled so the result is at most max_size wide."""
    t = np.tile(a, (n, n) + (1,) * (a.ndim - 2))
    step = int(np.ceil(t.shape[1] / max_size))
    return t[::step, ::step]


def to_uint8(a: np.ndarray) -> np.ndarray:
    return np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)


def to_uint16(a: np.ndarray) -> np.ndarray:
    return np.clip(np.round(a * 65535.0), 0, 65535).astype(np.uint16)


def png_bytes(a: np.ndarray) -> bytes:
    """Encode a uint8 or uint16 array (H, W) grey or (H, W, 3) RGB as PNG, rows top to bottom.

    Written here rather than through Blender's image saving so the bytes are exactly these
    values, whatever colour management Blender runs with.
    """
    if a.dtype not in (np.uint8, np.uint16):
        raise ValueError("png_bytes wants uint8 or uint16")
    if a.ndim == 2:
        a = a[..., None]
    h, w, c = a.shape
    colour_type = {1: 0, 3: 2, 4: 6}[c]
    depth = 16 if a.dtype == np.uint16 else 8
    rows = a.astype(">u2" if depth == 16 else np.uint8).reshape(h, -1).view(np.uint8)
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rows], axis=1).tobytes()  # filter 0 per row

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, colour_type, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def write_png(path: str, a: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(png_bytes(a))
