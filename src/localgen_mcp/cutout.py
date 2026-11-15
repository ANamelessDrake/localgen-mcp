"""remove_background: RGBA cut-outs of one or more images (BiRefNet in ComfyUI).

Pass 1 (workflows.remove_background.WORKFLOW) removes the background of every
image in one prompt. When trim, square, size or background is asked for, pass
2 (FINISH, no model) lays each cut-out out from integers computed here; the
subject's box comes from the small foreground mask pass 1 saved, read with the
standard library (the MCP server imports no imaging library).
"""

from __future__ import annotations

import re
import shutil
import struct
import time
import uuid
import zlib
from pathlib import Path

from . import jobs
from .config import Settings
from .workflows.remove_background import FINISH, MAX_IMAGES, SIZE_RANGE, SLOTS, WORKFLOW, color_int, layout

BOX_THRESHOLD = 12  # of 255: mask preview pixels above this count as subject when finding the box
_NAME = re.compile(r"_(\d{2})([cmf])_\d+_\.png$")


def png_channel(path: Path) -> tuple[int, int, list[bytes]]:
    """Width, height and the first channel's rows of an 8 bit, non-interlaced PNG.

    Only the first channel is unfiltered: PNG filters work byte by byte at a distance of one pixel,
    so each channel decodes on its own."""
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    pos, idat, ihdr = 8, [], None
    while pos + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
        pos += 12 + length
    if ihdr is None:
        raise ValueError(f"PNG without a header: {path}")
    width, height, depth, ctype, _, _, interlace = ihdr
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(ctype)
    if depth != 8 or channels is None or interlace:
        raise ValueError(f"unsupported PNG (bit depth {depth}, colour type {ctype}, interlace {interlace}): {path}")
    raw = zlib.decompress(b"".join(idat))
    stride = width * channels
    if len(raw) < (stride + 1) * height:
        raise ValueError(f"truncated PNG: {path}")
    rows, prev = [], bytes(width)
    for y in range(height):
        start = y * (stride + 1)
        ftype, line = raw[start], raw[start + 1:start + 1 + stride:channels]
        cur = bytearray(line)
        if ftype == 1:
            for x in range(1, width):
                cur[x] = (cur[x] + cur[x - 1]) & 255
        elif ftype == 2:
            cur = bytearray((a + b) & 255 for a, b in zip(line, prev))
        elif ftype == 3:
            for x in range(width):
                left = cur[x - 1] if x else 0
                cur[x] = (cur[x] + ((left + prev[x]) >> 1)) & 255
        elif ftype == 4:
            for x in range(width):
                a = cur[x - 1] if x else 0
                b = prev[x]
                c = prev[x - 1] if x else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                cur[x] = (cur[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        elif ftype != 0:
            raise ValueError(f"bad PNG filter {ftype}: {path}")
        prev = bytes(cur)
        rows.append(prev)
    return width, height, rows


def subject_box(mask_png: Path, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Bounding box (x0, y0, x1, y1, exclusive) of the subject in a width x height image, from the
    downscaled foreground mask; None if no pixel is above the threshold. Rounded outwards, so the
    downscaling never cuts the subject."""
    pw, ph, rows = png_channel(mask_png)
    xs, ys = [], []
    for y, row in enumerate(rows):
        hits = [x for x, v in enumerate(row) if v > BOX_THRESHOLD]
        if hits:
            ys.append(y)
            xs += (hits[0], hits[-1])
    if not ys:
        return None
    x0, x1 = min(xs) * width // pw, -((-(max(xs) + 1) * width) // pw)
    y0, y1 = min(ys) * height // ph, -((-(max(ys) + 1) * height) // ph)
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def validate(images: list[str], padding: float, size: int | None, background: str | None) -> list[Path]:
    if isinstance(images, str):
        images = [images]
    if not images or len(images) > MAX_IMAGES:
        raise ValueError(f"give 1 to {MAX_IMAGES} images")
    paths = []
    for raw in images:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"input image not found: {path}")
        if not jobs.is_image_file(path):
            raise ValueError(f"not a PNG, JPEG, WebP, GIF, BMP or TIFF image: {path}")
        paths.append(path)
    if not 0.0 <= padding <= 1.0:
        raise ValueError("padding must be between 0 and 1 (a fraction of the subject's longer side)")
    if size is not None and not SIZE_RANGE[0] <= size <= SIZE_RANGE[1]:
        raise ValueError(f"size must be between {SIZE_RANGE[0]} and {SIZE_RANGE[1]}")
    if background is not None:
        color_int(background)
    return paths


def _by_index(paths: list[str], kind: str) -> dict[int, str]:
    out = {}
    for p in paths:
        m = _NAME.search(p)
        if m and m.group(2) == kind:
            out[int(m.group(1))] = p
    return out


async def remove_background(settings: Settings, images: list[str], trim: bool = False, padding: float = 0.05,
                            square: bool = False, size: int | None = None, background: str | None = None,
                            output_dir: str | None = None, idle_timeout_seconds: int | None = None,
                            timeout: float = 900, progress=None) -> dict:
    paths = validate(images, padding, size, background)
    finish = trim or square or size is not None or background is not None
    outdir = Path(output_dir).expanduser() if output_dir else jobs.default_output_dir(settings, "cutout")
    work = settings.root / "tmp" / f"cutout-{uuid.uuid4().hex[:10]}"
    t0 = time.time()
    slots = {SLOTS[i]: str(p) for i, p in enumerate(paths)}
    try:
        cut = await jobs.run_workflow(settings, WORKFLOW, [{"trim": trim}], str(work if finish else outdir),
                                      idle_timeout_seconds, images=slots, timeout=timeout, progress=progress)
        cuts, masks = _by_index(cut["paths"], "c"), _by_index(cut["paths"], "m")
        if sorted(cuts) != list(range(len(paths))):
            raise jobs.JobError(f"expected {len(paths)} cut-outs, got {cut['paths']}")
        entries, warnings = [], []
        for i, src in enumerate(paths):
            w, h = jobs.image_size(Path(cuts[i]))
            box = None
            if trim:
                box = subject_box(Path(masks[i]), w, h)
                if box is None:
                    warnings.append(f"no subject found in {src}; kept the whole image")
            entries.append({"input": str(src), "width": w, "height": h, "subject_box": list(box) if box else None,
                            "layout": layout(w, h, box, trim, padding, square, size)})
        result = cut
        if finish:
            result = await jobs.run_workflow(
                settings, FINISH, [{"layout": {SLOTS[i]: e["layout"] for i, e in enumerate(entries)},
                                    "background": background}],
                str(outdir), idle_timeout_seconds, images={SLOTS[i]: cuts[i] for i in range(len(paths))},
                timeout=timeout, progress=progress)
        finals = _by_index(result["paths"], "f" if finish else "c")
        outputs = [{"input": e["input"], "path": finals[i], "size": [e["layout"]["out_w"], e["layout"]["out_h"]],
                    "subject_box": e["subject_box"]} for i, e in enumerate(entries)]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    keep = ("backend_url", "backend_managed", "idle_timeout_seconds")
    return {
        "workflow": WORKFLOW.name,
        "outputs": outputs,
        "paths": [o["path"] for o in outputs],
        "output_dir": str(outdir),
        "format": "PNG, RGB on the background colour" if background else "PNG, RGBA (transparent background)",
        "options": {"trim": trim, "padding": padding if trim else None, "square": square, "size": size,
                    "background": background},
        "warnings": warnings,
        "cut_seconds": cut["total_seconds"],
        "total_seconds": round(time.time() - t0, 2),
        **{k: result[k] for k in keep if k in result},
        "gpu_memory_used_peak_mb": cut["gpu_memory_used_peak_mb"],
    }
