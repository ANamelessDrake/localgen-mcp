"""Retopology of GLB meshes (blender/retopo.py, run headless by blender_run).

Quad remeshing uses Instant Meshes through this interpreter
(localgen_mcp.instant_remesh), so the MCP venv needs pynanoinstantmeshes;
without it Blender's Quadriflow is tried instead.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from . import blender_run, model_files
from .blender_run import BlenderError as RetopoError  # noqa: F401  (re-exported for callers and tests)
from .config import Settings

METHODS = ("quad", "decimate")
DETAIL_SOURCES = ("auto", "paint", "none")
FACE_RANGE = (200, 500_000)
TEXTURE_RANGE = (256, 8192)
SCRIPT = blender_run.SCRIPTS / "retopo.py"


def is_glb(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == b"glTF"


def validate(input_mesh: str, target_faces: int, method: str, texture_size: int,
             detail: str | None = None, detail_strength: float = 1.0) -> Path:
    path = Path(input_mesh).expanduser().resolve()
    if not path.is_file():
        raise RetopoError(f"mesh not found: {path}")
    if path.suffix.lower() != ".glb" or not is_glb(path):
        raise RetopoError(f"not a binary glTF (.glb) file: {path}")
    try:
        model_files.require_self_contained(path)
    except model_files.ModelFileError as exc:
        raise RetopoError(str(exc)) from None
    if method not in METHODS:
        raise RetopoError(f"method must be one of {METHODS}")
    lo, hi = FACE_RANGE
    if not lo <= int(target_faces) <= hi:
        raise RetopoError(f"target_faces must be in {lo}..{hi}")
    lo, hi = TEXTURE_RANGE
    if not lo <= int(texture_size) <= hi:
        raise RetopoError(f"texture_size must be in {lo}..{hi}")
    if detail is not None and detail not in DETAIL_SOURCES:
        raise RetopoError(f"detail must be one of {DETAIL_SOURCES}")
    if not 0.0 <= float(detail_strength) <= 1.0:
        raise RetopoError("detail_strength must be between 0 and 1")
    return path


def detail_for(method: str, detail: str | None) -> str:
    """Detail weighting applies to decimation only (the quad remeshers have one density everywhere)."""
    if method != "decimate":
        return "none"
    return detail or "auto"


def script_args(src: Path, dst: Path, target_faces: int, method: str, texture_size: int,
                detail: str | None = None, detail_strength: float = 1.0) -> list:
    return [src, dst, int(target_faces), method, int(texture_size), blender_run.PYTHON,
            detail_for(method, detail), float(detail_strength)]


def command(settings: Settings, src: Path, dst: Path, target_faces: int, method: str,
            texture_size: int, detail: str | None = None, detail_strength: float = 1.0) -> list[str]:
    return blender_run.command(settings, "retopo.py",
                               script_args(src, dst, target_faces, method, texture_size, detail, detail_strength))


def parse_report(stdout: str) -> dict:
    return blender_run.parse_report(stdout, "RETOPO_RESULT")


def output_name(src: Path, suffix: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", src.stem) + suffix


async def retopologize(settings: Settings, input_mesh: str, target_faces: int = 10000,
                       method: str = "quad", texture_size: int = 2048,
                       output_dir: str | None = None, detail: str | None = None,
                       detail_strength: float = 1.0) -> dict:
    src = validate(input_mesh, target_faces, method, texture_size, detail, detail_strength)
    outdir = Path(output_dir).expanduser() if output_dir else src.parent
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / output_name(src, f"_retopo_{method}_{int(target_faces)}.glb")
    t0 = time.time()
    report = await blender_run.run(settings, "retopo.py",
                                   script_args(src, dst, target_faces, method, texture_size, detail, detail_strength),
                                   "RETOPO_RESULT", dst)
    if detail not in (None, "none") and method != "decimate":
        report["detail_note"] = ("detail weighting applies to method='decimate' only; quad remeshing uses one "
                                 "density everywhere")
    report["path"] = str(dst)
    report["note"] = ("The GLB stores every quad as two triangles (glTF has no quads); the FBX keeps the "
                      "quads and embeds the textures.") if report.get("output_quads") else None
    report["total_seconds"] = round(time.time() - t0, 2)
    return report
