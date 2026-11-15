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
    try:
        path = model_files.input_glb(input_mesh)
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


MAX_LODS = 8


def validate_lods(lods) -> list[int] | None:
    """lods: face budgets for LOD0, LOD1, ...; each in FACE_RANGE and strictly smaller than the one before."""
    if lods is None:
        return None
    if isinstance(lods, (str, bytes)) or not isinstance(lods, (list, tuple)) or not 1 <= len(lods) <= MAX_LODS:
        raise RetopoError(f"lods must be a list of 1 to {MAX_LODS} face counts, for example [10000, 3000, 800]")
    out = []
    lo, hi = FACE_RANGE
    for f in lods:
        if isinstance(f, bool) or not isinstance(f, (int, float)) or int(f) != f:
            raise RetopoError(f"lods entries must be whole face counts, got {f!r}")
        if not lo <= int(f) <= hi:
            raise RetopoError(f"lods entries must be in {lo}..{hi}")
        if out and int(f) >= out[-1]:
            raise RetopoError("lods must be strictly decreasing (LOD0 has the most faces)")
        out.append(int(f))
    return out


def lod_object_name(src: Path) -> str:
    """The base object name for <name>_LOD0, <name>_LOD1, ... (letters, digits and underscores)."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", src.stem).strip("_")[:48] or "mesh"


def detail_for(method: str, detail: str | None) -> str:
    """Detail weighting applies to decimation only (the quad remeshers have one density everywhere)."""
    if method != "decimate":
        return "none"
    return detail or "auto"


def script_args(src: Path, dst: Path, target_faces: int, method: str, texture_size: int,
                detail: str | None = None, detail_strength: float = 1.0, lods: list[int] | None = None) -> list:
    args = [src, dst, int(target_faces), method, int(texture_size), blender_run.PYTHON,
            detail_for(method, detail), float(detail_strength)]
    if lods:
        args += [",".join(str(int(f)) for f in lods), lod_object_name(src)]
    return args


def command(settings: Settings, src: Path, dst: Path, target_faces: int, method: str,
            texture_size: int, detail: str | None = None, detail_strength: float = 1.0,
            lods: list[int] | None = None) -> list[str]:
    return blender_run.command(settings, "retopo.py", script_args(src, dst, target_faces, method, texture_size,
                                                                  detail, detail_strength, lods))


def parse_report(stdout: str) -> dict:
    return blender_run.parse_report(stdout, "RETOPO_RESULT")


def output_name(src: Path, suffix: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", src.stem) + suffix


async def retopologize(settings: Settings, input_mesh: str, target_faces: int = 10000,
                       method: str = "quad", texture_size: int = 2048,
                       output_dir: str | None = None, detail: str | None = None,
                       detail_strength: float = 1.0, lods: list[int] | None = None) -> dict:
    lods = validate_lods(lods)
    if lods:
        target_faces = lods[0]  # LOD0 is the retopology itself
    src = validate(input_mesh, target_faces, method, texture_size, detail, detail_strength)
    outdir = Path(output_dir).expanduser() if output_dir else src.parent
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = "lods_" + "_".join(map(str, lods)) if lods else str(int(target_faces))
    dst = outdir / output_name(src, f"_retopo_{method}_{suffix}.glb")
    t0 = time.time()
    report = await blender_run.run(settings, "retopo.py",
                                   script_args(src, dst, target_faces, method, texture_size, detail, detail_strength,
                                               lods),
                                   "RETOPO_RESULT", dst)
    if detail not in (None, "none") and method != "decimate":
        report["detail_note"] = ("detail weighting applies to method='decimate' only; quad remeshing uses one "
                                 "density everywhere")
    if lods:
        report["lod_note"] = ("One GLB and one FBX hold <name>_LOD0.. (Unity builds a LOD Group from the suffix). "
                              "All LODs share LOD0's UVs and one material; LOD1 and lower are triangle "
                              "decimations of LOD0 and their budgets count triangles.")
    report["path"] = str(dst)
    report["note"] = ("The GLB stores every quad as two triangles (glTF has no quads); the FBX keeps the "
                      "quads and embeds the textures.") if report.get("output_quads") else None
    report["total_seconds"] = round(time.time() - t0, 2)
    return report
