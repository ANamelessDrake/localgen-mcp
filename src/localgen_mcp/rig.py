"""Soft rigs for organic creatures (blender/rig_creature.py, run headless by blender_run).

Request validation and naming live here; the geometry, bones and skin weights
are computed inside Blender, so the MCP server imports nothing heavy.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

from . import blender_run, model_files
from .config import Settings

PRESETS = ("auto", "chain", "spine", "wings")
WEIGHTS = ("auto", "distance")
AXES = ("x", "y", "z")
ROOT_ENDS = ("auto", "min", "max")
BONES_RANGE = (2, 64)
WING_CHAINS_RANGE = (1, 8)
WING_BONES_RANGE = (1, 16)


class RigError(ValueError):
    pass


def _point(value, what: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise RigError(f"{what} must be a list of three numbers")
    try:
        out = [float(c) for c in value]
    except (TypeError, ValueError):
        raise RigError(f"{what} must be a list of three numbers") from None
    if not all(math.isfinite(c) for c in out):
        raise RigError(f"{what} must be finite")
    return out


def _in_range(value, rng: tuple[int, int], what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not rng[0] <= value <= rng[1]:
        raise RigError(f"{what} must be an integer in {rng[0]}..{rng[1]}")
    return value


def build_spec(input_mesh: str, preset: str = "auto", bones: int | None = None, axis=None, root_end=None,
               wing_chains: int = 3, wing_bones: int = 4, weights: str = "auto", name: str | None = None) -> dict:
    """Validate a request and return the spec rig_creature.py reads (without output paths)."""
    try:
        src = model_files.input_glb(input_mesh)
    except model_files.ModelFileError as exc:
        raise RigError(str(exc)) from None
    if preset not in PRESETS:
        raise RigError(f"preset must be one of {PRESETS}")
    if weights not in WEIGHTS:
        raise RigError(f"weights must be one of {WEIGHTS}")
    if bones is not None:
        bones = _in_range(bones, BONES_RANGE, "bones")
    wing_chains = _in_range(wing_chains, WING_CHAINS_RANGE, "wing_chains")
    wing_bones = _in_range(wing_bones, WING_BONES_RANGE, "wing_bones")
    if isinstance(axis, str):
        if axis.lower() not in AXES:
            raise RigError(f"axis must be one of {AXES} or [x, y, z]")
        axis = axis.lower()
    elif axis is not None:
        axis = _point(axis, "axis")
        if math.hypot(*axis) < 1e-6:
            raise RigError("axis must not be zero")
    if root_end is None or root_end == "auto":
        root_end = None
    elif isinstance(root_end, str):
        if root_end not in ROOT_ENDS:
            raise RigError(f"root_end must be one of {ROOT_ENDS} or a point [x, y, z]")
    else:
        root_end = _point(root_end, "root_end")
    if name is not None:
        if preset not in ("chain", "auto"):
            raise RigError("name sets the bone prefix of a 'chain' only")
        name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")[:32]
        if not name:
            raise RigError("name must contain letters or digits")
    return {"input": str(src), "preset": preset, "bones": bones, "axis": axis, "root_end": root_end,
            "wing_chains": wing_chains, "wing_bones": wing_bones, "weights": weights, "name": name}


def output_paths(src: Path, preset: str, output_dir: str | None) -> tuple[Path, Path]:
    outdir = Path(output_dir).expanduser() if output_dir else src.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", src.stem) + f"_rig_{preset}"
    return outdir / f"{stem}.glb", outdir / f"{stem}.fbx"


def parse_report(stdout: str) -> dict:
    return blender_run.parse_report(stdout, "RIG_RESULT")


async def rig_creature(settings: Settings, input_mesh: str, preset: str = "auto", bones: int | None = None,
                       axis=None, root_end=None, wing_chains: int = 3, wing_bones: int = 4,
                       weights: str = "auto", name: str | None = None, output_dir: str | None = None) -> dict:
    spec = build_spec(input_mesh, preset, bones, axis, root_end, wing_chains, wing_bones, weights, name)
    glb, fbx = output_paths(Path(spec["input"]), preset, output_dir)
    spec.update(out_glb=str(glb), out_fbx=str(fbx),
                mesh_name=re.sub(r"[^A-Za-z0-9_]", "_", Path(spec["input"]).stem)[:48] or "creature")
    t0 = time.time()
    report = await blender_run.run(settings, "rig_creature.py", [json.dumps(spec)], "RIG_RESULT", glb)
    report["paths"] = [str(glb), str(fbx)]
    report["total_seconds"] = round(time.time() - t0, 2)
    return report
