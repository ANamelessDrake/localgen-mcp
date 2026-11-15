"""Retopology of GLB meshes, run in a headless Blender process.

The work happens in blender/retopo.py inside Blender; this module validates
the request, runs Blender as a child process (CPU only, so it never competes
with ComfyUI for VRAM) and returns its report. Quad remeshing uses Instant
Meshes through this interpreter (localgen_mcp.instant_remesh), so the MCP venv
needs pynanoinstantmeshes; without it Blender's Quadriflow is tried instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

from .config import Settings

logger = logging.getLogger("localgen_mcp.retopo")

SCRIPT = Path(__file__).parent / "blender" / "retopo.py"
METHODS = ("quad", "decimate")
FACE_RANGE = (200, 500_000)
TEXTURE_RANGE = (256, 8192)


class RetopoError(RuntimeError):
    pass


def find_blender(settings: Settings) -> str:
    exe = settings.blender or shutil.which("blender")
    if not exe or not os.access(exe, os.X_OK):
        raise RetopoError("Blender not found; install it or set LOCALGEN_BLENDER to its executable")
    return exe


def is_glb(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == b"glTF"


def validate(input_mesh: str, target_faces: int, method: str, texture_size: int) -> Path:
    path = Path(input_mesh).expanduser().resolve()
    if not path.is_file():
        raise RetopoError(f"mesh not found: {path}")
    if not is_glb(path):
        raise RetopoError(f"not a binary glTF (.glb) file: {path}")
    if method not in METHODS:
        raise RetopoError(f"method must be one of {METHODS}")
    lo, hi = FACE_RANGE
    if not lo <= int(target_faces) <= hi:
        raise RetopoError(f"target_faces must be in {lo}..{hi}")
    lo, hi = TEXTURE_RANGE
    if not lo <= int(texture_size) <= hi:
        raise RetopoError(f"texture_size must be in {lo}..{hi}")
    return path


def command(settings: Settings, src: Path, dst: Path, target_faces: int, method: str,
            texture_size: int) -> list[str]:
    return [find_blender(settings), "-b", "--factory-startup", "-t", str(settings.blender_threads),
            "-P", str(SCRIPT), "--", str(src), str(dst), str(int(target_faces)), method,
            str(int(texture_size)), sys.executable]


def parse_report(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RETOPO_RESULT "):
            return json.loads(line[len("RETOPO_RESULT "):])
    raise RetopoError("Blender finished without a report")


async def retopologize(settings: Settings, input_mesh: str, target_faces: int = 10000,
                       method: str = "quad", texture_size: int = 2048,
                       output_dir: str | None = None) -> dict:
    src = validate(input_mesh, target_faces, method, texture_size)
    outdir = Path(output_dir).expanduser() if output_dir else src.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", src.stem)
    dst = outdir / f"{stem}_retopo_{method}_{int(target_faces)}.glb"
    tmp = settings.root / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TMPDIR": str(tmp)}
    cmd = command(settings, src, dst, target_faces, method, texture_size)
    logger.info("retopology: %s", " ".join(cmd))
    t0 = time.time()
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE, env=env)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=settings.retopo_timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RetopoError(f"retopology timed out after {settings.retopo_timeout} s")
    stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")
    if proc.returncode != 0 or not dst.is_file():
        tail = "\n".join((stdout + stderr).splitlines()[-25:])
        raise RetopoError(f"Blender failed (exit {proc.returncode}):\n{tail}")
    report = parse_report(stdout)
    report["path"] = str(dst)
    report["note"] = ("The GLB stores every quad as two triangles (glTF has no quads); the FBX keeps the "
                      "quads and embeds the textures.") if report.get("output_quads") else None
    report["total_seconds"] = round(time.time() - t0, 2)
    return report
