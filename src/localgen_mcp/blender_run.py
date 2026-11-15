"""Run one of the bundled Blender scripts (src/localgen_mcp/blender/) headless and read its report.

Blender runs as a child process on the CPU (Cycles bakes), so it never competes
with ComfyUI for VRAM. Each script prints a last line "<MARKER> {json}".
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from .config import Settings

logger = logging.getLogger("localgen_mcp.blender")

SCRIPTS = Path(__file__).parent / "blender"


class BlenderError(RuntimeError):
    pass


def find_blender(settings: Settings) -> str:
    exe = settings.blender or shutil.which("blender")
    if not exe or not os.access(exe, os.X_OK):
        raise BlenderError("Blender not found; install it or set LOCALGEN_BLENDER to its executable")
    return exe


def command(settings: Settings, script: str, args: list[str]) -> list[str]:
    return [find_blender(settings), "-b", "--factory-startup", "-t", str(settings.blender_threads),
            "-P", str(SCRIPTS / script), "--", *map(str, args)]


def parse_report(stdout: str, marker: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith(marker + " "):
            return json.loads(line[len(marker) + 1:])
    raise BlenderError("Blender finished without a report")


async def run(settings: Settings, script: str, args: list[str], marker: str, expect: Path) -> dict:
    tmp = settings.root / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = command(settings, script, args)
    logger.info("blender: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                                env={**os.environ, "TMPDIR": str(tmp)})
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=settings.retopo_timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise BlenderError(f"Blender timed out after {settings.retopo_timeout} s")
    stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")
    if proc.returncode != 0 or not expect.is_file():
        tail = "\n".join((stdout + stderr).splitlines()[-25:])
        raise BlenderError(f"Blender failed (exit {proc.returncode}):\n{tail}")
    return parse_report(stdout, marker)


PYTHON = sys.executable  # this interpreter, for helpers a Blender script runs (instant_remesh)
