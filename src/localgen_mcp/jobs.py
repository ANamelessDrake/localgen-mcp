"""Submit workflows to the shared backend, wait, and collect the files."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import random
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import backend
from .comfy import ComfyClient, ComfyUnavailable, history_error, output_files
from .config import Settings
from .workflows import Workflow

logger = logging.getLogger("localgen_mcp.jobs")

Progress = Callable[[float, float, str], Awaitable[None]]


class JobError(RuntimeError):
    pass


def _exec_seconds(entry: dict) -> float | None:
    """Execution time from ComfyUI's own execution_start/execution_success timestamps."""
    start = end = None
    for kind, msg in entry.get("status", {}).get("messages", []):
        if kind == "execution_start":
            start = msg.get("timestamp")
        elif kind in ("execution_success", "execution_error"):
            end = msg.get("timestamp")
    if start is None or end is None:
        return None
    return round((end - start) / 1000.0, 2)


_IMAGE_MAGIC = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"BM", b"II*\x00", b"MM\x00*")


def is_image_file(path: Path) -> bool:
    """True when the file starts with a known image signature (checked by content, not extension)."""
    with open(path, "rb") as f:
        head = f.read(16)
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return True
    return any(head.startswith(m) for m in _IMAGE_MAGIC)


MESH_SUFFIXES = (".glb", ".gltf", ".obj", ".stl")


def is_mesh_file(path: Path) -> bool:
    """A 3D model Get3DComponents reads, checked by content so arbitrary files are never uploaded:
    binary glTF by signature, glTF as JSON with an "asset" entry, binary STL by its exact size,
    ASCII STL and OBJ by their keywords."""
    suffix = path.suffix.lower()
    with open(path, "rb") as f:
        head = f.read(65536)
    if suffix == ".glb":
        return head[:4] == b"glTF"
    if suffix == ".gltf":
        return head.lstrip().startswith(b"{") and b'"asset"' in head
    if suffix == ".stl":
        if head[:5] == b"solid" and b"facet" in head:
            return True
        if len(head) >= 84:
            count = int.from_bytes(head[80:84], "little")
            return path.stat().st_size == 84 + 50 * count
        return False
    if suffix == ".obj":
        keys = {"v", "vn", "vt", "f", "o", "g", "s", "mtllib", "usemtl", "#"}
        lines = [ln.split(maxsplit=1) for ln in head.decode("utf-8", errors="replace").splitlines() if ln.strip()]
        return bool(lines) and all(ln[0] in keys or ln[0].startswith("#") for ln in lines[:-1]) \
            and any(ln[0] == "v" for ln in lines)
    return False


def placeholder_png(path: Path) -> Path:
    """An 8x8 black PNG, written once (Load3D wants viewport images even when run headless)."""
    if not path.exists():
        import struct
        import zlib

        def chunk(kind: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        raw = b"".join(b"\0" + b"\0\0\0" * 8 for _ in range(8))
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
    return path


def default_output_dir(settings: Settings, modality: str) -> Path:
    return settings.output_dir / modality / dt.date.today().isoformat()


async def run_workflow(
    settings: Settings,
    workflow: Workflow,
    jobs: list[dict[str, Any]],
    output_dir: str | None = None,
    idle_timeout_seconds: int | None = None,
    images: dict[str, str] | None = None,
    timeout: float = 1800,
    progress: Progress | None = None,
    mesh: str | None = None,
) -> dict:
    """Run one graph per entry of jobs (parameter overrides) and return file paths.

    images maps the workflow's image slot names to local file paths; each file is
    uploaded once and its ComfyUI name passed to the graph as params["images"].

    All graphs are queued at once so the model loads once; ComfyUI runs them in
    order. Seeds default to random and are reported so results reproduce.
    """
    async def report(done: float, total: float, msg: str) -> None:
        if progress is not None:
            try:
                await progress(done, total, msg)
            except Exception:  # progress is best effort
                logger.debug("progress notification failed", exc_info=True)

    missing = workflow.missing_models(settings.comfy_dir / "models")
    if missing:
        files = " ".join(f"--file {Path(m).name}" for m in missing)
        raise JobError(f"{workflow.name} is missing model files {missing}; run setup/download-models.sh {files}")

    image_paths: dict[str, Path] = {}
    for slot, raw in (images or {}).items():
        if not raw:
            continue
        if slot not in workflow.image_slots:
            raise JobError(f"{workflow.name} has no image slot {slot!r}; slots: {list(workflow.image_slots)}")
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise JobError(f"input image not found: {path}")
        if not is_image_file(path):
            # Uploads land in ComfyUI's input folder, which a LAN-bound manual
            # instance would serve; only ever copy real images there.
            raise JobError(f"not a PNG, JPEG, WebP, GIF, BMP or TIFF image: {path}")
        image_paths[slot] = path
    if len(image_paths) < workflow.min_images:
        raise JobError(f"{workflow.name} needs at least {workflow.min_images} input image(s) "
                       f"in slots {list(workflow.image_slots)}")
    mesh_path = None
    if workflow.needs_mesh:
        if not mesh:
            raise JobError(f"{workflow.name} needs an input mesh")
        mesh_path = Path(mesh).expanduser().resolve()
        if not mesh_path.is_file():
            raise JobError(f"input mesh not found: {mesh_path}")
        if mesh_path.suffix.lower() not in MESH_SUFFIXES or not is_mesh_file(mesh_path):
            raise JobError(f"not a GLB, glTF, OBJ or STL model: {mesh_path} (convert FBX to GLB first)")

    params_list = []
    base_seed = None
    for i, overrides in enumerate(jobs):
        p = workflow.params(overrides)
        if p.get("seed") is None:
            if base_seed is None:
                base_seed = random.randint(0, 2**32 - 1)
            p["seed"] = base_seed + i
        p["prefix"] = f"localgen/{workflow.modality}/{uuid.uuid4().hex[:10]}"
        params_list.append(p)

    outdir = Path(output_dir).expanduser() if output_dir else default_output_dir(settings, workflow.modality)
    outdir.mkdir(parents=True, exist_ok=True)

    await report(0, len(params_list), "starting backend")
    t_start = time.time()
    info = await asyncio.to_thread(backend.ensure_backend, settings, idle_timeout_seconds)
    client = ComfyClient(settings.base_url)

    async def submit_all() -> list[str]:
        if image_paths:
            uploaded = {slot: await asyncio.to_thread(client.upload_image, path)
                        for slot, path in image_paths.items()}
            for p in params_list:
                p["images"] = uploaded
        if mesh_path is not None:
            name = await asyncio.to_thread(client.upload_image, mesh_path, "3d/localgen")
            placeholder = await asyncio.to_thread(
                client.upload_image, placeholder_png(settings.root / "tmp" / "localgen_placeholder.png"))
            for p in params_list:
                p["mesh"], p["placeholder"] = name, placeholder
        backend.touch_activity(settings)
        ids = []
        for p in params_list:
            ids.append(await asyncio.to_thread(client.submit, workflow.build(p)))
        return ids

    try:
        prompt_ids = await submit_all()
    except ComfyUnavailable:
        # The backend went away between the check and the submit; start it again once.
        info = await asyncio.to_thread(backend.ensure_backend, settings, idle_timeout_seconds)
        prompt_ids = await submit_all()
    await report(0, len(params_list), f"queued {len(prompt_ids)} job(s)")

    results: list[dict | None] = [None] * len(prompt_ids)
    peak_vram = 0
    deadline = time.time() + timeout * len(prompt_ids)
    retried = False
    while any(r is None for r in results):
        if time.time() > deadline:
            raise JobError(f"timed out waiting for {workflow.name}; prompt ids {prompt_ids}")
        backend.touch_activity(settings)
        gpu = await asyncio.to_thread(backend.nvidia_smi)
        if gpu:
            peak_vram = max(peak_vram, gpu["memory_used_mb"])
        try:
            for i, pid in enumerate(prompt_ids):
                if results[i] is not None:
                    continue
                entry = await asyncio.to_thread(client.history, pid)
                if not entry:
                    continue
                err = history_error(entry)
                if err:
                    raise JobError(f"{workflow.name} failed: {err}")
                if not entry.get("status", {}).get("completed", True):
                    continue
                files = output_files(entry)
                if not files:
                    continue
                paths = [str(await asyncio.to_thread(client.download, f, outdir)) for f in files]
                if info.managed:
                    _remove_comfy_copies(settings, files)
                p = params_list[i]
                results[i] = {
                    "paths": paths,
                    "seed": p["seed"],
                    "exec_seconds": _exec_seconds(entry),
                    "settings": {k: v for k, v in p.items()
                                 if k not in ("prefix", "images", "prompt", "negative_prompt", "mesh", "placeholder")},
                }
                done = sum(r is not None for r in results)
                await report(done, len(results), f"{done}/{len(results)} done")
        except ComfyUnavailable:
            if retried:
                raise
            retried = True
            logger.warning("backend went away while waiting; restarting and resubmitting unfinished jobs")
            info = await asyncio.to_thread(backend.ensure_backend, settings, idle_timeout_seconds)
            todo = [i for i, r in enumerate(results) if r is None]
            for i in todo:
                prompt_ids[i] = await asyncio.to_thread(client.submit, workflow.build(params_list[i]))
            continue
        await asyncio.sleep(1)

    backend.touch_activity(settings)
    effective = backend.read_int(settings.timeout_file) if info.managed else None
    return {
        "workflow": workflow.name,
        "modality": workflow.modality,
        "prompt": params_list[0].get("prompt"),
        "input_images": {slot: str(path) for slot, path in image_paths.items()} or None,
        **({"input_mesh": str(mesh_path)} if mesh_path is not None else {}),
        "outputs": results,
        "paths": [p for r in results for p in r["paths"]],
        "output_dir": str(outdir),
        "total_seconds": round(time.time() - t_start, 2),
        "gpu_memory_used_peak_mb": peak_vram or None,
        **info.as_dict(),
        "idle_timeout_seconds": effective if info.managed else None,
    }


def _remove_comfy_copies(settings: Settings, files: list[dict]) -> None:
    """Delete ComfyUI's own copy after download so outputs are not stored twice."""
    out_root = (settings.comfy_dir / "output").resolve()
    for f in files:
        p = (out_root / f.get("subfolder", "") / f["filename"]).resolve()
        if out_root in p.parents and p.is_file():
            try:
                p.unlink()
            except OSError:
                pass
