"""MCP server entry point: local image, video and mesh generation.

Run with: python -m localgen_mcp.server

The server is deliberately thin: it imports no ML libraries and holds no
models. Every generation call makes sure the shared ComfyUI backend is up
(starting it on demand), submits the workflow, waits, and returns file paths.
The backend shuts itself down after the idle timeout.

Skeleton (logging to stderr, optional log file, signal handling, FastMCP)
follows the kicad-mcp server by the same author.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from mcp.server.fastmcp import Context, FastMCP

from . import backend, config, jobs, workflows


def _configure_logging() -> logging.Logger:
    _logger = logging.getLogger("localgen_mcp")
    level = getattr(logging, os.environ.get("LOCALGEN_MCP_LOG_LEVEL", "INFO").upper(), logging.INFO)
    _logger.setLevel(level)
    # Log to stderr; MCP uses stdout for JSON-RPC.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
    _logger.addHandler(handler)
    _logger.propagate = False
    log_file = os.environ.get("LOCALGEN_MCP_LOG_FILE")
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        _logger.addHandler(fh)
    return _logger


logger = _configure_logging()
settings = config.load()


def _timeout(name: str, default: int) -> int:
    return int(os.environ.get(f"LOCALGEN_TIMEOUT_{name.upper()}", default))


TIMEOUTS = {"image": _timeout("image", 900), "video": _timeout("video", 5400), "mesh": _timeout("mesh", 3600)}

MESH_PROMPT_SUFFIX = (", a single object, centered, full view, isolated on a plain light grey background, "
                      "studio lighting, no shadows, product photo")

mcp = FastMCP(
    "localgen",
    instructions=(
        "Local, offline image, video and 3D mesh generation on this machine's GPU. Tools return file "
        "paths (open them to view). The shared backend starts on the first call (about 10 s, plus model "
        "load) and stops itself after an idle timeout (default 300 s). Pass idle_timeout_seconds, or call "
        "start_backend first, when planning a long batch; call stop_backend when done to free GPU memory. "
        "Only one large model fits in VRAM at a time; switching modality reloads models."
    ),
)


def _progress(ctx: Context | None):
    if ctx is None:
        return None

    async def report(done: float, total: float, message: str) -> None:
        await ctx.report_progress(done, total, message)

    return report


# -- generation -----------------------------------------------------------


@mcp.tool()
async def generate_image(
    prompt: str,
    count: int = 1,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    steps: int | None = None,
    reference_images: list[str] | None = None,
    output_dir: str | None = None,
    idle_timeout_seconds: int | None = None,
    model: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """Generate images from a text prompt, optionally guided by reference images.

    Without reference_images the default model is Z-Image Turbo (1024x1024, 8 steps).
    reference_images: up to 4 local image paths to edit or combine; the prompt says what to do
    with them (for example "change the jacket to red", "the person from image 1 in the setting of
    image 2"). This uses FLUX.2 Klein 4B; output size follows the first reference unless width and
    height are given.
    count > 1 queues all images at once (one model load); seeds are seed, seed+1, ...
    Returns absolute PNG paths, the seeds and settings used, timings and the effective
    backend idle timeout. Default output folder: /ai/output/localgen/image/<date>/.
    idle_timeout_seconds: keep the backend up this long after the last call (bounded, default 300).
    """
    if not 1 <= count <= 64:
        raise ValueError("count must be between 1 and 64")
    refs = [r for r in (reference_images or []) if r]
    images = {f"ref{i + 1}": path for i, path in enumerate(refs)}
    wf = workflows.pick("image", model, set(images))
    overrides = {"prompt": prompt, "width": width, "height": height, "steps": steps}
    job_list = [dict(overrides, seed=None if seed is None else seed + i) for i in range(count)]
    logger.info("generate_image: %d x %r refs=%d (%s)", count, prompt[:80], len(refs), wf.name)
    return await jobs.run_workflow(settings, wf, job_list, output_dir, idle_timeout_seconds,
                                   images=images, timeout=TIMEOUTS["image"], progress=_progress(ctx))


@mcp.tool()
async def generate_video(
    prompt: str,
    input_image: str | None = None,
    width: int | None = None,
    height: int | None = None,
    frames: int | None = None,
    fps: int | None = None,
    steps: int | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
    output_dir: str | None = None,
    idle_timeout_seconds: int | None = None,
    model: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """Generate a video from a text prompt, or animate input_image (a local image path) guided by the prompt.

    Default model: Wan 2.2 TI2V 5B at 1280x704, 49 frames (about 2 s) at 24 fps, 20 steps, MP4 (H.264).
    frames must be 4n+1 (rounded down otherwise), up to 121 (5 s). Expect several minutes per clip on
    this GPU; longer clips and higher resolutions take proportionally longer.
    Returns the MP4 path, seed, settings, timings and the effective backend idle timeout.
    """
    images = {"start_image": input_image} if input_image else {}
    wf = workflows.pick("video", model, set(images))
    overrides = {"prompt": prompt, "width": width, "height": height, "frames": frames, "fps": fps,
                 "steps": steps, "seed": seed, "negative_prompt": negative_prompt}
    logger.info("generate_video: %r image=%s", prompt[:80], input_image)
    return await jobs.run_workflow(settings, wf, [overrides], output_dir, idle_timeout_seconds,
                                   images=images, timeout=TIMEOUTS["video"], progress=_progress(ctx))


@mcp.tool()
async def generate_mesh(
    input_image: str | None = None,
    prompt: str | None = None,
    left_image: str | None = None,
    back_image: str | None = None,
    right_image: str | None = None,
    resolution: int | None = None,
    texture_size: int | None = None,
    face_count: int | None = None,
    remove_background: bool = True,
    framing: str = "crop",
    seed: int | None = None,
    output_dir: str | None = None,
    idle_timeout_seconds: int | None = None,
    model: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """Generate a textured 3D mesh (GLB with PBR materials) from an image of one object.

    Give input_image (a local path), or give only prompt to first generate a reference image with the
    image model (a plain background suffix is added) and then mesh it.
    Default model: TRELLIS.2 with shape resolution 1024, 2048 px textures, 300k faces.
    Multi-view: also give left_image, back_image and/or right_image (views 90 degrees apart at the
    same scale, input_image being the front) to use Pixal3D multi-view instead.
    framing (multi-view only): "crop" (default) crops each view to its own outline, which suits
    figures; it stretches the side views of objects wider from the front than the side. "as_framed"
    uses square views exactly as given (object centred, about 1/1.1 of the frame at its widest, one
    scale for all views, as from a 3D render rig at 20 degrees horizontal FOV).
    remove_background=False uses the image's own alpha channel instead of BiRefNet.
    Returns the GLB path, the reference image path when one was generated, seed, settings and timings.
    """
    if not input_image and not prompt:
        raise ValueError("give input_image or prompt")
    side_views = {"left": left_image, "back": back_image, "right": right_image}
    side_views = {k: v for k, v in side_views.items() if v}
    if side_views and not input_image:
        raise ValueError("multi-view needs input_image as the front view")
    reference = None
    if not input_image:
        img = await generate_image(prompt + MESH_PROMPT_SUFFIX, seed=seed, output_dir=output_dir,
                                   idle_timeout_seconds=idle_timeout_seconds, ctx=ctx)
        input_image = img["paths"][0]
        reference = img
    if side_views:
        images = {"front": input_image, **side_views}
    else:
        images = {"image": input_image}
        if model and "image" not in workflows.get("mesh", model).image_slots:
            images = {"front": input_image}  # a multi-view model asked for with a single view
    wf = workflows.pick("mesh", model, set(images))
    overrides = {"resolution": resolution, "texture_size": texture_size, "face_count": face_count,
                 "remove_background": remove_background, "seed": seed}
    if "framing" in wf.defaults:
        overrides["framing"] = framing
    logger.info("generate_mesh: %s (%s)", images, wf.name)
    result = await jobs.run_workflow(settings, wf, [overrides], output_dir, idle_timeout_seconds,
                                     images=images, timeout=TIMEOUTS["mesh"], progress=_progress(ctx))
    if reference is not None:
        result["reference_image"] = reference["paths"][0]
        result["reference_image_seconds"] = reference["total_seconds"]
    return result


# -- backend control ------------------------------------------------------


def _models_report() -> dict:
    models_dir = settings.comfy_dir / "models"
    report = {}
    for wf in workflows.REGISTRY.values():
        missing = wf.missing_models(models_dir)
        report[wf.name] = {"modality": wf.modality, "default": wf.is_default, "installed": not missing,
                           "missing_files": missing, "description": wf.description,
                           "image_input": wf.image_input, "image_slots": list(wf.image_slots),
                           "defaults": wf.defaults}
    return report


@mcp.tool()
async def status() -> dict:
    """Backend state: running or not, managed by localgen or not, effective idle timeout and bounds,
    queue length, GPU memory, and which workflows have their model files installed."""
    out = await asyncio.to_thread(backend.status, settings)
    out["workflows"] = _models_report()
    out["output_dir"] = str(settings.output_dir)
    return out


@mcp.tool()
async def start_backend(idle_timeout_seconds: int | None = None) -> dict:
    """Start (or keep) the shared backend ahead of a batch.

    idle_timeout_seconds: how long the backend stays up after the last activity. If this call starts
    the backend the value is used as is; if it is already running, a longer value extends it and a
    shorter one is ignored (the effective value is the maximum requested since it started). Bounded
    to the configured min and max (default 60 to 7200 s). Returns the effective value.
    """
    info = await asyncio.to_thread(backend.ensure_backend, settings, idle_timeout_seconds)
    out = info.as_dict()
    out["idle_timeout_bounds_seconds"] = [settings.idle_timeout_min, settings.idle_timeout_max]
    if not info.managed:
        out["note"] = "an unmanaged ComfyUI (for example run.sh) is on the port; idle timeout does not apply"
    return out


@mcp.tool()
async def stop_backend(force: bool = False) -> dict:
    """Stop the shared backend now and free all GPU and system memory it holds.

    Refuses while jobs are queued or running (possibly another agent's) unless force=True.
    An unmanaged ComfyUI (started by hand) is never stopped.
    """
    return await asyncio.to_thread(backend.stop_backend, settings, force)


# -- entry point ----------------------------------------------------------


def main() -> None:
    def _handle_shutdown(signum: int, _frame: object) -> None:
        logger.info("Received %s, shutting down...", signal.Signals(signum).name)
        os._exit(0)

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)
    logger.info("localgen MCP server starting (backend %s, idle timeout %ss)",
                settings.base_url, settings.idle_timeout)
    mcp.run()


if __name__ == "__main__":
    main()
