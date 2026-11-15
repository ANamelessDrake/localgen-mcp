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
import json
import logging
import os
import re
import signal
import shutil
import sys
import uuid
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP

from . import backend, blender_run, config, cutout, jobs, materials, model_files, retopo, rig, speech, workflows
from .workflows.image_edit import edit_images
from .workflows.part_isolate import part_prompt


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

MESH_INPUTS = (".glb", ".gltf", ".obj", ".stl", ".ply", ".fbx")

MESH_PROMPT_SUFFIX = (", a single object, centered, full view, isolated on a plain light grey background, "
                      "studio lighting, no shadows, product photo")

mcp = FastMCP(
    "localgen",
    instructions=(
        "Local, offline image, video and 3D mesh generation on this machine's GPU. Tools return file "
        "paths (open them to view). The shared backend starts on the first call (about 10 s, plus model "
        "load) and stops itself after an idle timeout (default 300 s). Pass idle_timeout_seconds, or call "
        "start_backend first, when planning a long batch; call stop_backend when done to free GPU memory. "
        "Only one large model fits in VRAM at a time; switching modality reloads models. "
        "speak (text to speech) runs on the CPU without the backend; list_voices shows its voices."
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
    input_image: str | None = None,
    denoise: float | None = None,
    mask_image: str | None = None,
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
    input_image: a local image to start from (image to image) instead of noise. The output keeps
    its size (width and height are not allowed); it is worked on at about 1 megapixel. The prompt
    describes the whole desired result, not only the change.
    denoise (0.05 to 1.0, needs input_image): how much may change. Z-Image Turbo default 0.6.
    Z-Image holds on to the input strongly: 0.3 and 0.6 change only fine detail, 0.9 keeps the
    layout but changes colours and materials, 1.0 ignores the input (except outside a mask).
    mask_image (needs input_image): a black and white image, white = change, black = keep; any
    size (it is stretched to input_image). Only the white area is regenerated, and the result is
    pasted back over input_image with an 8 pixel feathered edge, so every pixel more than about
    18 px outside the white area is exactly the original. To replace what is in the region use
    denoise 1.0; to recolour an object, Klein (model="flux2-klein-4b") follows the prompt better
    than Z-Image at 0.9, which keeps some of the old colour.
    input_image with reference_images (or model="flux2-klein-4b"): FLUX.2 Klein edits input_image
    as image 1 and the references are images 2 to 4 (at most 3), for example "give the person in
    image 1 the hat from image 2"; Klein's denoise default is 1.0 (a full edit guided by image 1),
    and mask_image limits it to a region the same way.
    count > 1 queues all images at once (one model load); seeds are seed, seed+1, ...
    Returns absolute PNG paths, the seeds and settings used, timings and the effective
    backend idle timeout. Default output folder: /ai/output/localgen/image/<date>/.
    idle_timeout_seconds: keep the backend up this long after the last call (bounded, default 300).
    """
    if not 1 <= count <= 64:
        raise ValueError("count must be between 1 and 64")
    refs = [r for r in (reference_images or []) if r]
    images = edit_images(input_image, mask_image, refs, denoise, width, height)
    wf = workflows.pick("image", model, set(images))
    overrides = {"prompt": prompt, "width": width, "height": height, "steps": steps, "denoise": denoise}
    job_list = [dict(overrides, seed=None if seed is None else seed + i) for i in range(count)]
    logger.info("generate_image: %d x %r refs=%d image=%s mask=%s (%s)", count, prompt[:80], len(refs),
                input_image, mask_image, wf.name)
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
    retopo_faces: int | None = None,
    retopo_method: str = "quad",
    retopo_lods: list[int] | None = None,
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
    retopo_faces (for example 10000): also make a game weight copy with retopologize_mesh
    (retopo_method "quad" or "decimate"); the result's "retopo" entry has its path. retopo_lods (for
    example [10000, 3000, 800]) makes a LOD chain instead, as retopologize_mesh's lods. For a plain
    triangle mesh at a given size, face_count alone does it inside the generation.
    Returns the GLB path, the reference image path when one was generated, seed, settings and timings.
    """
    if not input_image and not prompt:
        raise ValueError("give input_image or prompt")
    retopo.validate_lods(retopo_lods)  # before the GPU work, not after it
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
    if retopo_faces or retopo_lods:
        result["retopo"] = await retopo.retopologize(settings, result["paths"][0], retopo_faces or 10000,
                                                     retopo_method, texture_size or 2048, output_dir,
                                                     lods=retopo_lods)
    return result


@mcp.tool()
async def retopologize_mesh(
    input_mesh: str,
    target_faces: int = 10000,
    method: str = "quad",
    texture_size: int = 2048,
    output_dir: str | None = None,
    detail: str | None = None,
    detail_strength: float = 1.0,
    lods: list[int] | None = None,
) -> dict:
    """Make a game weight copy of a textured GLB: new topology, new UVs, and baked PBR textures.

    method "quad" (default): an all quad mesh with even edge flow (Instant Meshes), suited to
    rigging, deformation and further modelling. "decimate": triangles, closest silhouette.
    The new mesh gets a fresh UV layout, and base color, roughness, metallic and a tangent space
    normal map baked from the original (target_faces 200 to 500000, texture_size 256 to 8192).
    Where the faces go (method "decimate" only): detail "auto" (default) keeps more faces where the
    surface bends sharply (fingers, joints, face features) and fewer on flat panels; "paint" reads the
    mesh's vertex colours as a detail map you painted in Blender (white = most detail, black = least;
    export as GLB); "none" is uniform. detail_strength 0 to 1 sets how strongly (default 1, the most
    that keeps the shape intact). The quad
    method always uses one density everywhere.
    Runs in headless Blender on the CPU, about 10 to 30 s at 10k faces; the GPU is not used.
    Output: <output_dir or the input's folder>/<name>_retopo_<method>_<faces>.glb, plus an .fbx
    with embedded textures that keeps real quads (glTF stores only triangles).
    Returns the path, face and vertex counts before and after, the method that ran and timings.
    lods (for example [10000, 3000, 800]): a LOD chain for a game engine instead of one mesh. lods[0]
    replaces target_faces and is LOD0, the retopology above with its baked textures; each further
    entry is a triangle budget for a decimation of LOD0 that keeps LOD0's UVs, so every LOD uses the
    same textures and one material. Entries must be strictly decreasing (1 to 8 of them). Output:
    <name>_retopo_<method>_lods_<budgets>.glb and .fbx holding objects <name>_LOD0, <name>_LOD1, ...
    (Unity's FBX importer builds a LOD Group from that suffix). The report's "lods" lists each LOD's
    face and triangle counts and texture_fit: uv_drift_texels_median (how far its UVs sit from LOD0's
    at the same place; a few texels is right, hundreds means broken mapping) and colour_mismatch_share
    (share of its triangles whose base colour differs from LOD0's there; it grows as small features
    collapse). With method "decimate", the detail map also steers the LOD decimations.
    """
    return await retopo.retopologize(settings, input_mesh, target_faces, method, texture_size, output_dir,
                                     detail, detail_strength, lods)


@mcp.tool()
async def rig_creature(
    input_mesh: str,
    preset: str = "auto",
    bones: int | None = None,
    axis: str | list[float] | None = None,
    root_end: str | list[float] | None = None,
    wing_chains: int = 3,
    wing_bones: int = 4,
    weights: str = "auto",
    name: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """Soft rig for an organic creature (tentacles, tails, vines, rays, fish): bone chains with smooth skin
    weights, so engine scripts or physics (Verlet, springs) can bend it. No animation clips are made.

    Call it on a clean mesh: run retopologize_mesh first (generated meshes are about 300k faces with
    stray shells; a quad retopology of 5k to 20k faces skins and deforms far better). input_mesh: a GLB.
    preset:
      "chain": one chain of `bones` (default 8) along the mesh's main axis, or `axis`; for tentacles,
        tails, vines. Bones are named <name>_01.. from the root (name default "tentacle").
      "spine": the same along the body, named spine_01.. (default 6 bones).
      "wings": stingray, manta, bat or bird: a spine along the body length (`bones`, default 6) and,
        on each side, `wing_chains` (default 3) chains of `wing_bones` (default 4) fanning from the body
        to the wing edge. Named spine_01.., wing_L1_01.. and the mirrored wing_R1_01.. (L is the
        positive side of the left-right axis: +X when that is X, the creature's left when it faces -Y as
        generated meshes do; chain 1 is nearest the head).
      "auto" (default): chain if long and thin, wings if flat and broad, else spine; the report says
        which and why.
    axis: "x", "y", "z" or [x, y, z]: the chain direction, or the body length for "wings"
    (default: from the mesh's principal axes; for wings, the left-right axis is the one the mesh is
    mirror symmetric across).
    root_end: which end the chain (or the spine, at the head) starts from: "min" or "max" along the
    axis, or a point [x, y, z] (the end nearest it). Default: a vertical chain starts at its lower end,
    otherwise at the thicker end (a tentacle's base); wings start at the wider end (a tail is narrow).
    weights: "auto" tries Blender's heat weights and falls back to a smooth distance falloff from the
    bones when they fail (more than 1% of the vertices unweighted or on a far bone, common on generated
    meshes); "distance" always uses the falloff. Every vertex gets at most 4 influences, normalised.
    Bone axes: local Y runs along each bone; rotate about local X to bend (for wings, local Z is the
    surface normal, so local X rotation flaps up and down).
    Output: <output_dir or the input's folder>/<input name>_rig_<preset>.glb and .fbx (armature and skinned
    mesh, rest pose). Returns the bones (head, tail, parent), the preset used, weight coverage and
    method, and warnings. Headless Blender on the CPU, a few seconds.
    """
    return await rig.rig_creature(settings, input_mesh, preset, bones, axis, root_end, wing_chains, wing_bones,
                                  weights, name, output_dir)


def _reuse_file(root: Path, entry, kind: str) -> str | None:
    """A file recorded in a parts state file, only if it is a relative path inside root and still passes
    the checks a fresh input would. The state file sits in a caller-chosen folder, so it is not trusted."""
    if not isinstance(entry, str) or Path(entry).is_absolute():
        return None
    path = (root / entry).resolve()
    if root not in path.parents or not path.is_file():
        return None
    try:
        if kind == "image":
            return str(path) if jobs.is_image_file(path) else None
        if path.suffix.lower() != ".glb" or not retopo.is_glb(path):
            return None
        model_files.require_self_contained(path)
        return str(path)
    except (OSError, ValueError):
        return None


def _part_boxes(parts: list[dict], width: int, height: int) -> list[dict]:
    """Validate the part list; boxes given as fractions (0..1) become pixels."""
    if not parts:
        raise ValueError("give at least one part")
    seen, out = set(), []
    for i, p in enumerate(parts):
        if not isinstance(p, dict) or "box" not in p:
            raise ValueError(f"part {i} needs a box [x0, y0, x1, y1]")
        name = re.sub(r"[^A-Za-z0-9_]+", "_", str(p.get("name") or f"part{i}")).strip("_") or f"part{i}"
        if name in seen:
            raise ValueError(f"duplicate part name {name!r}")
        seen.add(name)
        box = [float(c) for c in p["box"]]
        if len(box) != 4:
            raise ValueError(f"{name}: box must be [x0, y0, x1, y1]")
        if max(box) <= 1.0:
            box = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
        x0, y0, x1, y1 = (int(round(c)) for c in box)
        x0, x1 = sorted((max(0, min(width, x0)), max(0, min(width, x1))))
        y0, y1 = sorted((max(0, min(height, y0)), max(0, min(height, y1))))
        if x1 - x0 < 16 or y1 - y0 < 16:
            raise ValueError(f"{name}: box is too small")
        out.append({"name": name, "box": [x0, y0, x1, y1], "parent": p.get("parent"),
                    "description": str(p.get("description") or "")})
    return out


GUIDE_FACES = 50000  # the guide mesh only has to carry the shape; fewer faces render faster


@mcp.tool()
async def generate_turnaround(
    concept_image: str | None = None,
    prompt: str | None = None,
    subject: str | None = None,
    three_quarter: bool = False,
    symmetric: bool = False,
    background: str = "#d8d8d8",
    mesh: bool = False,
    guide: str = "none",
    guide_denoise: float | None = None,
    guide_source: str = "front",
    view_size: int | None = None,
    steps: int | None = None,
    seed: int | None = None,
    output_dir: str | None = None,
    idle_timeout_seconds: int | None = None,
    ctx: Context | None = None,
) -> dict:
    """Make consistent turnaround views of one object (front, left, back and right) from a concept image.

    Give concept_image (a local PNG or JPEG of the object, any angle), or only prompt to generate the
    concept first with the image model. subject: a few words naming the object and its key parts
    ("a salvage robot with a cargo cage on its back and tread feet"); it goes into every view's prompt
    and defaults to prompt. Naming the parts helps them appear in every view.
    FLUX.2 Klein first redraws the concept as a clean front view (flat even lighting, no cast shadows,
    plain background). Views follow Pixal3D's rig: the left view shows the object's own left side,
    facing image-left.
    guide="none" (default, better for standing objects): each other view is drawn from the clean
    front and the concept from an empty latent, then scaled by silhouette height (which assumes a
    standing object seen level). Flat, winged or wide objects tend to come out facing the camera in
    every view (a moth with its wings spread flat in the side views).
    guide="mesh" (for flat, winged or wide objects): TRELLIS.2 meshes the clean front view (a guide
    mesh; guide_source="concept" meshes the concept instead, and a failure from the front retries
    from the concept), Blender renders it from the rig's cameras (turned by guide_yaw, found by
    matching the mesh to that image, since TRELLIS.2 does not always face a mesh's front to the
    rig's front), and Klein repaints each render image to image (guide_denoise, default 0.6, 0.3 to
    0.95: lower keeps more of the render) with the clean front and the concept as references. The
    renders fix each view's pose and silhouette, so a moth's wings are edge-on from the side, and
    the views come out already framed for the rig. The views are only as right as TRELLIS.2's guess
    at the hidden sides: on a robot it copied the face onto the back, which guide="none" got right.
    three_quarter=True adds front_left, back_left, back_right and front_right views (for reference;
    meshing uses the four main views). symmetric=True draws only the left-hand views and mirrors them
    into the right-hand ones, which keeps a left-right symmetric object consistent (do not use it for
    asymmetric objects).
    Every view is square, the object centred at one scale, the widest view at 1/1.1 of the frame.
    Views are RGBA PNGs (alpha is a BiRefNet foreground mask) on `background`; a sheet puts them
    side by side.
    mesh=True also meshes the four views with generate_mesh's multi-view path (framing "as_framed").
    Views are generated, so they can still disagree on details: look at them before relying on them.
    Returns the view paths by name, the sheet, the concept, the clean front, and with guide="mesh"
    the guide mesh ("guide_mesh", a usable single view TRELLIS.2 mesh on its own) and its renders
    ("guide_renders"); with guide="none" the raw (unframed) views; the mesh result when asked.
    About 6 s per view warm plus model loads; guide="mesh" adds a TRELLIS.2 run (about 1 to 2
    minutes) and the Blender renders (about 10 s per view on the CPU).
    """
    from .workflows import turnaround

    if not concept_image and not prompt:
        raise ValueError("give concept_image or prompt")
    wf = workflows.get("turnaround")
    overrides = {"subject": (subject or prompt or "the object").strip().rstrip("."), "three_quarter": three_quarter,
                 "symmetric": symmetric, "background": background, "view_size": view_size, "steps": steps,
                 "seed": seed, "guide": guide, "guide_denoise": guide_denoise}
    params = wf.params(overrides)
    turnaround.check(params)  # fail on bad options before any GPU work
    if guide_source not in ("front", "concept"):
        raise ValueError('guide_source must be "front" or "concept"')
    concept = None
    if concept_image:
        path = Path(concept_image).expanduser().resolve()
        if not path.is_file() or not jobs.is_image_file(path):
            raise ValueError(f"concept_image must be a PNG or JPEG image: {path}")
        concept = str(path)
    progress = _progress(ctx)
    reference = None
    if concept is None:
        reference = await generate_image(prompt + turnaround.CONCEPT_SUFFIX, seed=seed, output_dir=output_dir,
                                         idle_timeout_seconds=idle_timeout_seconds, ctx=ctx)
        concept = reference["paths"][0]
    logger.info("generate_turnaround: %s (%s, guide=%s)", concept, wf.name, guide)
    if guide == "mesh":
        result = await _guided_turnaround(wf, overrides, params, concept, guide_source, output_dir,
                                          idle_timeout_seconds, ctx)
    else:
        result = await jobs.run_workflow(settings, wf, [{**overrides, "stage": "full"}], output_dir,
                                         idle_timeout_seconds, images={"concept": concept},
                                         timeout=TIMEOUTS["image"], progress=progress)
        result.update(turnaround.classify(result["paths"]))
        result["clean_front"] = result["raw"].get("front")
    result["concept_image"] = concept
    if reference is not None:
        result["concept_seconds"] = reference["total_seconds"]
    if mesh:
        v = result["views"]
        result["mesh"] = await generate_mesh(input_image=v["front"], left_image=v.get("left"),
                                             back_image=v.get("back"), right_image=v.get("right"),
                                             framing="as_framed", remove_background=False, seed=seed,
                                             output_dir=output_dir, idle_timeout_seconds=idle_timeout_seconds,
                                             ctx=ctx)
    return result


async def _guided_turnaround(wf, overrides, params, concept, guide_source, output_dir, idle_timeout_seconds,
                             ctx) -> dict:
    """guide="mesh": clean front, TRELLIS.2 guide mesh, rig renders in Blender, views painted from the renders."""
    from .workflows import turnaround

    progress = _progress(ctx)
    front = await jobs.run_workflow(settings, wf, [{**overrides, "stage": "front"}], output_dir,
                                    idle_timeout_seconds, images={"concept": concept},
                                    timeout=TIMEOUTS["image"], progress=progress)
    clean_front = turnaround.classify(front["paths"])["clean_front"]
    seed = front["outputs"][0]["seed"]  # one seed for every stage, so a run reproduces from the reported seed
    sources = {"front": clean_front, "concept": concept}
    order = [guide_source] + [k for k in sources if k != guide_source]
    warnings = []
    for i, source in enumerate(order):
        try:
            guide_mesh = await generate_mesh(input_image=sources[source], face_count=GUIDE_FACES, texture_size=1024,
                                             seed=seed, output_dir=front["output_dir"],
                                             idle_timeout_seconds=idle_timeout_seconds, ctx=ctx)
            break
        except jobs.JobError as e:
            # A furry or spiky clean front can make TRELLIS.2 extract tens of millions of faces, whose
            # remesh runs out of GPU memory; the concept usually meshes. The concept must then show the
            # object from the front, since the guide mesh's front is the image's view.
            if i + 1 == len(order):
                raise
            warnings.append(f"guide mesh from the {source} failed ({str(e).splitlines()[0][:200]}); "
                            f"made it from the {order[i + 1]} instead")
            logger.warning("generate_turnaround: %s", warnings[-1])
    glb = guide_mesh["paths"][0]
    views = turnaround.ordered(turnaround.views_for(bool(params["three_quarter"])))
    todo = turnaround.sampled(views, bool(params["symmetric"]))
    stem = Path(clean_front).name.split("_clean_front")[0]
    prefix = Path(front["output_dir"]) / f"{stem}_guide"
    bg = params["background"]
    spec = {"mesh": glb, "prefix": str(prefix), "size": int(params["view_size"]), "samples": 32,
            "views": {v: turnaround.ALL_VIEWS[v] for v in todo},
            "background": [int(bg[i:i + 2], 16) for i in (1, 3, 5)],
            "match": sources[source]}  # TRELLIS.2 does not always face the mesh's front to -Y
    renders = await blender_run.run(settings, "render_rig_views.py", [json.dumps(spec)], "RIG_VIEWS_RESULT",
                                    Path(f"{prefix}_{todo[-1]}.png"))
    images = {"concept": concept, "front": clean_front,
              **{turnaround.GUIDE_SLOT + v: p for v, p in renders["views"].items()}}
    result = await jobs.run_workflow(settings, wf, [{**overrides, "stage": "paint", "seed": seed}],
                                     front["output_dir"], idle_timeout_seconds, images=images,
                                     timeout=TIMEOUTS["image"], progress=progress)
    result.update(turnaround.classify(result["paths"]))
    result["paths"] = [clean_front, glb, *renders["views"].values(), *result["paths"]]
    result["clean_front"] = clean_front
    result["guide_mesh"] = glb
    result["seed"] = seed
    result["guide_source"] = source
    result["warnings"] = warnings
    result["guide_renders"] = renders["views"]
    result["guide_bbox"] = renders["bbox"]
    result["guide_yaw"] = renders.get("yaw", 0)
    result["stage_seconds"] = {"front": front["total_seconds"], "guide_mesh": guide_mesh["total_seconds"],
                               "renders": renders["seconds"], "paint": result["total_seconds"]}
    result["total_seconds"] = round(sum(result["stage_seconds"].values()), 2)
    return result


@mcp.tool()
async def generate_material(
    prompt: str,
    size: int = 1024,
    reference_image: str | None = None,
    tile_image: str | None = None,
    name: str | None = None,
    normal_strength: float = 1.0,
    directx_normal: bool = False,
    invert_height: bool = False,
    roughness: float = 0.7,
    roughness_variation: float = 0.25,
    ao_strength: float = 0.6,
    seam_passes: int = 2,
    seam_denoise: float = 0.7,
    seam_width: float = 0.25,
    preview: str = "sphere",
    seed: int | None = None,
    output_dir: str | None = None,
    idle_timeout_seconds: int | None = None,
    ctx: Context | None = None,
) -> dict:
    """Make a tileable (seamless) PBR surface material for environment art: rock, rusted metal, floors.

    prompt describes the surface only ("rough grey basalt rock with cracks"); "seamless top-down texture"
    wording is added. size 1024 (default) or 2048 (512 to 2048, a multiple of 64). reference_image: a
    picture whose colours and material to follow (FLUX.2 Klein). tile_image: make this existing image
    tileable instead of generating one (the prompt still guides the seam repaint).
    How it tiles: the image is offset by half its size with wrap around, then the cross where the old
    edges meet is repainted image to image under a soft, ragged mask (Z-Image Turbo) and only the
    masked part composited back; the arms stop short of the edges and the leftover spots are
    repainted after a further quarter offset, so no repaint ever crosses an edge and the result
    wraps exactly. seam_width (0.05 to 0.3 of the size, default 0.25) is how wide the band is;
    seam_denoise (0.3 to 1.0, default 0.7) how freely it is repainted: higher hides a stubborn seam
    but drifts from the surrounding texture. The wrap and the repainted centre are then measured;
    while the centre still shows a line, the repaint runs again (seam_passes in all, default 2; a
    rerun is kept only if it helps).
    Maps, derived from the base color in Blender on the CPU, all tiling: height (16 bit; luminance,
    high pass, light blur; brighter is higher, invert_height=True for the reverse), normal (tangent
    space, OpenGL +Y green up as Unity and Blender expect; directx_normal=True writes DirectX -Y for
    Unreal; normal_strength 0 to 10), roughness (roughness 0 to 1 is the base, 0.2 glossy, 0.8 matte;
    rougher in cavities and dark spots by up to roughness_variation), ambient occlusion from the
    height's cavities (ao_strength 0 to 1).
    Previews: a 3x3 tiling of the base color, and a lit Cycles render (preview "sphere", "plane" or
    "none") using every map. Look at both to judge seams and relief.
    Output: <output_dir or /ai/output/localgen/material/<date>>/<name>_<seed>/ with <stem>_basecolor.png,
    _height.png, _normal_gl.png (or _normal_dx.png), _roughness.png, _ao.png, _tile3x3.png,
    _preview.png. Returns the paths, seed, the normal convention, and seam metrics per map ("tiles"
    is true when every wrap edge is within the image's own neighbouring pixel differences).
    About 25 s on the GPU at 1024 when warm (longer at 2048, and again for each extra seam pass),
    plus 5 to 15 s in Blender on the CPU.
    """
    logger.info("generate_material: %r size=%d", prompt[:80], size)
    return await materials.generate(
        settings, prompt, size=size, reference_image=reference_image, tile_image=tile_image, name=name,
        normal_strength=normal_strength, directx_normal=directx_normal, invert_height=invert_height,
        roughness=roughness, roughness_variation=roughness_variation, ao_strength=ao_strength,
        seam_passes=seam_passes, seam_denoise=seam_denoise, seam_width=seam_width, preview=preview, seed=seed,
        output_dir=output_dir, idle_timeout_seconds=idle_timeout_seconds, timeout=TIMEOUTS["image"], progress=_progress(ctx))


@mcp.tool()
async def generate_parts(
    concept_image: str,
    parts: list[dict],
    subject: str = "character",
    full_mesh: str | None = None,
    face_count: int = 20000,
    rig: bool = True,
    seed: int | None = None,
    output_dir: str | None = None,
    idle_timeout_seconds: int | None = None,
    ctx: Context | None = None,
) -> dict:
    """Build a model as separate part meshes from a front view concept, assembled and rigidly rigged.

    parts: one entry per part, {"name": "left_arm", "box": [x0, y0, x1, y1], "parent": optional part
    name, "description": optional words to tell it apart}. The box is where that part is in
    concept_image, as fractions of the image (0 to 1) or pixels; look at the image and give a rough
    box around each part (overlaps are fine). subject names the object ("robot", "stingray").
    Each part is cropped, cleaned into a standalone part by FLUX.2 Klein (hidden areas completed) and
    meshed by TRELLIS.2 with face_count faces. The parts are placed by their boxes, with depth from a
    whole-object mesh (generated from the concept unless full_mesh, a GLB of it, is given).
    rig=True adds an armature with one bone per part: the hierarchy follows the parents given, else
    which parts touch (rooted at the largest part), and each bone pivots where its part meets its
    parent. Parts are parented rigidly to their bones, the natural rig for robots and other hard
    surface models. If a run fails part way, call again with the same output_dir: finished parts are
    reused. Output: an assembled GLB and FBX (each part a separate object), plus every part's
    image and mesh. About 1 to 2 minutes per part on the GPU.
    """
    concept = Path(concept_image).expanduser().resolve()
    if not concept.is_file() or not jobs.is_image_file(concept):
        raise ValueError(f"concept_image must be a PNG or JPEG: {concept}")
    width, height = jobs.image_size(concept)
    specs = _part_boxes(parts, width, height)
    if full_mesh:
        fm = Path(full_mesh).expanduser().resolve()
        if fm.suffix.lower() != ".glb" or not retopo.is_glb(fm):
            raise ValueError(f"full_mesh must be a GLB: {fm}")
        model_files.require_self_contained(fm)
    outdir = (Path(output_dir).expanduser() if output_dir
              else jobs.default_output_dir(settings, "parts") / uuid.uuid4().hex[:10])
    outdir.mkdir(parents=True, exist_ok=True)
    progress = _progress(ctx)
    # Resume: a rerun into the same output_dir reuses every part image and mesh made with the same
    # settings, so one failure (say, the GPU running out of memory) does not cost the finished parts.
    state_path = outdir / "parts_state.json"
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        state = {}

    def key(sp: dict, i: int) -> str:
        return json.dumps([sp["name"], sp["box"], sp["description"], subject, str(concept),
                           concept.stat().st_mtime, None if seed is None else seed + i, face_count])

    def save_state() -> None:
        state_path.write_text(json.dumps(state, indent=1))

    root = outdir.resolve()

    def rel(path: str) -> str:
        return str(Path(path).resolve().relative_to(root))

    def reuse(entry, kind):
        return _reuse_file(root, entry, kind)

    for i, sp in enumerate(specs):
        old = state.get(sp["name"], {})
        if isinstance(old, dict) and old.get("key") == key(sp, i):
            for kind in ("image", "mesh"):
                found = reuse(old.get(kind), kind)
                if found:
                    sp[kind] = found

    # 1. Isolate the parts not done yet in one batch, so Klein loads once.
    iso = workflows.get("part")
    overrides, todo = [], []
    for i, sp in enumerate(specs):
        if sp.get("image"):
            continue
        todo.append((i, sp))
        x0, y0, x1, y1 = sp["box"]
        pad_x, pad_y = int(0.04 * (x1 - x0)), int(0.04 * (y1 - y0))
        cx0, cy0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
        cx1, cy1 = min(width, x1 + pad_x), min(height, y1 + pad_y)
        overrides.append({"x": cx0, "y": cy0, "w": cx1 - cx0, "h": cy1 - cy0, "seed": None if seed is None else seed + i,
                          "prompt": part_prompt(sp["name"].replace("_", " "), subject, sp["description"])})
    if overrides:
        isolated = await jobs.run_workflow(settings, iso, overrides, str(outdir), idle_timeout_seconds,
                                           images={"concept": str(concept)}, timeout=TIMEOUTS["image"],
                                           progress=progress)
        for (i, sp), out in zip(todo, isolated["outputs"]):
            sp["image"] = out["paths"][0]
            state[sp["name"]] = {"key": key(sp, i), "image": rel(sp["image"])}
        save_state()

    # 2. Mesh the whole object (placement reference) and each part.
    mesh_wf = workflows.get("mesh", "trellis2")
    ref = state.get("_reference") if isinstance(state.get("_reference"), dict) else {}
    if not full_mesh and ref.get("key") == json.dumps([str(concept), concept.stat().st_mtime, seed]):
        full_mesh = reuse(ref.get("mesh"), "mesh")
    if not full_mesh:
        whole = await jobs.run_workflow(settings, mesh_wf, [{"seed": seed}], str(outdir / "reference"),
                                        idle_timeout_seconds, images={"image": str(concept)},
                                        timeout=TIMEOUTS["mesh"], progress=progress)
        full_mesh = whole["paths"][0]
        state["_reference"] = {"key": json.dumps([str(concept), concept.stat().st_mtime, seed]), "mesh": rel(full_mesh)}
        save_state()
    for i, sp in enumerate(specs):
        if sp.get("mesh"):
            continue
        res = await jobs.run_workflow(settings, mesh_wf, [{"seed": seed, "face_count": face_count}],
                                      str(outdir / "parts"), idle_timeout_seconds, images={"image": sp["image"]},
                                      timeout=TIMEOUTS["mesh"], progress=progress)
        sp["mesh"] = res["paths"][0]
        state[sp["name"]]["mesh"] = rel(sp["mesh"])
        save_state()

    # 3. Assemble and rig in Blender.
    out_glb = outdir / "assembled.glb"
    spec = {"concept": str(concept), "full_mesh": str(full_mesh), "rig": rig,
            "out_glb": str(out_glb), "out_fbx": str(outdir / "assembled.fbx"),
            "parts": [{"name": s["name"], "mesh": s["mesh"], "box": s["box"], "parent": s["parent"]} for s in specs]}
    spec_path = outdir / "assembly.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    report = await blender_run.run(settings, "assemble_parts.py", [spec_path], "ASSEMBLE_RESULT", out_glb)
    return {"assembled": str(out_glb), "fbx": report["fbx"], "output_dir": str(outdir), "full_mesh": str(full_mesh),
            "parts": [{**{k: s[k] for k in ("name", "box", "image", "mesh")}, **report["parts"].get(s["name"], {})}
                      for s in specs],
            "root": report.get("root"), "warnings": report.get("warnings", []),
            "paths": [str(out_glb), report["fbx"]] + [s["mesh"] for s in specs]}


@mcp.tool()
async def texture_mesh(
    input_mesh: str,
    prompt: str,
    style_image: str | None = None,
    views: int = 4,
    texture_size: int = 2048,
    denoise: float = 0.9,
    bump: float = 0.5,
    roughness: float = 0.6,
    metallic: float = 0.0,
    keep_uvs: bool = False,
    seed: int | None = None,
    output_dir: str | None = None,
    idle_timeout_seconds: int | None = None,
    ctx: Context | None = None,
) -> dict:
    """Texture an existing mesh (GLB, glTF, OBJ, STL, PLY or FBX; untextured is fine) from a text prompt.

    Builds everything a PBR material needs: a fresh UV layout (keep_uvs=True keeps the mesh's own),
    a base color painted by FLUX.2 Klein on `views` clay renders around the mesh and projected into
    the UVs (style_image: an optional picture whose colours and style the front view follows),
    ambient occlusion, a normal map from a bump derived from the painted detail (bump 0 to 1, 0 for
    none), and roughness (base value, rougher where the paint is darker) and metallic values.
    denoise (0.6 to 1.0) is how freely views are painted: lower keeps more of the clay shading.
    Best on clean meshes of a few thousand to 50k faces, for example from retopologize_mesh.
    Outputs a GLB and an FBX with embedded textures, plus the painted views as PNGs. About 1 to 2
    minutes on the GPU, then a few seconds in Blender on the CPU.
    """
    if not 0.6 <= denoise <= 1.0:
        raise ValueError("denoise must be between 0.6 and 1.0")
    if not (0.0 <= bump <= 1.0 and 0.0 <= roughness <= 1.0 and 0.0 <= metallic <= 1.0):
        raise ValueError("bump, roughness and metallic must be between 0 and 1")
    src = Path(input_mesh).expanduser().resolve()
    if not src.is_file():
        raise ValueError(f"input mesh not found: {src}")
    if src.suffix.lower() not in MESH_INPUTS:
        raise ValueError(f"input mesh must be one of {MESH_INPUTS}")
    model_files.require_self_contained(src)  # no glTF buffer or image URIs pointing at other files
    # Re-serialise through Blender and upload only that: ComfyUI's input folder can be served on
    # the LAN by a hand-started instance, so the caller's own bytes are never copied there.
    work = settings.root / "tmp" / f"texture-{uuid.uuid4().hex[:10]}"
    work.mkdir(parents=True, exist_ok=True)
    clean = work / "input.glb"
    try:
        await blender_run.run(settings, "convert_mesh.py", [src, clean], "CONVERT_RESULT", clean)
        return await _texture(clean, src, prompt, style_image, views, texture_size, denoise, bump, roughness,
                              metallic, keep_uvs, seed, output_dir, idle_timeout_seconds, ctx)
    finally:
        shutil.rmtree(work, ignore_errors=True)


async def _texture(clean, src, prompt, style_image, views, texture_size, denoise, bump, roughness, metallic,
                   keep_uvs, seed, output_dir, idle_timeout_seconds, ctx) -> dict:
    wf = workflows.get("texture")
    overrides = {"prompt": prompt, "views": views, "texture_size": texture_size, "denoise": denoise,
                 "unwrap": not keep_uvs, "seed": seed}
    images = {"style": style_image} if style_image else None
    logger.info("texture_mesh: %s (%s)", src, wf.name)
    result = await jobs.run_workflow(settings, wf, [overrides], output_dir, idle_timeout_seconds, images=images,
                                     timeout=TIMEOUTS["mesh"], progress=_progress(ctx), mesh=str(clean))
    result["input_mesh"] = str(src)
    glb = next(Path(p) for p in result["paths"] if p.endswith(".glb"))
    final = glb.with_name(glb.stem + "_pbr.glb")
    maps = await blender_run.run(settings, "texture_maps.py", [glb, final, bump, roughness, metallic],
                                 "TEXTURE_MAPS_RESULT", final)
    glb.unlink(missing_ok=True)  # the intermediate (base color and AO only); the _pbr GLB supersedes it
    result["views"] = [p for p in result["paths"] if p.endswith(".png")]
    result["paths"] = [str(final), maps["fbx"]] + result["views"]
    result["textured_mesh"] = str(final)
    result["fbx"] = maps["fbx"]
    result["maps"] = maps
    return result


@mcp.tool()
async def remove_background(
    images: list[str],
    trim: bool = False,
    padding: float = 0.05,
    square: bool = False,
    size: int | None = None,
    background: str | None = None,
    output_dir: str | None = None,
    idle_timeout_seconds: int | None = None,
    ctx: Context | None = None,
) -> dict:
    """Cut the subject out of one or more images: PNG cut-outs with a transparent background (BiRefNet).

    images: 1 to 64 local image paths (PNG, JPEG, WebP, GIF, BMP or TIFF; the first frame of an
    animation). All are processed in one batch, so the model loads once; pass them together.
    With no other option each output is the same size as its input, the subject opaque and the
    background transparent, so it lines up with the original for compositing.
    trim=True crops to the subject's bounding box, plus padding on every side (a fraction of the
    subject's longer side, default 0.05; 0 is a tight crop). padding applies only with trim.
    square=True centres the result on a square transparent canvas (for icons and sprites).
    size: scale so the longer side is this many pixels (16 to 4096); with square, size x size.
    background: a colour like "#ffffff" to flatten onto instead of transparency (RGB PNG).
    Typical calls: UI icon trim=True, square=True, size=256; item sprite trim=True, padding=0;
    cut-out for compositing: no options.
    Returns per image the input, the output path, its size and the subject box found (x0, y0, x1,
    y1 in input pixels, with trim). A second, model free pass does the layout when any option is
    set. Default output folder: /ai/output/localgen/cutout/<date>/.
    """
    logger.info("remove_background: %d image(s) trim=%s square=%s size=%s", len(images or []), trim, square, size)
    return await cutout.remove_background(settings, images, trim, padding, square, size, background, output_dir,
                                          idle_timeout_seconds, TIMEOUTS["image"], _progress(ctx))


# -- speech ---------------------------------------------------------------


@mcp.tool()
async def speak(
    text: str,
    voice: str | None = None,
    speed: float = 1.0,
    effect: str = "none",
    intensity: float = 0.5,
    seed: int | None = None,
    name: str | None = None,
    output_dir: str | None = None,
    keep_dry: bool = True,
) -> dict:
    """Speak a line of text (English) to a WAV file, with an optional game voice effect. Runs on the CPU.

    Model: Kokoro-82M (Apache 2.0) in its own venv, offline; ComfyUI and the GPU are not used, so it
    never waits on image or video jobs. About 10 s to load plus about 1 s per sentence.
    voice: an installed voice name (see list_voices; default "af_heart"), or a comma separated blend
    of up to 4 voices with the same accent, averaged ("am_michael,bm_george" is invalid: two accents;
    "am_michael,am_fenrir" is fine). Prefix a = American, b = British; f = female, m = male.
    speed: 0.5 to 2.0 (1.0 normal).
    effect: "none"; "radio" (band limited, compressed, light hiss, like a comms channel); "robotic"
    (flat buzzing monotone with a metallic echo); "glitch" (a voice struggling on low power: stutters
    on word onsets, dropouts, bursts of bit crushing). intensity 0 to 1 (default 0.5) scales the effect.
    seed makes the glitch pattern and noise reproducible (returned in the result either way).
    name: file name stem for the line (for example "pathfinder_hull_40_lowpower"); an existing file
    with that name is overwritten. Default: <voice>_<effect>_<random>.
    keep_dry: with an effect, also keep the unprocessed take as <name>_dry.wav.
    Output: 24 kHz mono 16 bit WAV, plus an OGG Vorbis copy when ffmpeg is installed, in output_dir
    (default /ai/output/localgen/audio/<date>/). Returns the paths, duration, sample rate, RMS and peak
    level, word timings and the seed. Text up to 2000 characters; one call per line of dialogue.
    """
    logger.info("speak: %r voice=%s effect=%s", text[:80], voice, effect)
    return await speech.speak(settings, text, voice, speed, effect, output_dir, intensity, seed, name, keep_dry)


@mcp.tool()
async def list_voices() -> dict:
    """Text to speech voices installed for speak(): name, accent, gender and Kokoro's quality grade
    (A best, F worst; higher grades had more and cleaner training audio), the effects available,
    and whether ffmpeg (for effects and OGG output) was found."""
    return await asyncio.to_thread(speech.voices_report, settings)

# -- upscaling ------------------------------------------------------------

UPSCALE_MAX_SIDE = 8192  # longest side of the result, in pixels


def _upscale_input(input_image: str, scale: int) -> tuple[Path, int, int, bool]:
    """Validate an upscale input; returns its path, size and whether it is a PNG with alpha."""
    if scale not in (2, 4):
        raise ValueError("scale must be 2 or 4")
    src = Path(input_image).expanduser().resolve()
    if not src.is_file():
        raise ValueError(f"input image not found: {src}")
    if not jobs.is_image_file(src):  # checked by content: only real images are uploaded
        raise ValueError(f"not an image file: {src}")
    try:
        width, height = jobs.image_size(src)
    except jobs.JobError:
        raise ValueError(f"input_image must be a PNG or JPEG: {src}") from None
    if min(width, height) < 16:
        raise ValueError(f"input image is {width}x{height}; it must be at least 16 px on each side")
    if max(width, height) * scale > UPSCALE_MAX_SIDE:
        raise ValueError(f"{width}x{height} at {scale}x would be {width * scale}x{height * scale}; the result "
                         f"is limited to {UPSCALE_MAX_SIDE} px per side (use scale 2, or crop the image first)")
    return src, width, height, jobs.png_has_alpha(src)


@mcp.tool()
async def upscale_image(
    input_image: str,
    scale: int = 4,
    model: str | None = None,
    keep_alpha: bool = True,
    output_dir: str | None = None,
    idle_timeout_seconds: int | None = None,
    ctx: Context | None = None,
) -> dict:
    """Upscale an image (PNG or JPEG) 2x or 4x with a super resolution model, adding plausible detail.

    model: "realesrgan-x4plus" (default; general purpose, suits concept art, renders and photos, and
    cleans up mild noise and JPEG artefacts) or "pbrify-v4" (made for game textures such as albedo,
    roughness or normal maps; keeps fine surface detail; about 3 times slower).
    scale: 4 (the models' native scale) or 2 (the 4x result downscaled by half with Lanczos).
    keep_alpha: a PNG with transparency keeps it; the alpha channel is upscaled smoothly, the colour
    by the model. keep_alpha=False writes an opaque RGB PNG.
    Large images are processed in tiles inside the backend, so any size fits in VRAM; the result is
    limited to 8192 px per side (for example 2048 px at 4x, 4096 px at 2x) and is an 8-bit PNG.
    Measured on this GPU: about 0.5 s per megapixel of 4x output with the default model and 1.5 s
    with pbrify-v4 (a 1024 px image takes about 9 s and 25 s), plus any wait for the shared queue.
    Output: <output_dir or /ai/output/localgen/upscale/<date>>/<input name>_x<scale>.png.
    Returns the path, input and output sizes, whether alpha was kept, and timings.
    """
    src, width, height, alpha = await asyncio.to_thread(_upscale_input, input_image, scale)
    wf = workflows.pick("upscale", model, {"image"})
    keep = keep_alpha and alpha
    logger.info("upscale_image: %s %dx%d x%d alpha=%s (%s)", src, width, height, scale, keep, wf.name)
    result = await jobs.run_workflow(settings, wf, [{"scale": scale, "alpha": keep}], output_dir,
                                     idle_timeout_seconds, images={"image": str(src)},
                                     timeout=TIMEOUTS["image"], progress=_progress(ctx))
    out = Path(result["paths"][0])
    final = out.with_name(f"{src.stem}_x{scale}.png")
    n = 1
    while final.exists():
        n += 1
        final = out.with_name(f"{src.stem}_x{scale}_{n}.png")
    out.rename(final)
    result["paths"] = [str(final)]
    result["outputs"][0]["paths"] = [str(final)]
    result["upscaled_image"] = str(final)
    result["input_size"] = [width, height]
    result["output_size"] = [width * scale, height * scale]
    result["alpha_kept"] = keep
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
