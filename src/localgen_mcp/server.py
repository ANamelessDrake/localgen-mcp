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

from . import backend, blender_run, config, jobs, model_files, retopo, workflows
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
    face and triangle counts and uv_texel_error (how far its UVs sit from LOD0's, in texels; a few at
    most is right). With method "decimate", the detail map also steers the LOD decimations.
    """
    return await retopo.retopologize(settings, input_mesh, target_faces, method, texture_size, output_dir,
                                     detail, detail_strength, lods)


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
