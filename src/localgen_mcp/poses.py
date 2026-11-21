"""Pose library: bake mocap clips once (Blender), list them, and find one for generate_animation.

Blender runs only in bake_pose_library: blender/bake_pose_clips.py imports each FBX or BVH, maps
its skeleton to the canonical humanoid joints and writes one compact JSON per clip (see
pose_math.py for the format). Everything after that (retargeting, the camera, drawing the guides)
is numpy in the pixel helper, so making a sprite sheet never starts Blender.

The library is a folder outside the repository, Settings.pose_library (LOCALGEN_POSE_DIR, default
/ai/data/localgen/poses), holding <clip>.json files and index.json, a manifest of every clip's
metadata. Like the rest of the server process, this module imports no numpy.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path

from . import blender_run
from .config import Settings

INDEX = "index.json"
LIBRARY_FORMAT = "localgen-pose-library"
SUFFIXES = (".fbx", ".bvh")
MAX_CLIPS = 500
ROOT_MOTION = ("in_place", "keep")
SUMMARY_KEYS = ("action", "loop", "frames", "fps", "impact_frame", "weapon_hand", "key_frames", "stature",
                "root_travel", "naming", "source_file")


class PoseLibraryError(ValueError):
    pass


def clip_name(text: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:64]
    if not name:
        raise PoseLibraryError("a clip name must contain letters or digits")
    return name


def collect_inputs(inputs: list[str] | str, recursive: bool = True) -> list[Path]:
    """FBX and BVH files from files and folders, in a stable order, without duplicates."""
    items = [inputs] if isinstance(inputs, str) else list(inputs or [])
    if not items:
        raise PoseLibraryError("give one or more FBX or BVH files or folders")
    found: list[Path] = []
    for raw in items:
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            walk = p.rglob("*") if recursive else p.glob("*")
            found.extend(sorted(f for f in walk if f.is_file() and f.suffix.lower() in SUFFIXES))
        elif p.is_file():
            if p.suffix.lower() not in SUFFIXES:
                raise PoseLibraryError(f"not an FBX or BVH file: {p}")
            found.append(p)
        else:
            raise PoseLibraryError(f"not found: {p}")
    out = list(dict.fromkeys(found))
    if not out:
        raise PoseLibraryError("no FBX or BVH files found")
    if len(out) > MAX_CLIPS:
        raise PoseLibraryError(f"{len(out)} files; bake at most {MAX_CLIPS} at a time")
    return out


def read_index(library: Path) -> dict:
    path = library / INDEX
    if not path.is_file():
        return {"format": LIBRARY_FORMAT, "clips": {}}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        raise PoseLibraryError(f"the pose library index is unreadable: {path}") from None
    if data.get("format") != LIBRARY_FORMAT or not isinstance(data.get("clips"), dict):
        raise PoseLibraryError(f"not a pose library index: {path}")
    return data


def write_index(library: Path, data: dict) -> None:
    tmp = library / f".{INDEX}.{uuid.uuid4().hex[:8]}"
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    os.replace(tmp, library / INDEX)


def bake_spec(files: list[Path], library: Path, *, name: str | None, action: str | None, loop: bool | None,
              weapon_hand: str | None, take: str | None, root_motion: str) -> list[dict]:
    if name is not None and len(files) != 1:
        raise PoseLibraryError("name can only be given when baking one file")
    if weapon_hand not in (None, "left", "right"):
        raise PoseLibraryError('weapon_hand must be "left" or "right"')
    if root_motion not in ROOT_MOTION:
        raise PoseLibraryError(f"root_motion must be one of {ROOT_MOTION}")
    if action is not None:
        action = clip_name(action)
    clips, used = [], set()
    for f in files:
        base = clip_name(name or f.stem)
        n, out = 1, base
        while out in used:
            n += 1
            out = f"{base}_{n}"
        used.add(out)
        clips.append({"input": str(f), "output": str(library / f"{out}.json"), "name": out, "action": action,
                      "loop": loop, "weapon_hand": weapon_hand, "take": take, "root_motion": root_motion})
    return clips


async def bake_pose_library(settings: Settings, inputs: list[str] | str, *, name: str | None = None,
                            action: str | None = None, loop: bool | None = None, weapon_hand: str | None = None,
                            take: str | None = None, root_motion: str = "in_place", recursive: bool = True) -> dict:
    files = collect_inputs(inputs, recursive)
    library = settings.pose_library
    library.mkdir(parents=True, exist_ok=True)
    clips = bake_spec(files, library, name=name, action=action, loop=loop, weapon_hand=weapon_hand, take=take,
                      root_motion=root_motion)
    work = settings.root / "tmp"
    work.mkdir(parents=True, exist_ok=True)
    report_path = work / f"pose-bake-{uuid.uuid4().hex[:10]}.json"
    t0 = time.time()
    try:
        report = await blender_run.run(settings, "bake_pose_clips.py",
                                       [json.dumps({"clips": clips, "report": str(report_path)})],
                                       "POSE_BAKE_RESULT", report_path)
    finally:
        report_path.unlink(missing_ok=True)
    index = read_index(library)
    baked, failed = [], []
    for r in report["clips"]:
        if not r.get("ok"):
            failed.append({"name": r["name"], "input": r.get("input"), "error": r.get("error")})
            continue
        entry = {k: r.get(k) for k in SUMMARY_KEYS}
        entry.update(file=Path(r["path"]).name, baked=time.strftime("%Y-%m-%dT%H:%M:%S"))
        index["clips"][r["name"]] = entry
        baked.append({"name": r["name"], "path": r["path"], **entry, "joint_bones": r.get("joint_bones")})
    write_index(library, index)
    return {"library": str(library), "index": str(library / INDEX), "baked": baked, "failed": failed,
            "clips_in_library": len(index["clips"]), "total_seconds": round(time.time() - t0, 2)}


def list_pose_clips(settings: Settings, action: str | None = None) -> dict:
    library = settings.pose_library
    index = read_index(library)
    clips = []
    for name, e in sorted(index["clips"].items()):
        if action and e.get("action") != action.lower().strip():
            continue
        clips.append({"name": name, **{k: e.get(k) for k in ("action", "loop", "frames", "fps", "impact_frame",
                                                              "weapon_hand", "source_file")},
                      "duration": round(e["frames"] / e["fps"], 3) if e.get("frames") and e.get("fps") else None,
                      "key_frames": e.get("key_frames")})
    actions = sorted({e.get("action") for e in index["clips"].values() if e.get("action")})
    return {"library": str(library), "clips": clips, "actions": actions, "count": len(clips)}


def clip_roots(settings: Settings) -> list[Path]:
    """Where a pose clip file may be read from: the library, the output folder and the install's tmp folder."""
    return [settings.pose_library.resolve(), settings.output_dir.resolve(), (settings.root / "tmp").resolve()]


def resolve_clip(settings: Settings, pose_clip: str) -> tuple[Path, str]:
    """(path, clip name) for a library clip name, an action tag (the first clip with that action,
    by name) or a clip .json file under clip_roots()."""
    text = str(pose_clip or "").strip()
    if not text:
        raise PoseLibraryError("pose_clip is empty")
    if text.endswith(".json") or "/" in text:
        path = Path(text).expanduser().resolve()
        if path.suffix.lower() != ".json":
            raise PoseLibraryError("a pose clip file must be a .json file")
        if not any(path.is_relative_to(r) for r in clip_roots(settings)):
            raise PoseLibraryError(f"a pose clip file must be under {settings.pose_library}, {settings.output_dir} "
                                   f"or {settings.root / 'tmp'}")
        if not path.is_file():
            raise PoseLibraryError(f"pose clip not found: {path}")
        return path, path.stem
    library = settings.pose_library
    clips = read_index(library)["clips"]
    key = clip_name(text)
    if key in clips:
        return library / clips[key]["file"], key
    by_action = sorted(n for n, e in clips.items() if e.get("action") == key)
    if by_action:
        return library / clips[by_action[0]]["file"], by_action[0]
    raise PoseLibraryError(f"no pose clip or action named {text!r} in {library}; "
                           "list_pose_clips shows what is there, bake_pose_library adds clips")
