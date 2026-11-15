"""Checks on 3D model files supplied by a caller.

A glTF file (and a GLB, for buffers beyond the first) can name other files by
URI: a buffer read as raw vertex data, or an image used as a texture. Imported
and re-exported, such a reference would carry an arbitrary local file into the
output. Only self-contained models are accepted: every buffer and image must be
embedded (a GLB's binary chunk or a data: URI).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path


class ModelFileError(ValueError):
    pass


def _gltf_json(path: Path) -> dict:
    data = path.read_bytes()
    if path.suffix.lower() == ".glb" or data[:4] == b"glTF":
        if len(data) < 20 or data[:4] != b"glTF":
            raise ModelFileError(f"not a binary glTF file: {path}")
        length, kind = struct.unpack_from("<II", data, 12)
        if kind != 0x4E4F534A:  # "JSON"
            raise ModelFileError(f"GLB without a JSON chunk: {path}")
        raw = data[20:20 + length]
    else:
        raw = data
    try:
        doc = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelFileError(f"unreadable glTF JSON in {path}: {exc}") from None
    if not isinstance(doc, dict) or "asset" not in doc:
        raise ModelFileError(f"not a glTF document: {path}")
    return doc


def external_references(path: Path) -> list[str]:
    """URIs in a .glb or .gltf that point outside the file (anything but data: URIs)."""
    doc = _gltf_json(path)
    refs = []
    for section in ("buffers", "images"):
        for item in doc.get(section, []) or []:
            uri = item.get("uri") if isinstance(item, dict) else None
            if isinstance(uri, str) and not uri.startswith("data:"):
                refs.append(f"{section}: {uri}")
    return refs


def is_gltf(path: Path) -> bool:
    """glTF by name or by content (a binary glTF signature), so a renamed file is still caught."""
    if path.suffix.lower() in (".glb", ".gltf"):
        return True
    with open(path, "rb") as f:
        return f.read(4) == b"glTF"


def input_glb(input_mesh: str) -> Path:
    """A caller's GLB path, resolved: it must exist, be named .glb, carry the binary glTF signature and
    embed every buffer and image."""
    path = Path(input_mesh).expanduser().resolve()
    if not path.is_file():
        raise ModelFileError(f"mesh not found: {path}")
    with open(path, "rb") as f:
        signature = f.read(4)
    if path.suffix.lower() != ".glb" or signature != b"glTF":
        raise ModelFileError(f"not a binary glTF (.glb) file: {path}")
    require_self_contained(path)
    return path


def require_self_contained(path: Path) -> None:
    """Raise unless a glTF/GLB embeds every buffer and image. Other formats are not checked here."""
    if is_gltf(path):
        refs = external_references(path)
        if refs:
            raise ModelFileError(f"{path} references external files {refs[:5]}; "
                                 "only self-contained models (everything embedded) are accepted")
