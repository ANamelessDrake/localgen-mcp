"""Caller-supplied model files must be self-contained: no glTF buffer or image URI may name another file."""

from __future__ import annotations

import json
import struct

import pytest

from localgen_mcp.model_files import ModelFileError, external_references, require_self_contained


def _glb(doc: dict, bin_chunk: bytes = b"") -> bytes:
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js
    if bin_chunk:
        bin_chunk += b"\0" * (-len(bin_chunk) % 4)
        body += struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk
    return b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body


ASSET = {"version": "2.0"}


def test_embedded_glb_is_accepted(tmp_path):
    p = tmp_path / "ok.glb"
    p.write_bytes(_glb({"asset": ASSET, "buffers": [{"byteLength": 4}],
                        "images": [{"bufferView": 0, "mimeType": "image/png"}]}, b"\0\0\0\0"))
    assert external_references(p) == []
    require_self_contained(p)


@pytest.mark.parametrize("doc", [
    {"asset": ASSET, "buffers": [{"byteLength": 64, "uri": "/home/someone/.ssh/id_ed25519"}]},
    {"asset": ASSET, "buffers": [{"byteLength": 64, "uri": "../../.aws/credentials"}]},
    {"asset": ASSET, "images": [{"uri": "file:///etc/shadow"}]},
])
def test_external_references_are_refused(tmp_path, doc):
    glb = tmp_path / "bad.glb"
    glb.write_bytes(_glb(doc))
    gltf = tmp_path / "bad.gltf"
    gltf.write_text(json.dumps(doc))
    for p in (glb, gltf):
        with pytest.raises(ModelFileError, match="external"):
            require_self_contained(p)


def test_data_uris_are_embedded(tmp_path):
    p = tmp_path / "inline.gltf"
    p.write_text(json.dumps({"asset": ASSET, "buffers": [{"byteLength": 4, "uri": "data:application/octet-stream;base64,AAAAAA=="}]}))
    require_self_contained(p)


def test_garbage_is_not_a_model(tmp_path):
    p = tmp_path / "x.gltf"
    p.write_text("password=hunter2")
    with pytest.raises(ModelFileError):
        require_self_contained(p)
