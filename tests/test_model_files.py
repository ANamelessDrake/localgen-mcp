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


def test_renamed_glb_is_still_checked(tmp_path):
    p = tmp_path / "innocent.bin"
    p.write_bytes(_glb({"asset": ASSET, "buffers": [{"byteLength": 64, "uri": "/etc/hostname"}]}))
    with pytest.raises(ModelFileError, match="external"):
        require_self_contained(p)


def test_retopology_wants_a_glb_name_and_content(tmp_path):
    from localgen_mcp import retopo
    p = tmp_path / "mesh.bin"
    p.write_bytes(_glb({"asset": ASSET}))
    with pytest.raises(retopo.RetopoError, match="binary glTF"):
        retopo.validate(str(p), 10000, "quad", 2048)


def test_resume_state_cannot_point_outside_the_run(tmp_path):
    from localgen_mcp.server import _reuse_file
    run = tmp_path / "run"
    (run / "parts").mkdir(parents=True)
    good = run / "parts" / "a.glb"
    good.write_bytes(_glb({"asset": ASSET}))
    png = run / "a.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 32)
    outside = tmp_path / "secret.glb"
    outside.write_bytes(_glb({"asset": ASSET}))
    leaky = run / "parts" / "leaky.glb"
    leaky.write_bytes(_glb({"asset": ASSET, "buffers": [{"byteLength": 64, "uri": "/etc/hostname"}]}))
    notes = run / "notes.png"
    notes.write_text("not an image")
    root = run.resolve()
    assert _reuse_file(root, "parts/a.glb", "mesh") == str(good.resolve())
    assert _reuse_file(root, "a.png", "image") == str(png.resolve())
    assert _reuse_file(root, str(outside), "mesh") is None          # absolute path
    assert _reuse_file(root, "../secret.glb", "mesh") is None       # escapes the run folder
    assert _reuse_file(root, "parts/leaky.glb", "mesh") is None     # external buffer reference
    assert _reuse_file(root, "notes.png", "image") is None          # not really an image
    assert _reuse_file(root, {"x": 1}, "mesh") is None
