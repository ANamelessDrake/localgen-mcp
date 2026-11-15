"""upscale_image: graph wiring, input validation, alpha detection and the pinned model downloads."""

from __future__ import annotations

import importlib.util
import struct
import sys
import zlib
from pathlib import Path

import pytest

from localgen_mcp import jobs, workflows

SETUP = Path(__file__).resolve().parent.parent / "setup"


def _downloader():
    spec = importlib.util.spec_from_file_location("download_models", SETUP / "download_models.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["download_models"] = mod  # dataclasses look the module up while building Entry
    spec.loader.exec_module(mod)
    return mod


def _png(path: Path, w: int, h: int, color_type: int = 2, trns: bool = False) -> Path:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    raw = b"".join(b"\0" + b"\0" * (w * channels) for _ in range(h))
    body = chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0))
    if color_type == 3:
        body += chunk(b"PLTE", b"\0\0\0")
    if trns:
        body += chunk(b"tRNS", b"\0")
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + body + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return path


def _params(wf, **over):
    p = wf.params(over)
    p.update(seed=1, prefix="localgen/upscale/x", images={"image": "localgen/in.png"})
    return p


def _check_graph(g: dict) -> None:
    for nid, node in g.items():
        assert node.get("class_type"), nid
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in g, f"node {nid} links to missing node {v[0]}"


def test_default_model_and_one_default():
    assert workflows.pick("upscale", slots={"image"}).name == "realesrgan-x4plus"
    assert sum(w.is_default for w in workflows.for_modality("upscale")) == 1
    with pytest.raises(ValueError):
        workflows.pick("upscale", slots=set())  # an input image is required


def test_4x_runs_the_model_and_saves_its_output():
    wf = workflows.get("upscale")
    g = wf.build(_params(wf, scale=4))
    _check_graph(g)
    assert g["model"]["inputs"]["model_name"] == "RealESRGAN_x4plus.pth"
    assert g["upscale"]["inputs"] == {"upscale_model": ["model", 0], "image": ["load", 0]}
    assert g["save"]["inputs"]["images"] == ["upscale", 0]
    assert {n["class_type"] for n in g.values()}.isdisjoint({"ImageScaleBy", "JoinImageWithAlpha"})


def test_2x_downscales_the_4x_result_with_lanczos():
    wf = workflows.get("upscale", "pbrify-v4")
    g = wf.build(_params(wf, scale=2))
    _check_graph(g)
    assert g["model"]["inputs"]["model_name"] == "4x-PBRify_UpscalerV4.safetensors"
    resize = g["resize"]
    assert resize["class_type"] == "ImageScaleBy"
    assert resize["inputs"] == {"image": ["upscale", 0], "upscale_method": "lanczos", "scale_by": 0.5}
    assert g["save"]["inputs"]["images"] == ["resize", 0]


def test_alpha_is_joined_before_the_model():
    wf = workflows.get("upscale")
    g = wf.build(_params(wf, scale=4, alpha=True))
    _check_graph(g)
    # LoadImage's mask output (1 - alpha) is what JoinImageWithAlpha expects; the model node then
    # upscales the RGB channels and resizes the alpha channel itself.
    assert g["rgba"] == {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["load", 0], "alpha": ["load", 1]}}
    assert g["upscale"]["inputs"]["image"] == ["rgba", 0]


def test_unsupported_scale_rejected():
    wf = workflows.get("upscale")
    with pytest.raises(ValueError):
        wf.build(_params(wf, scale=3))
    with pytest.raises(ValueError):
        wf.params({"scale": 8})


def test_png_alpha_detection(tmp_path):
    assert not jobs.png_has_alpha(_png(tmp_path / "rgb.png", 4, 4, 2))
    assert jobs.png_has_alpha(_png(tmp_path / "rgba.png", 4, 4, 6))
    assert jobs.png_has_alpha(_png(tmp_path / "grey_alpha.png", 4, 4, 4))
    assert jobs.png_has_alpha(_png(tmp_path / "palette_trns.png", 4, 4, 3, trns=True))
    assert not jobs.png_has_alpha(_png(tmp_path / "palette.png", 4, 4, 3))
    jpg = tmp_path / "a.jpg"
    jpg.write_bytes(b"\xff\xd8\xff\xe0" + b"\0" * 32)
    assert not jobs.png_has_alpha(jpg)


def test_upscale_input_validation(tmp_path):
    from localgen_mcp.server import _upscale_input
    ok = _png(tmp_path / "ok.png", 1024, 512, 6)
    src, w, h, alpha = _upscale_input(str(ok), 4)
    assert (src, w, h, alpha) == (ok.resolve(), 1024, 512, True)
    with pytest.raises(ValueError, match="scale"):
        _upscale_input(str(ok), 3)
    big = _png(tmp_path / "big.png", 4096, 16, 2)
    assert _upscale_input(str(big), 2)[1] == 4096  # 8192 px result is allowed
    with pytest.raises(ValueError, match="8192"):
        _upscale_input(str(big), 4)
    with pytest.raises(ValueError, match="16 px"):
        _upscale_input(str(_png(tmp_path / "tiny.png", 8, 64)), 4)
    secret = tmp_path / "key.png"  # an image extension does not make it an image
    secret.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    with pytest.raises(ValueError, match="not an image"):
        _upscale_input(str(secret), 4)
    gif = tmp_path / "a.gif"
    gif.write_bytes(b"GIF89a" + b"\0" * 32)
    with pytest.raises(ValueError, match="PNG or JPEG"):
        _upscale_input(str(gif), 4)
    with pytest.raises(ValueError, match="not found"):
        _upscale_input(str(tmp_path / "missing.png"), 4)


def test_manifest_pins_every_upscale_model():
    dm = _downloader()
    entries = {e.target: e for e in dm.read_manifest(dm.DEFAULT_MANIFEST)}
    for wf in workflows.for_modality("upscale"):
        for model in wf.models:
            e = entries[model]
            assert e.group == "upscale" and e.size > 0 and len(e.sha256) == 64
            assert e.is_github and e.url.startswith("https://github.com/")
            assert e.url.endswith(f"/releases/download/{e.revision}/{e.path}")
    rgan = entries["upscale_models/RealESRGAN_x4plus.pth"]
    assert rgan.url == "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    hf = next(e for e in entries.values() if e.group == "image")
    assert not hf.is_github and hf.url.startswith("https://huggingface.co/")


def test_manifest_rejects_bad_lines(tmp_path):
    dm = _downloader()
    good = "a" * 64
    for line in (f"upscale x/y.pth github:onlyowner v1 y.pth 10 {good}",  # owner/repo required
                 f"textures x/y.pth github:o/r v1 y.pth 10 {good}",  # unknown group
                 "upscale x/y.pth github:o/r v1 y.pth 10 abc"):  # not a sha256
        m = tmp_path / "m"
        m.write_text(line + "\n")
        with pytest.raises(SystemExit):
            dm.read_manifest(m)


def test_hf_token_is_not_sent_to_github(tmp_path, monkeypatch):
    dm = _downloader()
    sent = []

    class Stop(Exception):
        pass

    def fake_urlopen(req, timeout=None):
        sent.append((req.full_url, req.get_header("Authorization")))
        raise Stop

    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setattr(dm.urllib.request, "urlopen", fake_urlopen)
    gh = dm.Entry("upscale", "upscale_models/a.pth", "github:o/r", "v1", "a.pth", 1, "0" * 64)
    hf = dm.Entry("image", "vae/a.safetensors", "org/repo", "abc", "a.safetensors", 1, "0" * 64)
    for e in (gh, hf):
        with pytest.raises(Stop):
            dm.download(e, tmp_path / e.target)
    assert sent == [("https://github.com/o/r/releases/download/v1/a.pth", None),
                    ("https://huggingface.co/org/repo/resolve/abc/a.safetensors", "Bearer hf_secret")]
