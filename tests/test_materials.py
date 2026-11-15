"""generate_material: the seamless graph (executed in numpy with ComfyUI's node semantics and
a stand-in for the model), the seam metric, wrap-around map maths, the PNG writer, and (when Blender
is installed) a real run of the map script."""

from __future__ import annotations

import asyncio
import json
import shutil
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np
import pytest

from localgen_mcp import blender_run, materials, workflows
from localgen_mcp.blender import pbr_maps
from localgen_mcp.workflows import seamless_material as sm

BLENDER = shutil.which("blender")


def _params(**over):
    wf = workflows.get("material")
    p = wf.params({"prompt": "grey rock", "seed": 7, **over})
    p["prefix"] = "localgen/material/x"
    return p


def _rough_image(n=256, seed=0):
    """A texture that does not tile: smooth noise on top of a left to right and top to bottom ramp."""
    rng = np.random.default_rng(seed)
    noise = pbr_maps.blur_wrap(rng.random((n, n)), 2.0)
    y, x = np.mgrid[0:n, 0:n] / n
    base = 0.2 + 0.5 * x + 0.2 * y + 0.6 * (noise - noise.mean())
    return np.clip(np.stack([base, base * 0.8, base * 0.6], -1), 0, 1)


def _upscale_bilinear(a: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize as torch's interpolate (align_corners=False) does."""
    n = a.shape[0]
    src = np.clip((np.arange(size) + 0.5) * n / size - 0.5, 0, n - 1)
    rows = np.array([np.interp(src, np.arange(n), r) for r in a])
    return np.array([np.interp(src, np.arange(n), c) for c in rows.T]).T


class _Comfy:
    """Evaluates a graph with numpy, following the semantics of ComfyUI's ImageCrop,
    ImageCompositeMasked, mask and noise nodes. Images are (H, W, C), masks (H, W). The model is
    replaced: every VAEDecode returns `painted`, a stand-in for whatever the sampler draws, so the
    test sees exactly which pixels a repaint can reach."""

    def __init__(self, g: dict, inputs: dict, painted: np.ndarray, seed: int = 0):
        self.g, self.inputs, self.painted = g, inputs, painted
        self.rng = np.random.default_rng(seed)
        self.cache: dict = {}

    def __call__(self, link: list) -> np.ndarray:
        nid = link[0]
        if nid in self.inputs:
            return self.inputs[nid]
        if nid not in self.cache:
            self.cache[nid] = self._eval(self.g[nid])
        return self.cache[nid]

    def _eval(self, node: dict) -> np.ndarray:
        a, kind = node["inputs"], node["class_type"]
        if kind == "ImageCrop":
            return self(a["image"])[a["y"]:a["y"] + a["height"], a["x"]:a["x"] + a["width"]]
        if kind == "ImageCompositeMasked":
            dest, src = self(a["destination"]).copy(), self(a["source"])
            x, y = a["x"], a["y"]
            h, w = min(src.shape[0], dest.shape[0] - y), min(src.shape[1], dest.shape[1] - x)
            m = self(a["mask"])[:h, :w, None] if "mask" in a else 1.0
            dest[y:y + h, x:x + w] = m * src[:h, :w] + (1 - m) * dest[y:y + h, x:x + w]
            return dest
        if kind == "SolidMask":
            return np.full((a["height"], a["width"]), a["value"])
        if kind == "MaskComposite":
            out, src = self(a["destination"]).copy(), self(a["source"])
            x, y = a["x"], a["y"]
            h, w = min(src.shape[0], out.shape[0] - y), min(src.shape[1], out.shape[1] - x)
            region = out[y:y + h, x:x + w]
            out[y:y + h, x:x + w] = region * src[:h, :w] if a["operation"] == "multiply" else region + src[:h, :w]
            return np.clip(out, 0, 1)
        if kind in ("MaskToImage", "ImageToMask"):
            return self(a["mask"] if kind == "MaskToImage" else a["image"])
        if kind == "ImageAddNoise":
            img = self(a["image"])
            return np.clip(img + a["strength"] * self.rng.standard_normal(img.shape), 0, 1)
        if kind == "ImageScale":
            assert a["upscale_method"] == "bilinear"
            return _upscale_bilinear(self(a["image"]), a["width"])
        if kind == "VAEDecode":
            return self.painted
        raise AssertionError(f"unexpected node {kind}")


# -- graph wiring ---------------------------------------------------------


@pytest.mark.parametrize("dx, dy", [(128, 128), (64, 64), (192, 64)])
def test_roll_is_a_wrap_around_offset(dx, dy):
    n = 256
    img = _rough_image(n)
    g: dict = {}
    out = sm.add_roll(g, "r_", ["src", 0], n, dx, dy)
    assert np.array_equal(_Comfy(g, {"src": img}, img)(out), np.roll(img, (dy, dx), axis=(0, 1)))


def test_half_roll_moves_the_seam_to_the_centre():
    img = _rough_image(256)
    rolled = np.roll(img, (128, 128), axis=(0, 1))
    before, after = pbr_maps.seam_metrics(img), pbr_maps.seam_metrics(rolled)
    assert not before["ok"]
    assert after["ok"] and not after["centre_ok"]


@pytest.mark.parametrize("size, width", [(512, 0.25), (1024, 0.25), (1024, 0.125), (2048, 0.3), (512, 0.05)])
def test_repaint_hides_the_whole_seam(size, width):
    """With a repaint that draws a flat colour, the old edges (now the centre lines) must be fully
    painted over, and most of the image must keep its pixels."""
    img = _rough_image(size)
    g = workflows.get("material").build(_params(size=size, seam_width=width))
    rolled = np.roll(img, (size // 2, size // 2), axis=(0, 1))
    for seed in range(2):
        out = _Comfy(g, {"base_dec": img}, np.full_like(img, 0.123), seed)(g["save"]["inputs"]["images"])
        c = size // 2
        assert np.allclose(out[:, c - 2:c + 2], 0.123) and np.allclose(out[c - 2:c + 2, :], 0.123)
        kept = np.isclose(out, rolled).all(axis=-1).mean()
        assert 0.3 < kept < 0.97


@pytest.mark.parametrize("size, width", [(512, 0.25), (1024, 0.125), (512, 0.05)])
def test_repaint_never_paints_across_an_edge(size, width):
    """With a repaint that draws an image that does not tile (a stand-in for anything the model
    might draw), the result must still wrap: no pass may repaint up to the edge of the frame it
    paints in, or the two sides of that edge would be painted independently."""
    img = _rough_image(size)
    g = workflows.get("material").build(_params(size=size, seam_width=width))
    for seed in range(2):
        painted = _rough_image(size, seed=10 + seed)[::-1, ::-1]
        out = _Comfy(g, {"base_dec": img}, painted, seed)(g["save"]["inputs"]["images"])
        m = pbr_maps.seam_metrics(out)
        assert m["ok"] and m["x"]["ratio"] < 2 and m["y"]["ratio"] < 2
        # Where the edges were not repainted, they are the original's pixels, side by side again.
        rolled = np.roll(img, (size // 2, size // 2), axis=(0, 1))
        assert np.isclose(out[0], rolled[0]).all(axis=-1).mean() > 0.6


def test_text_graph_wiring():
    p = _params()
    g = workflows.get("material").build(p)
    assert g["z_text"]["inputs"]["text"].startswith("grey rock, seamless tileable texture")
    assert g["base_sample"]["inputs"]["denoise"] == 1 and g["base_latent"]["inputs"]["width"] == 1024
    assert g["roll_crop0"]["inputs"]["image"] == ["base_dec", 0]
    # Each pass: encode -> noise mask -> sampler (partial denoise) -> decode -> masked composite.
    for ns, mask in (("pass_", "seam_mask"), ("pass2_", "spot_mask")):
        assert g[ns + "noise_mask"]["inputs"] == {"samples": [ns + "enc", 0], "mask": [mask, 0]}
        assert g[ns + "sample"]["inputs"]["latent_image"] == [ns + "noise_mask", 0]
        assert g[ns + "sample"]["inputs"]["denoise"] == 0.7
        comp = g[ns + "out"]["inputs"]
        assert comp["destination"] == g[ns + "enc"]["inputs"]["pixels"] and comp["source"] == [ns + "dec", 0]
        assert comp["mask"] == [mask, 0]
    assert g["pass_enc"]["inputs"]["pixels"] == ["roll_paste3", 0]
    assert g["pass2_enc"]["inputs"]["pixels"] == ["quarter_paste3", 0]
    assert g["save"]["inputs"]["images"] == ["back_paste3", 0]
    assert g["save_untiled"]["inputs"]["images"] == ["base_dec", 0]


def test_mask_layers():
    layers = sm.cross_layers(0.25)
    assert [r[0][2] for r in layers] == [16, 14, 12, 10, 8, 6, 4, 2]
    assert min(r[0][1] for r in layers) == 1  # one clear cell at the edges
    assert sm.cross_gap(0.25) == 4
    spots = sm.patch_layers(0.25)
    for rects in spots:
        for x, y, w, h in rects:
            assert x >= 1 and y >= 1 and x + w <= sm.MASK_CELLS - 1 and y + h <= sm.MASK_CELLS - 1


def test_reference_uses_klein_and_source_can_skip_the_roll():
    g = workflows.get("material").build(_params(images={"reference": "localgen/ref.png"}))
    assert g["reference"]["inputs"]["image"] == "localgen/ref.png"
    assert g["roll_crop0"]["inputs"]["image"] == ["base_decode", 0]
    assert "Turn image 1 into a flat surface texture of grey rock" in g["base_text"]["inputs"]["text"]
    g = workflows.get("material").build(_params(images={"source": "localgen/b.png"}, roll=False, save_untiled=False))
    assert "roll_crop0" not in g and "save_untiled" not in g and "k_unet" not in g
    assert g["pass_enc"]["inputs"]["pixels"] == ["base", 0]


def test_size_must_suit_the_latent_grid():
    with pytest.raises(ValueError):
        workflows.get("material").build(_params(size=1000))
    with pytest.raises(ValueError):
        workflows.get("material").params({"size": 4096})


def test_tool_arguments_are_checked():
    ok = dict(size=1024, normal_strength=1.0, roughness=0.5, roughness_variation=0.2, ao_strength=0.5,
              seam_passes=2, seam_denoise=0.8, seam_width=0.125, preview="sphere")
    materials.validate(**ok)
    for bad in ({"size": 1000}, {"roughness": 1.5}, {"seam_passes": 0}, {"seam_denoise": 0.1},
                {"preview": "cube"}, {"seam_width": 0.5}):
        with pytest.raises(ValueError):
            materials.validate(**{**ok, **bad})
    assert materials.slug("Rusted hull plating, with rivets!") == "rusted_hull_plating_with"


# -- seam metric and map maths -------------------------------------------


def test_seam_metric_tells_tiling_from_not():
    img = _rough_image()
    m = pbr_maps.seam_metrics(img)
    assert not m["ok"] and m["x"]["ratio"] > 3 and m["x"]["wrap_percentile"] == 100.0
    periodic = pbr_maps.blur_wrap(np.random.default_rng(3).random((256, 256)), 2.0)
    m = pbr_maps.seam_metrics(periodic)
    assert m["ok"] and m["centre_ok"] and 0.7 < m["x"]["ratio"] < 1.3


def _periodic_height(n=128):
    y, x = np.mgrid[0:n, 0:n] / n
    return 0.5 + 0.25 * np.sin(2 * np.pi * 3 * x + 0.7) + 0.2 * np.cos(2 * np.pi * 2 * y + 0.4)


def test_derivatives_wrap_around():
    n = 128
    h = _periodic_height(n)
    dx, drow = pbr_maps.gradients_wrap(h)
    # At the edges the wrapped central difference matches the analytic slope, as everywhere else
    # (a one-sided difference at the border, as np.gradient takes, does not).
    t = np.arange(n) / n
    analytic_dx = 0.25 * np.cos(2 * np.pi * 3 * t + 0.7) * 2 * np.pi * 3 / n
    analytic_drow = -0.2 * np.sin(2 * np.pi * 2 * t + 0.4) * 2 * np.pi * 2 / n
    assert np.allclose(dx[5], analytic_dx, atol=3e-4)
    assert np.allclose(drow[:, 7], analytic_drow, atol=3e-4)
    assert abs(np.gradient(h, axis=1)[5, 0] - analytic_dx[0]) > 1e-3


def test_normal_map_of_a_tiling_height_tiles(monkeypatch):
    h = pbr_maps.normalise(pbr_maps.blur_wrap(np.random.default_rng(5).random((128, 128)), 1.5))
    assert pbr_maps.seam_metrics(pbr_maps.normal_from_height(h, 4.0))["ok"]
    # The metric catches border derivatives that do not wrap.
    monkeypatch.setattr(pbr_maps, "gradients_wrap", lambda a: (np.gradient(a, axis=1), np.gradient(a, axis=0)))
    assert not pbr_maps.seam_metrics(pbr_maps.normal_from_height(h, 4.0))["ok"]


def test_normal_convention_opengl_and_directx():
    n = 64
    y, x = np.mgrid[0:n, 0:n] - n / 2
    bump = np.exp(-(x ** 2 + y ** 2) / 50.0)  # a mound in the middle
    gl = pbr_maps.normal_from_height(bump, 2.0)
    dxn = pbr_maps.normal_from_height(bump, 2.0, directx=True)
    above, right = gl[n // 2 - 4, n // 2], gl[n // 2, n // 2 + 4]
    assert above[1] > 0.55  # OpenGL: the slope facing the top of the image is green up
    assert right[0] > 0.55  # the slope facing right is red up
    assert np.allclose(dxn[..., 1], 1.0 - gl[..., 1]) and np.allclose(dxn[..., [0, 2]], gl[..., [0, 2]])
    assert np.allclose(np.linalg.norm(gl * 2 - 1, axis=-1), 1.0)


def test_maps_from_a_tiling_image_tile():
    rgb = np.stack([pbr_maps.blur_wrap(np.random.default_rng(i).random((128, 128)), 2.0) for i in range(3)], -1)
    rgb = pbr_maps.normalise(rgb)
    h = pbr_maps.height_from_albedo(rgb)
    for m in (h, pbr_maps.normal_from_height(h), pbr_maps.ambient_occlusion(h),
              pbr_maps.roughness_map(h, rgb, 0.5, 0.3)):
        assert pbr_maps.seam_metrics(m)["ok"]
    ao = pbr_maps.ambient_occlusion(h, 0.6)
    assert ao.min() >= 0.4 - 1e-6 and ao.max() <= 1.0
    assert (pbr_maps.ambient_occlusion(h)[h < np.percentile(h, 10)].mean()
            < pbr_maps.ambient_occlusion(h)[h > np.percentile(h, 90)].mean())


def _decode_png(data: bytes) -> np.ndarray:
    """Decode the filter-0 PNGs png_bytes writes."""
    w, h, depth, colour = struct.unpack(">IIBB", data[16:26])
    channels = {0: 1, 2: 3, 6: 4}[colour]
    pos, idat = 8, b""
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        if kind == b"IDAT":
            idat += data[pos + 8:pos + 8 + length]
        pos += 12 + length
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, -1)
    assert (raw[:, 0] == 0).all()
    rows = raw[:, 1:].copy()
    a = rows.view(">u2").astype(np.uint16) if depth == 16 else rows
    return a.reshape(h, w, channels).squeeze()


def test_png_writer_round_trips():
    rgb = (np.random.default_rng(1).random((5, 7, 3)) * 255).astype(np.uint8)
    assert np.array_equal(_decode_png(pbr_maps.png_bytes(rgb)), rgb)
    grey16 = (np.random.default_rng(2).random((4, 9)) * 65535).astype(np.uint16)
    assert np.array_equal(_decode_png(pbr_maps.png_bytes(grey16)), grey16)


# -- Blender --------------------------------------------------------------


@pytest.mark.skipif(BLENDER is None, reason="Blender is not installed")
def test_blender_maps_keep_bytes_orientation_and_tiling(tmp_path):
    img = pbr_maps.to_uint8(np.roll(_rough_image(128), (64, 64), axis=(0, 1)))
    img[8:12, 8:24] = [255, 0, 0]  # a red mark near the top left shows the orientation
    base = tmp_path / "base.png"
    pbr_maps.write_png(str(base), img)
    spec = {"mode": "maps", "base": str(base), "prefix": str(tmp_path / "m"), "preview": "none",
            "directx": True}
    cmd = blender_run.command(type("S", (), {"blender": BLENDER, "blender_threads": 2})(), "material_maps.py",
                              [json.dumps(spec)])
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300).stdout
    report = blender_run.parse_report(out, materials.MARKER)
    tiled = _decode_png(Path(report["files"]["tile_3x3"]).read_bytes())
    assert np.array_equal(tiled, np.tile(img, (3, 3, 1)))  # raw bytes, top row first
    assert report["files"]["normal"].endswith("_normal_dx.png")
    assert _decode_png(Path(report["files"]["height"]).read_bytes()).dtype == np.uint16
    assert report["seams"]["base_color"]["ok"]
    assert all(report["seams"][k]["ok"] for k in ("height", "normal", "roughness", "ambient_occlusion"))


# -- orchestration --------------------------------------------------------


def _seams(centre_ratio: float, ok: bool = True) -> dict:
    axis = {"ok": ok, "centre_ratio": centre_ratio, "centre_ok": centre_ratio < 2}
    return {"x": dict(axis), "y": dict(axis), "ok": ok, "centre_ok": centre_ratio < 2}


def test_seam_repaint_repeats_and_keeps_only_an_improvement(tmp_path, monkeypatch):
    settings = type("S", (), {"output_dir": tmp_path})()
    calls = []

    async def fake_run(settings, wf, jobs_, outdir, idle, images=None, timeout=0, progress=None):
        calls.append((jobs_[0], images))
        n = len(calls)
        names = [f"r{n}_00001_.png"] + ([f"r{n}_untiled_00001_.png"] if jobs_[0]["save_untiled"] else [])
        paths = []
        for name in names:
            (Path(outdir) / name).write_text(f"image {n}")
            paths.append(str(Path(outdir) / name))
        return {"paths": paths, "workflow": wf.name, "total_seconds": 1.0}

    # pass 1: centre line 5x (not ok); pass 2: worse (dropped); pass 3: better (kept)
    centre = iter([5.0, 7.0, 1.5])

    async def fake_blender(settings, spec, expect):
        if spec["mode"] == "check":
            seams = {"base_color": _seams(next(centre))}
            if "untiled" in spec:
                seams["untiled"] = _seams(9.0, ok=False)
            return {"seams": seams}
        return {"files": {"normal": "n.png"}, "normal_convention": "OpenGL", "seconds": 1.0,
                "seams": {"base_color": _seams(1.0), "normal": _seams(1.0)}, "stats": {}}

    monkeypatch.setattr(materials.jobs, "run_workflow", fake_run)
    monkeypatch.setattr(materials, "_blender", fake_blender)
    r = asyncio.run(materials.generate(settings, "rock", seed=10, seam_passes=3, output_dir=str(tmp_path)))
    assert [c[0].get("roll", True) for c in calls] == [True, False, False]
    assert [c[1] for c in calls][1:] == [{"source": r["files"]["base_color"]}] * 2
    assert [p["kept"] for p in r["seam_passes"]] == [True, False, True]
    assert [p["denoise"] for p in r["seam_passes"]] == [0.7, 0.8, 0.9]
    assert Path(r["files"]["base_color"]).read_text() == "image 3"
    assert r["tiles"] and not r["seams"]["untiled_base_color"]["ok"]
    assert sorted(p.name for p in Path(r["output_dir"]).iterdir()) == ["rock_10_basecolor.png"]
