"""generate_material: the seamless graph's wiring (its roll is executed in numpy with ComfyUI's
node semantics), the seam metric, wrap-around map maths, the PNG writer, and (when Blender
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


def _run(g: dict, link: list, inputs: dict) -> np.ndarray:
    """Evaluate the IMAGE graph behind link with numpy, as ComfyUI's ImageCrop and
    ImageCompositeMasked (no mask) do. inputs maps node ids to arrays."""
    nid, _ = link
    if nid in inputs:
        return inputs[nid]
    node = g[nid]
    a = node["inputs"]
    if node["class_type"] == "ImageCrop":
        img = _run(g, a["image"], inputs)
        return img[a["y"]:a["y"] + a["height"], a["x"]:a["x"] + a["width"]]
    if node["class_type"] == "ImageCompositeMasked":
        assert "mask" not in a
        dest = _run(g, a["destination"], inputs).copy()
        src = _run(g, a["source"], inputs)
        h = min(src.shape[0], dest.shape[0] - a["y"])
        w = min(src.shape[1], dest.shape[1] - a["x"])
        dest[a["y"]:a["y"] + h, a["x"]:a["x"] + w] = src[:h, :w]
        return dest
    raise AssertionError(f"unexpected node {node['class_type']}")


# -- graph wiring ---------------------------------------------------------


def test_roll_is_a_half_offset_with_wrap_around():
    n = 256
    img = _rough_image(n)
    g: dict = {}
    out = sm.add_roll(g, "r_", ["src", 0], n)
    rolled = _run(g, out, {"src": img})
    assert np.array_equal(rolled, np.roll(img, (n // 2, n // 2), axis=(0, 1)))
    # The input does not wrap; the rolled image does, and its old edges meet in the centre.
    before, after = pbr_maps.seam_metrics(img), pbr_maps.seam_metrics(rolled)
    assert not before["ok"]
    assert after["ok"] and not after["centre_ok"]


def test_text_graph_rolls_then_repaints_only_the_seam_band():
    p = _params()
    g = workflows.get("material").build(p)
    assert g["z_text"]["inputs"]["text"].startswith("grey rock, seamless tileable texture")
    assert g["base_sample"]["inputs"]["denoise"] == 1 and g["base_latent"]["inputs"]["width"] == 1024
    # base -> roll -> encode -> noise mask -> sampler (partial denoise) -> decode -> masked composite -> save
    assert g["roll_crop0"]["inputs"]["image"] == ["base_dec", 0]
    assert g["pass_enc"]["inputs"]["pixels"] == ["roll_paste3", 0]
    assert g["pass_noise_mask"]["inputs"] == {"samples": ["pass_enc", 0], "mask": ["seam_mask", 0]}
    assert g["pass_sample"]["inputs"]["latent_image"] == ["pass_noise_mask", 0]
    assert g["pass_sample"]["inputs"]["denoise"] == 0.7
    assert g["pass_sample"]["inputs"]["seed"] == 8
    comp = g["pass_out"]["inputs"]
    assert comp["destination"] == ["roll_paste3", 0] and comp["source"] == ["pass_dec", 0]
    assert comp["mask"] == ["seam_mask", 0]
    assert g["save"]["inputs"]["images"] == ["pass_out", 0]
    assert g["save_untiled"]["inputs"]["images"] == ["base_dec", 0]


def _upscale_bilinear(a: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize as torch's interpolate (align_corners=False) does."""
    n = a.shape[0]
    src = np.clip((np.arange(size) + 0.5) * n / size - 0.5, 0, n - 1)
    rows = np.array([np.interp(src, np.arange(n), r) for r in a])
    return np.array([np.interp(src, np.arange(n), c) for c in rows.T]).T


def _mask(g: dict, link: list, rng) -> np.ndarray:
    """Evaluate the seam mask graph with numpy, following ComfyUI's mask node semantics."""
    node = g[link[0]]
    a, kind = node["inputs"], node["class_type"]
    if kind == "SolidMask":
        return np.full((a["height"], a["width"]), a["value"])
    if kind == "MaskComposite":
        out = _mask(g, a["destination"], rng).copy()
        src = _mask(g, a["source"], rng)
        x, y = a["x"], a["y"]
        h, w = min(src.shape[0], out.shape[0] - y), min(src.shape[1], out.shape[1] - x)
        region = out[y:y + h, x:x + w]
        out[y:y + h, x:x + w] = region * src[:h, :w] if a["operation"] == "multiply" else region + src[:h, :w]
        return np.clip(out, 0, 1)
    if kind in ("MaskToImage", "ImageToMask"):
        return _mask(g, a["mask" if kind == "MaskToImage" else "image"], rng)
    if kind == "ImageAddNoise":
        img = _mask(g, a["image"], rng)
        return np.clip(img + a["strength"] * rng.standard_normal(img.shape), 0, 1)
    if kind == "ImageScale":
        assert a["upscale_method"] == "bilinear"
        return _upscale_bilinear(_mask(g, a["image"], rng), a["width"])
    raise AssertionError(f"unexpected node {kind}")


@pytest.mark.parametrize("size, width", [(1024, 0.25), (2048, 0.25), (512, 0.125)])
def test_seam_mask_covers_the_old_edges_and_leaves_the_new_ones(size, width):
    g: dict = {}
    link = sm.add_seam_mask(g, "m_", size, width, seed=3)
    for trial in range(3):
        m = _mask(g, link, np.random.default_rng(trial))
        assert m.shape == (size, size)
        c = size // 2
        # Fully repainted over the old edges (the centre lines, a few pixels either side) ...
        assert m[:, c - 4:c + 4].min() == 1.0 and m[c - 4:c + 4, :].min() == 1.0
        # ... untouched at the new edges and in the corners, so the wrap cannot change.
        edge = int(size * (0.5 - width / 2)) - size // 32  # one mask cell outside the band
        for corner in (m[:edge, :edge], m[:edge, -edge:], m[-edge:, :edge], m[-edge:, -edge:]):
            assert corner.max() == 0.0
        assert m[0, :edge].max() == 0.0 and m[:edge, 0].max() == 0.0
        # The band's edge is ragged: the ramp is not the same along every row.
        ramp = m[edge + size // 32: c - 8, c - size // 16]
        assert ramp.size and 0 < ramp.mean() < 1


def test_seam_layers():
    assert sm.seam_layers(0.25) == [8, 6, 4, 2]
    assert sm.seam_layers(0.05) == [4, 2]


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
