from __future__ import annotations

import asyncio
import struct
import zlib
from pathlib import Path

import pytest

from localgen_mcp import cutout, workflows
from localgen_mcp.workflows.remove_background import FINISH, SLOTS, WORKFLOW, layout


def _params(wf, **over):
    p = wf.params(over)
    p.setdefault("seed", 1)
    p["prefix"] = "localgen/cutout/abc"
    return p


def _src(g, ref):
    return g[ref[0]]


def _check_graph(g: dict) -> None:
    for nid, node in g.items():
        assert node.get("class_type"), nid
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in g, f"node {nid} links to missing node {v[0]}"


def _of(g, cls):
    return [k for k, n in g.items() if n["class_type"] == cls]


def test_registered_as_default_cutout_workflow():
    assert workflows.get("cutout").name == WORKFLOW.name
    assert WORKFLOW.models == ["background_removal/birefnet.safetensors"]


def test_cut_mask_chain_gives_foreground_alpha():
    g = WORKFLOW.build(_params(WORKFLOW, images={SLOTS[0]: "localgen/a.png", SLOTS[1]: "localgen/b.png"}))
    _check_graph(g)
    joins = _of(g, "JoinImageWithAlpha")
    assert len(joins) == 2
    for j in joins:
        join = g[j]["inputs"]
        # JoinImageWithAlpha computes alpha = 1 - input, so its input must be the background:
        # InvertMask of RemoveBackground's foreground, never the foreground itself.
        inv = _src(g, join["alpha"])
        assert inv["class_type"] == "InvertMask"
        fg = _src(g, inv["inputs"]["mask"])
        assert fg["class_type"] == "RemoveBackground"
        # The colour and the mask come from the same (first) frame of the same image.
        assert fg["inputs"]["image"] == join["image"]
        frame = _src(g, join["image"])
        assert frame["class_type"] == "ImageFromBatch" and frame["inputs"]["length"] == 1
        assert _src(g, frame["inputs"]["image"])["class_type"] == "LoadImage"
        saves = [n for n in g.values() if n["class_type"] == "SaveImage" and n["inputs"]["images"] == [j, 0]]
        assert len(saves) == 1


def test_cut_batches_every_image_with_one_model_load():
    names = {s: f"localgen/{s}.png" for s in SLOTS[:5]}
    g = WORKFLOW.build(_params(WORKFLOW, images=names))
    _check_graph(g)
    assert len(_of(g, "LoadBackgroundRemovalModel")) == 1
    assert len(_of(g, "RemoveBackground")) == 5
    assert sorted(g[k]["inputs"]["image"] for k in _of(g, "LoadImage")) == sorted(names.values())
    prefixes = [g[k]["inputs"]["filename_prefix"] for k in _of(g, "SaveImage")]
    assert prefixes == [f"localgen/cutout/abc_{i:02d}c" for i in range(5)]
    # Each RemoveBackground reads its own image, never a neighbour's.
    for i in range(5):
        fg = g[f"fg{i:02d}"]["inputs"]["image"]
        load = g[fg[0]]["inputs"]["image"][0]
        assert g[load]["inputs"]["image"] == names[SLOTS[i]]


def test_cut_trim_saves_a_small_foreground_mask():
    g = WORKFLOW.build(_params(WORKFLOW, images={SLOTS[3]: "a.png"}, trim=True))
    _check_graph(g)
    save = g["msave03"]["inputs"]
    assert save["filename_prefix"].endswith("_03m")
    small = _src(g, save["images"])
    assert small["class_type"] == "ImageScaleToMaxDimension" and small["inputs"]["largest_size"] == 512
    as_image = _src(g, small["inputs"]["image"])
    # The box is read from the foreground (subject = white), not the inverted mask.
    assert as_image["class_type"] == "MaskToImage"
    assert _src(g, as_image["inputs"]["mask"])["class_type"] == "RemoveBackground"
    assert "MaskToImage" not in [n["class_type"] for n in WORKFLOW.build(_params(WORKFLOW, images={SLOTS[0]: "a.png"})).values()]


def test_layout():
    # No options: the whole image, unchanged.
    assert layout(800, 600, None, False, 0.05, False, None) == {
        "x": 0, "y": 0, "w": 800, "h": 600, "canvas_w": 800, "canvas_h": 600, "ox": 0, "oy": 0,
        "out_w": 800, "out_h": 600}
    # Trim with 10% padding of the longer side (200 px box side -> 20 px).
    L = layout(800, 600, (100, 50, 300, 150), True, 0.1, False, None)
    assert (L["x"], L["y"], L["w"], L["h"]) == (100, 50, 200, 100)
    assert (L["canvas_w"], L["canvas_h"], L["ox"], L["oy"]) == (240, 140, 20, 20)
    # Square centres the subject; size scales the canvas.
    L = layout(800, 600, (100, 50, 300, 150), True, 0.1, True, 120)
    assert (L["canvas_w"], L["canvas_h"], L["ox"], L["oy"]) == (240, 240, 20, 70)
    assert (L["out_w"], L["out_h"]) == (120, 120)
    # Without trim the box is ignored and padding does not apply; square pads the full image.
    L = layout(800, 600, (100, 50, 300, 150), False, 0.5, True, None)
    assert (L["w"], L["h"], L["canvas_w"], L["canvas_h"], L["ox"], L["oy"]) == (800, 600, 800, 800, 0, 100)
    L = layout(800, 600, None, False, 0.05, False, 400)
    assert (L["out_w"], L["out_h"]) == (400, 300)


def _finish(**over):
    lay = layout(800, 600, (100, 50, 300, 150), True, 0.1, True, None)
    lay.update(over.pop("lay", {}))
    return FINISH.build(_params(FINISH, images={SLOTS[0]: "localgen/cut.png"}, layout={SLOTS[0]: lay}, **over))


def test_finish_keeps_the_cutout_alpha():
    g = _finish()
    _check_graph(g)
    rgba = g["rgba00"]["inputs"]
    # LoadImage's mask is 1 - alpha, which is what JoinImageWithAlpha inverts back: no InvertMask here.
    assert rgba["alpha"] == ["load00", 1] and rgba["image"] == ["load00", 0]
    assert "InvertMask" not in [n["class_type"] for n in g.values()]
    crop = g["crop00"]["inputs"]
    assert (crop["image"], crop["x"], crop["y"], crop["width"], crop["height"]) == (["rgba00", 0], 100, 50, 200, 100)
    # The canvas is transparent: JoinImageWithAlpha of a solid 1 mask gives alpha 0.
    canvas = g["canvas00"]["inputs"]
    assert g[canvas["alpha"][0]]["class_type"] == "SolidMask" and g[canvas["alpha"][0]]["inputs"]["value"] == 1.0
    place = g["place00"]["inputs"]
    assert place["destination"] == ["canvas00", 0] and place["source"] == ["crop00", 0]
    assert (place["x"], place["y"], g["blank00"]["inputs"]["width"], g["blank00"]["inputs"]["height"]) == (20, 70, 240, 240)
    assert "mask" not in place  # a full paste carries the crop's alpha channel onto the canvas
    assert "ImageScale" not in [n["class_type"] for n in g.values()]
    assert g["save00"]["inputs"]["images"] == ["place00", 0]
    assert g["save00"]["inputs"]["filename_prefix"].endswith("_00f")


def test_finish_resize_and_background():
    g = _finish(lay={"out_w": 128, "out_h": 128}, background="#FF8000")
    _check_graph(g)
    scale = g["scale00"]["inputs"]
    assert scale["image"] == ["place00", 0] and (scale["width"], scale["height"]) == (128, 128)
    flat = g["flat00"]["inputs"]
    fill = _src(g, flat["destination"])["inputs"]
    assert fill["color"] == 0xFF8000 and (fill["width"], fill["height"]) == (128, 128)
    assert flat["source"] == ["scale00", 0]
    alpha = _src(g, flat["mask"])
    assert alpha["class_type"] == "ImageToMask" and alpha["inputs"] == {"image": ["scale00", 0], "channel": "alpha"}
    assert g["save00"]["inputs"]["images"] == ["flat00", 0]


# -- PNG reading (the server has no imaging library) ------------------------


def _paeth(a, b, c):
    pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
    return a if pa <= pb and pa <= pc else b if pb <= pc else c


def _png(path: Path, rows: list[list[tuple]], ctype: int = 2, filters=(0, 1, 2, 3, 4)) -> Path:
    """Encode rows of pixels, cycling through every PNG filter type."""
    bpp = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
    h, w = len(rows), len(rows[0])
    raw, prev = bytearray(), bytes(w * bpp)
    for y, row in enumerate(rows):
        line = bytes(v for px in row for v in px)
        f = filters[y % len(filters)]
        out = bytearray()
        for x, v in enumerate(line):
            a = line[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            pred = (0, a, b, (a + b) >> 1, _paeth(a, b, c))[f]
            out.append((v - pred) & 255)
        raw += bytes([f]) + out
        prev = line

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b""))
    return path


def test_png_channel_decodes_every_filter(tmp_path):
    import random
    rnd = random.Random(3)
    rows = [[(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(17)] for _ in range(11)]
    w, h, got = cutout.png_channel(_png(tmp_path / "a.png", rows))
    assert (w, h) == (17, 11)
    assert [list(r) for r in got] == [[px[0] for px in row] for row in rows]
    rgba = [[(v, 0, 0, 255) for v in range(20)] for _ in range(5)]
    assert list(cutout.png_channel(_png(tmp_path / "b.png", rgba, ctype=6))[2][4]) == list(range(20))
    bad = tmp_path / "c.png"
    bad.write_bytes(b"GIF89a" + b"\0" * 20)
    with pytest.raises(ValueError):
        cutout.png_channel(bad)


def test_subject_box_maps_preview_to_full_size_outwards(tmp_path):
    # 10x8 preview of a 1000x800 image; subject pixels at x 2..5, y 3..4 (one faint pixel below the threshold).
    rows = [[(0, 0, 0)] * 10 for _ in range(8)]
    for y in (3, 4):
        for x in range(2, 6):
            rows[y][x] = (255, 255, 255)
    rows[7][9] = (cutout.BOX_THRESHOLD, 0, 0)
    mask = _png(tmp_path / "m.png", rows)
    assert cutout.subject_box(mask, 1000, 800) == (200, 300, 600, 500)
    assert cutout.subject_box(mask, 999, 799) == (199, 299, 600, 500)  # rounded outwards
    empty = _png(tmp_path / "e.png", [[(0, 0, 0)] * 4] * 4)
    assert cutout.subject_box(empty, 100, 100) is None


# -- input validation ---------------------------------------------------------


def test_validation(tmp_path):
    png = _png(tmp_path / "a.png", [[(1, 2, 3)]])
    secret = tmp_path / "id_rsa.png"  # an image extension does not make it an image
    secret.write_bytes(b"-----BEGIN OPENSSH PRIVATE KEY-----\n")
    assert cutout.validate([str(png)], 0.05, None, None) == [png.resolve()]
    assert cutout.validate(str(png), 0.05, None, None) == [png.resolve()]  # a single path is accepted
    with pytest.raises(ValueError, match="not a PNG"):
        cutout.validate([str(png), str(secret)], 0.05, None, None)
    with pytest.raises(ValueError, match="not found"):
        cutout.validate([str(tmp_path / "missing.png")], 0.05, None, None)
    with pytest.raises(ValueError, match="1 to 64"):
        cutout.validate([], 0.05, None, None)
    with pytest.raises(ValueError, match="1 to 64"):
        cutout.validate([str(png)] * 65, 0.05, None, None)
    with pytest.raises(ValueError, match="padding"):
        cutout.validate([str(png)], 1.5, None, None)
    with pytest.raises(ValueError, match="size"):
        cutout.validate([str(png)], 0.05, 8, None)
    with pytest.raises(ValueError, match="colour"):
        cutout.validate([str(png)], 0.05, None, "red")
    cutout.validate([str(png)], 0.05, 256, "#00ff00")


# -- orchestration, with run_workflow replaced by a stand-in that writes files like ComfyUI ------


def _rgba_png(path: Path, w: int, h: int) -> str:
    _png(path, [[(0, 0, 0, 0)] * w for _ in range(h)], ctype=6, filters=(0,))
    return str(path)


def _fake_runs(monkeypatch, tmp_path):
    calls = []

    async def fake(settings, workflow, jobs_list, output_dir, idle, images=None, timeout=0, progress=None, mesh=None):
        calls.append({"workflow": workflow, "jobs": jobs_list, "images": dict(images), "output_dir": output_dir})
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for slot in images:
            i = SLOTS.index(slot)
            if workflow is WORKFLOW:
                paths.append(_rgba_png(out / f"abc_{i:02d}c_00001_.png", 40, 20))
                if jobs_list[0].get("trim"):
                    rows = [[(255, 255, 255) if 4 <= x < 8 and 2 <= y < 6 else (0, 0, 0) for x in range(20)]
                            for y in range(10)]
                    paths.append(str(_png(out / f"abc_{i:02d}m_00001_.png", rows)))
            else:
                paths.append(_rgba_png(out / f"abc_{i:02d}f_00001_.png", 2, 2))
        return {"paths": paths, "total_seconds": 1.0, "gpu_memory_used_peak_mb": 5000, "backend_managed": True}

    monkeypatch.setattr(cutout.jobs, "run_workflow", fake)
    return calls


def _settings(tmp_path):
    class S:
        root = tmp_path / "root"
        output_dir = tmp_path / "out"
    return S()


def test_no_options_is_one_pass(monkeypatch, tmp_path):
    calls = _fake_runs(monkeypatch, tmp_path)
    a, b = (_png(tmp_path / n, [[(1, 2, 3)]]) for n in ("a.png", "b.png"))
    res = asyncio.run(cutout.remove_background(_settings(tmp_path), [str(a), str(b)], output_dir=str(tmp_path / "o")))
    assert len(calls) == 1 and calls[0]["images"] == {SLOTS[0]: str(a.resolve()), SLOTS[1]: str(b.resolve())}
    assert [o["input"] for o in res["outputs"]] == [str(a.resolve()), str(b.resolve())]
    assert [Path(p).name for p in res["paths"]] == ["abc_00c_00001_.png", "abc_01c_00001_.png"]
    assert res["outputs"][0]["size"] == [40, 20]


def test_trim_runs_the_layout_pass_with_the_box(monkeypatch, tmp_path):
    calls = _fake_runs(monkeypatch, tmp_path)
    a, b = (_png(tmp_path / n, [[(1, 2, 3)]]) for n in ("a.png", "b.png"))
    res = asyncio.run(cutout.remove_background(_settings(tmp_path), [str(a), str(b)], trim=True, padding=0.0,
                                               square=True, size=64, output_dir=str(tmp_path / "o")))
    cut, fin = calls
    assert cut["workflow"] is WORKFLOW and cut["jobs"] == [{"trim": True}]
    assert fin["workflow"] is FINISH and len(fin["jobs"]) == 1  # one prompt for the whole batch
    # Pass 2 uploads pass 1's own cut-outs, each in its own slot.
    assert [Path(p).name for p in fin["images"].values()] == ["abc_00c_00001_.png", "abc_01c_00001_.png"]
    # Preview 20x10 of a 40x20 cut-out, subject at x 4..7, y 2..5 -> 8..16, 4..12 in full pixels.
    lay = fin["jobs"][0]["layout"][SLOTS[0]]
    assert (lay["x"], lay["y"], lay["w"], lay["h"]) == (8, 4, 8, 8)
    assert (lay["out_w"], lay["out_h"]) == (64, 64)
    assert res["outputs"][1]["subject_box"] == [8, 4, 16, 12]
    assert [Path(p).name for p in res["paths"]] == ["abc_00f_00001_.png", "abc_01f_00001_.png"]
    assert not any((tmp_path / "root" / "tmp").glob("cutout-*"))  # intermediate files removed
