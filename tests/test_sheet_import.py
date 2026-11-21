"""Sprite sheet import (import_sprite_sheet), pose guide sheets (render_pose_guides) and GIF cycles.

Every sheet here is synthetic: the test figure from test_sprites drawn onto a flat background at
uneven spacing, on a drifting baseline, in one or several rows.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pose_fixtures as pf
from localgen_mcp import config, pixel_ops as po, sprites
from test_poses import library_with
from test_sprites import FakeGPU, character, colours
from test_sprites import save as save_rgba

BG = (214, 218, 222)
FIG = 160  # figure canvas side in the sheet


def figure(stride: int) -> np.ndarray:
    """The test figure at FIG px, uint8 RGBA, legs apart by stride (the pose changes per frame)."""
    img = np.clip(np.rint(character(stride=stride) * 255), 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(img, "RGBA").resize((FIG, FIG), Image.NEAREST))


def sheet(path: Path, layout: list[list[tuple[int, int]]], width: int = 1200, height: int | None = None,
          stride=lambda i: 6 * (i % 4), shadow: bool = False, transparent: bool = False, extra=None) -> str:
    """layout: per row, (x, y) of each figure's top left. Returns the PNG path."""
    height = height or 40 + 200 * len(layout)
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0) if transparent else BG + (255,))
    i = 0
    for row in layout:
        for x, y in row:
            fig = Image.fromarray(figure(stride(i)), "RGBA")
            if shadow:  # a flat grey ground shadow under the feet, wider than the figure
                feet = y + int(460 / 512 * FIG)
                d = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
                d.paste((170, 170, 170, 255), (x + 20, feet - 2, x + FIG - 20, feet + 5))
                canvas.alpha_composite(d)
            canvas.alpha_composite(fig, (x, y))
            # A raised tool at head height, longer frame by frame, attached to the body: it moves the
            # frame's box and centre of mass but not its torso, so only a torso anchor stays put.
            canvas.paste((70, 70, 80, 255), (x + 100, y + 35, x + 106 + 12 * (i % 4), y + 45))
            i += 1
    if extra:
        extra(canvas)
    canvas.save(path)
    return str(path)


def run(tmp_path: Path, src: str, **spec) -> dict:
    spec = {"input": src, "output_prefix": str(tmp_path / "cells" / "s"), **spec}
    (tmp_path / "cells").mkdir(exist_ok=True)
    return po.op_import_sheet(spec)


def alpha(path: str) -> np.ndarray:
    return np.asarray(Image.open(path))[..., 3] > 0


def check_aligned(res: dict) -> None:
    """Every cell: the same size, lowest opaque row on the ground line, torso anchor on the anchor
    column within a pixel, transparent corners."""
    sides = set()
    for c in res["cells"]:
        a = alpha(c)
        sides.add(a.shape)
        assert np.flatnonzero(a.any(1))[-1] == res["ground_line"] - 1
        assert abs(po.torso_anchor(a) - res["anchor_column"]) <= 1
        assert not a[0, 0] and not a[-1, -1]
    assert len(sides) == 1


# -- finding and aligning frames -----------------------------------------------


def test_one_row_uneven_spacing_and_drifting_baseline(tmp_path):
    xs = [10, 190, 400, 560, 770, 1000]          # uneven gaps
    ys = [20, 26, 14, 31, 22, 9]                 # the baseline drifts by up to 22 px
    src = sheet(tmp_path / "row.png", [list(zip(xs, ys))])
    res = run(tmp_path, src, frames=6, rows=1)
    assert res["frames_found"] == 6 and res["count_ok"] and res["per_row"] == [6]
    assert res["background"] == po.to_hex(BG)
    check_aligned(res)
    # Frames are in sheet order: each frame's box starts right of the one before.
    assert [f["sheet_box"][0] for f in res["frames"]] == sorted(f["sheet_box"][0] for f in res["frames"])


def test_grid_is_read_left_to_right_then_top_to_bottom(tmp_path):
    rows = [[(30 + 280 * c, 20 + 3 * c) for c in range(4)],
            [(60 + 290 * c, 230 - 4 * c) for c in range(3)]]           # last row is short
    src = sheet(tmp_path / "grid.png", rows, stride=lambda i: 4 * i)   # wider stride frame by frame
    res = run(tmp_path, src, frames=7, rows=2, columns=4)
    assert res["count_ok"] and res["per_row"] == [4, 3] and res["rows_found"] == 2
    assert [f["row"] for f in res["frames"]] == [0, 0, 0, 0, 1, 1, 1]
    lefts = [f["sheet_box"][0] for f in res["frames"]]
    assert lefts[:4] == sorted(lefts[:4]) and lefts[4:] == sorted(lefts[4:])
    assert res["frames"][4]["sheet_box"][1] > res["frames"][3]["sheet_box"][3]   # row 2 starts below row 1
    check_aligned(res)


def test_wrong_count_fails_with_what_was_found_or_is_kept(tmp_path):
    src = sheet(tmp_path / "row.png", [[(20 + 190 * i, 20) for i in range(6)]])
    with pytest.raises(ValueError, match=r"found 6 frames but 5 were expected .*\[6\]"):
        run(tmp_path, src, frames=5, overlay=str(tmp_path / "ov.png"))
    assert (tmp_path / "ov.png").is_file()     # the boxes found are drawn even when it fails
    res = run(tmp_path, src, frames=5, on_mismatch="keep")
    assert not res["count_ok"] and res["frames_found"] == 6 and len(res["cells"]) == 6
    with pytest.raises(ValueError, match="rows"):
        run(tmp_path, src, rows=2)
    assert run(tmp_path, src)["frames_found"] == 6   # no count given: take what is found


def test_a_detached_piece_joins_its_frame_and_touching_frames_are_split(tmp_path):
    def tool_head(canvas):  # a small block floating left of frame 2 (subject from x 475), across a gap
        canvas.paste((60, 60, 70, 255), (450, 60, 462, 90))

    src = sheet(tmp_path / "frag.png", [[(20 + 200 * i, 20) for i in range(5)]], extra=tool_head)
    res = run(tmp_path, src, frames=5)
    assert res["count_ok"] and any("narrow piece" in n for n in res["notes"])
    assert res["frames"][2]["sheet_box"][0] == 450 and not any("split" in n for n in res["notes"])
    # Two frames whose subjects touch (each is 50 px wide from x + 55): one run twice as wide, split.
    xs = [20, 200, 250, 560, 740]
    src = sheet(tmp_path / "touch.png", [[(x, 20) for x in xs]], stride=lambda i: 0)
    res = run(tmp_path, src, frames=5)
    assert res["count_ok"] and any("split" in n for n in res["notes"])
    check_aligned(res)


def test_ground_shadow_is_cleared(tmp_path):
    src = sheet(tmp_path / "shadow.png", [[(20 + 200 * i, 20 + 5 * i) for i in range(4)]], shadow=True)
    res = run(tmp_path, src, frames=4)
    assert all(f["ground_shadow_pixels"] > 0 for f in res["frames"])
    check_aligned(res)
    # The shadow is wider than the feet: without clearing it, the cells are wider at the bottom.
    kept = run(tmp_path, src, frames=4, shadow=False, output_prefix=str(tmp_path / "cells" / "kept"))
    feet = [alpha(c)[res["ground_line"] - 1].sum() for c in res["cells"]]
    shadowed = [alpha(c)[kept["ground_line"] - 1].sum() for c in kept["cells"]]
    assert min(shadowed) > max(feet)


def test_transparent_sheet_and_kept_background(tmp_path):
    src = sheet(tmp_path / "clear.png", [[(20 + 210 * i, 20 + 7 * i) for i in range(3)]], transparent=True)
    res = run(tmp_path, src, frames=3)
    assert res["count_ok"]
    check_aligned(res)
    flat = sheet(tmp_path / "flat.png", [[(20 + 210 * i, 20) for i in range(3)]])
    keep = run(tmp_path, flat, frames=3, background="keep")   # for a BiRefNet cut-out afterwards
    arr = np.asarray(Image.open(keep["cells"][0]))
    assert (arr[..., 3] == 255).all() and tuple(arr[0, 0, :3]) == BG


def test_grid_op_fits_images_onto_a_canvas(tmp_path):
    paths = []
    for i in range(5):
        p = tmp_path / f"g{i}.png"
        Image.new("RGB", (100, 200), (10 * i, 0, 0)).save(p)
        paths.append(str(p))
    res = po.op_grid({"images": paths, "columns": 3, "canvas": [600, 400], "output": str(tmp_path / "grid.png")})
    assert (res["columns"], res["rows"]) == (3, 2) and Image.open(res["path"]).size == (600, 400)
    res = po.op_grid({"images": paths, "columns": 5, "output": str(tmp_path / "row.png")})
    assert Image.open(res["path"]).size == (500, 200)


# -- the tools -------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALGEN_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("LOCALGEN_POSE_DIR", str(tmp_path / "poses"))
    return config.load()


def gif_frames(path: str) -> int:
    return Image.open(path).n_frames


def test_import_sprite_sheet_tool(tmp_path, settings):
    rows = [[(30 + 280 * c, 20 + 5 * (c % 2)) for c in range(4)], [(40 + 280 * c, 240) for c in range(4)]]
    src = sheet(tmp_path / "walk.png", rows, stride=lambda i: 6 * (i % 4))
    res = asyncio.run(sprites.import_sprite_sheet(settings, src, frames=8, rows=2, columns=4, size=32,
                                                  palette="moriveil", action="walk", gif_loops=3,
                                                  output_dir=str(tmp_path / "out")))
    strip = np.asarray(Image.open(res["strip"]))
    assert strip.shape == (32, 32 * 8, 4)
    for i, f in enumerate(res["frames"]):
        assert (strip[:, 32 * i:32 * (i + 1)] == np.asarray(Image.open(f))).all()
    def rgb(cs):
        return {tuple(int(c[j:j + 2], 16) for j in (1, 3, 5)) for c in cs}

    shared = rgb(res["palette"])
    assert shared <= rgb(sprites.read_text_palette(sprites.PALETTE_DIR / "moriveil.gpl"))
    for f in res["frames"]:
        assert colours(np.asarray(Image.open(f))) <= shared
    # The pixelized frames share one ground line (fit="frame" on aligned canvases).
    bottoms = {np.flatnonzero((np.asarray(Image.open(f))[..., 3] > 0).any(1))[-1] for f in res["frames"]}
    assert len(bottoms) == 1
    data = json.loads(Path(res["json"]).read_text())
    assert data["frame_count"] == 8 and data["fps"] == 10 and data["loop"] and data["sheet"]["per_row"] == [4, 4]
    assert gif_frames(res["preview_gif"]) == 8 * 3
    assert Path(res["overlay"]).is_file() and not res["warnings"]
    assert res["consistency"]["static_frames"] == 0


def test_import_sprite_sheet_reports_a_wrong_count(tmp_path, settings):
    src = sheet(tmp_path / "row.png", [[(20 + 190 * i, 20) for i in range(6)]])
    with pytest.raises(sprites.PixelError, match="found 6 frames but 4 were expected.*Overlay"):
        asyncio.run(sprites.import_sprite_sheet(settings, src, frames=4, output_dir=str(tmp_path / "out")))
    res = asyncio.run(sprites.import_sprite_sheet(settings, src, frames=4, on_count_mismatch="keep", size=32,
                                                  output_dir=str(tmp_path / "out")))
    assert len(res["frames"]) == 6 and res["warnings"] and not res["sheet"]["count_ok"]
    with pytest.raises(ValueError, match="background"):
        asyncio.run(sprites.import_sprite_sheet(settings, src, background="magic"))
    with pytest.raises(ValueError, match="gif_loops"):
        asyncio.run(sprites.import_sprite_sheet(settings, src, gif_loops=0))


def test_import_with_birefnet_cuts_each_cell(tmp_path, settings, monkeypatch):
    gpu = FakeGPU(tmp_path, bad=set())
    cut_inputs = []

    async def fake_cut(settings_, paths, outdir, *a):
        cut_inputs.extend(paths)
        out = []
        for p in paths:  # stand in for BiRefNet: key the flat background the cell kept
            arr = np.asarray(Image.open(p).convert("RGBA")).copy()
            arr[..., 3] = np.where((np.abs(arr[..., :3].astype(int) - BG).sum(-1) > 45), 255, 0)
            dst = Path(outdir) / f"cut_{len(out)}_{Path(p).stem}.png"
            Image.fromarray(arr, "RGBA").save(dst)
            out.append(str(dst))
        return out

    monkeypatch.setattr(sprites, "cut", fake_cut)
    src = sheet(tmp_path / "row.png", [[(20 + 200 * i, 20) for i in range(3)]])
    res = asyncio.run(sprites.import_sprite_sheet(settings, src, frames=3, size=32, background="birefnet",
                                                  output_dir=str(tmp_path / "out")))
    assert len(cut_inputs) == 3 and all(np.asarray(Image.open(p))[..., 3].min() == 255 for p in cut_inputs)
    assert np.asarray(Image.open(res["frames"][0]))[0, 0, 3] == 0
    assert gpu.calls == []


def test_render_pose_guides_tool(tmp_path, settings):
    library_with(settings, pf.synthetic_clip(name="fixture_attack"))
    res = asyncio.run(sprites.render_pose_guides(settings, "fixture_attack", frames=8, size=256,
                                                 sheet_size=[1536, 1024], output_dir=str(tmp_path / "g")))
    assert Image.open(res["sheet"]).size == (1536, 1024)
    assert (res["columns"], res["rows"]) == (6, 2) and len(res["frames"]) == 8
    w, h = Image.open(res["frames"][0]).size
    assert abs(w / h - (1536 / 6) / (1024 / 2)) < 0.02   # each guide drawn at its cell's shape
    assert "2 rows of 6" in res["prompt_hint"] and len(res["durations_ms"]) == 8
    data = json.loads(Path(res["json"]).read_text())
    assert data["clip_frames"] == res["clip_frames"] and data["view"] == "right"
    one = asyncio.run(sprites.render_pose_guides(settings, "attack", frames=4, size=128, layout=4,
                                                 output_dir=str(tmp_path / "g")))
    assert Image.open(one["sheet"]).size == (512, 128) and "one row" in one["prompt_hint"]
    with pytest.raises(ValueError, match="sheet_size"):
        asyncio.run(sprites.render_pose_guides(settings, "attack", sheet_size=[10, 10]))


class StridingGPU(FakeGPU):
    """FakeGPU whose frames differ in pose (wider stride each frame), so GIF frames never merge."""

    async def draw(self, settings, prompts, seeds, outdir, idle, timeout, progress, refs=None, input_image=None):
        paths = []
        for prompt in prompts:
            frame = int(prompt.split("This is frame ")[1].split(" ")[0]) - 1 if "This is frame" in prompt else 0
            self.n += 1
            paths.append(save_rgba(character(stride=30 * frame), Path(outdir) / f"draw_{self.n}.png"))
        return {"paths": paths, "outputs": [{"paths": [p]} for p in paths], "workflow": "fake"}


def test_generate_animation_gif_holds_gif_loops_cycles(tmp_path, settings, monkeypatch):
    gpu = StridingGPU(tmp_path, bad=set())
    monkeypatch.setattr(sprites, "draw", gpu.draw)
    monkeypatch.setattr(sprites, "cut", gpu.cut)
    res = asyncio.run(sprites.generate_animation(settings, "a test hero", "walk", frames=3, size=32, seed=4,
                                                 retries=0, min_motion=0.0, gif_loops=3,
                                                 output_dir=str(tmp_path / "anim")))
    assert gif_frames(res["preview_gif"]) == 9
    one = asyncio.run(sprites.generate_animation(settings, "a test hero", "walk", frames=3, size=32, seed=4,
                                                 retries=0, min_motion=0.0, output_dir=str(tmp_path / "anim")))
    assert gif_frames(one["preview_gif"]) == 3
