"""Pixel art sprite tools: the pixel pass, sheets, consistency metrics and the orchestration.

pixel_ops is imported directly here (numpy and Pillow); the MCP server only ever runs it as a
child process, which the orchestration tests do for real.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from localgen_mcp import config, pixel_ops as po, sprites

SRC = Path(__file__).resolve().parent.parent / "src"
PALETTE = sprites.PALETTE_DIR / "moriveil.gpl"
PAL = po.read_palette(sprites.read_text_palette(PALETTE))


# -- fixtures ----------------------------------------------------------------


def character(size: int = 512, shift: int = 0, stride: int = 0) -> np.ndarray:
    """A float RGBA test figure on a transparent background: head, body, legs, several colours and
    a shading gradient, standing on the bottom third."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    xx -= shift
    img = np.zeros((size, size, 4))
    s = size / 512
    head = np.hypot(xx - 256 * s, yy - 150 * s) < 60 * s
    body = (np.abs(xx - 256 * s) < 80 * s) & (yy > 200 * s) & (yy < 360 * s)
    # stride swings the legs apart (a pose change, unlike shift, which the bottom anchor undoes)
    legs = ((np.abs(xx - (220 - stride) * s) < 25 * s) | (np.abs(xx - (292 + stride) * s) < 25 * s)) \
        & (yy >= 360 * s) & (yy < 460 * s)
    shade = 0.75 + 0.25 * (xx / size)
    img[head, :3] = np.stack([0.85 * shade, 0.55 * shade, 0.40 * shade], -1)[head]
    img[body, :3] = np.stack([0.30 * shade, 0.45 * shade, 0.75 * shade], -1)[body]
    img[legs, :3] = [0.35, 0.22, 0.12]
    img[head | body | legs, 3] = 1.0
    return img


def colours(arr: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(int(v) for v in c) for c in arr[arr[..., 3] == 255][:, :3]}


def border(mask: np.ndarray) -> np.ndarray:
    """Opaque pixels with a transparent 4-neighbour (or on the cell edge)."""
    pad = np.pad(mask, 1)
    inner = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return mask & ~inner


def save(arr: np.ndarray, path: Path) -> str:
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr * 255), 0, 255).astype(np.uint8)
    Image.fromarray(arr, "RGBA").save(path)
    return str(path)


# -- pixelize ----------------------------------------------------------------


@pytest.mark.parametrize("size", [16, 32, 64, 100, 128])
def test_output_is_exactly_the_cell_size(size):
    out, rep = po.pixelize_array(character(), size)
    assert out.shape == (size, size, 4) and out.dtype == np.uint8
    assert rep["size"] == [size, size]


@pytest.mark.parametrize("max_colors", [32, 12, 5])
def test_colours_come_from_the_palette_and_never_exceed_it(max_colors):
    pal = PAL
    out, rep = po.pixelize_array(character(), 64, palette=pal, max_colors=max_colors)
    used = colours(out)
    assert used <= {tuple(int(v) for v in c) for c in pal}
    assert len(used) <= max_colors
    assert rep["colours_used"] == len(used)


@pytest.mark.parametrize("max_colors,steps", [(16, None), (8, None), (31, 3), (6, 2)])
def test_auto_palette_respects_max_colors(max_colors, steps):
    out, rep = po.pixelize_array(character(), 64, max_colors=max_colors, shading_steps=steps)
    assert len(colours(out)) <= max_colors
    assert len(rep["palette"]) <= max_colors


def test_shading_steps_build_tone_ramps():
    """With shading_steps the auto palette is hue families of that many tones each."""
    rgb = character()[..., :3][character()[..., 3] > 0] * 255
    pal = po.auto_palette(rgb, 9, 3)
    assert len(pal) <= 9
    lab = po.oklab(pal.astype(float))
    assert np.ptp(lab[:, 0]) > 0.1  # tones differ in lightness


def test_outline_ring_in_the_outline_colour():
    pal = PAL
    dark = tuple(int(v) for v in pal[0])  # the palette's darkest entry, listed first
    out, rep = po.pixelize_array(character(), 64, palette=pal, outline=True, outline_color="darkest")
    mask = out[..., 3] == 255
    edge = border(mask)
    assert edge.sum() > 40
    assert all(tuple(int(v) for v in out[y, x, :3]) == dark for y, x in zip(*np.nonzero(edge)))
    assert rep["outline_color"] == po.to_hex(dark)
    # The outline is a ring outside the silhouette: without it the same border pixels are not all dark.
    plain, _ = po.pixelize_array(character(), 64, palette=pal, outline=False)
    pedge = border(plain[..., 3] == 255)
    assert not all(tuple(int(v) for v in plain[y, x, :3]) == dark for y, x in zip(*np.nonzero(pedge)))
    assert mask.sum() > (plain[..., 3] == 255).sum()


def test_auto_outline_is_darker_than_what_it_borders():
    pal = PAL
    flat, body, _ = po.cell_colours(character(), 64)
    rgb = po.quantise(flat, body, pal, pal[0])
    out, mask = po.add_outline(rgb, body, pal, pal[0], "auto")
    ring = mask & ~body
    lab = po.oklab(out.astype(float))
    darker = []
    for y, x in zip(*np.nonzero(ring)):
        nb = [(y + dy, x + dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
              if 0 <= y + dy < 64 and 0 <= x + dx < 64 and body[y + dy, x + dx]]
        assert nb
        darker.append(lab[y, x, 0] < min(lab[p][0] for p in nb))
    assert len(darker) > 40 and all(darker)
    # Not one flat colour: the outline follows what it borders.
    assert len({tuple(out[y, x, :3]) for y, x in zip(*np.nonzero(ring))}) > 1


def test_fixed_outline_colour_keeps_its_slot():
    out, _ = po.pixelize_array(character(), 64, max_colors=8, outline_color="#000000")
    used = colours(out)
    assert (0, 0, 0) in used and len(used) <= 8


def test_transparent_background_is_kept():
    src = character()
    out, _ = po.pixelize_array(src, 64, outline=False, blur=0)
    alpha = out[..., 3]
    assert set(np.unique(alpha)) <= {0, 255}
    for corner in (alpha[0, 0], alpha[0, -1], alpha[-1, 0], alpha[-1, -1]):
        assert corner == 0
    assert (out[alpha == 0][:, :3] == 0).all()
    # The silhouette matches the source's, downsampled the same way (box of the subject on the cell).
    box = po.subject_box(src[..., 3])
    where = po.place(src, 64, "trim", "center", 0)
    ys = np.floor(where["y0"] + (np.arange(64) + 0.5) * where["sy"]).astype(int).clip(0, 511)
    xs = np.floor(where["x0"] + (np.arange(64) + 0.5) * where["sx"]).astype(int).clip(0, 511)
    ref = src[..., 3][np.ix_(ys, xs)] >= 0.5
    assert box is not None
    assert po.silhouette_iou(out, np.dstack([np.zeros((64, 64, 3)), ref * 255]).astype(np.uint8)) > 0.9


def test_opaque_fill_stays_opaque():
    img = character()
    img[..., 3] = 1.0
    out, _ = po.pixelize_array(img, 32, fit="fill", outline=False)
    assert (out[..., 3] == 255).all()


def test_wrap_keeps_a_seamless_image_seamless():
    """Rolling a periodic source by one cell pixel rolls the wrapped result by exactly one pixel; a
    clamped blur breaks that at the edges."""
    n = 1024
    yy, xx = np.mgrid[0:n, 0:n] / n
    img = np.ones((n, n, 4))
    img[..., 0] = 0.5 + 0.4 * np.sin(2 * np.pi * 3 * xx)
    img[..., 1] = 0.5 + 0.4 * np.cos(2 * np.pi * 2 * yy)
    img[..., 2] = 0.4
    pal = PAL
    a, _ = po.pixelize_array(img, 64, palette=pal, fit="fill", outline=False, wrap=True, blur=2)
    b, _ = po.pixelize_array(np.roll(img, 16, axis=1), 64, palette=pal, fit="fill", outline=False, wrap=True, blur=2)
    assert (np.roll(a, 1, axis=1) == b).all()
    c, _ = po.pixelize_array(img, 64, palette=pal, fit="fill", outline=False, wrap=False, blur=2)
    d, _ = po.pixelize_array(np.roll(img, 16, axis=1), 64, palette=pal, fit="fill", outline=False, wrap=False, blur=2)
    assert not (np.roll(c, 1, axis=1) == d).all()


def test_fixed_scale_keeps_size_between_frames():
    """Animation frames use the base sprite's scale: a smaller figure stays smaller."""
    base, rep = po.pixelize_array(character(), 64, anchor="bottom")
    small = np.zeros_like(character())
    small[256:, 128:384] = character(256)
    frame, frep = po.pixelize_array(small, 64, anchor="bottom", scale=rep["scale"])
    assert frep["scale"] == rep["scale"]
    assert (frame[..., 3] == 255).sum() < 0.4 * (base[..., 3] == 255).sum()
    # Feet on the bottom row (above the outline row) in both.
    assert (base[-2, :, 3] == 255).any() and (frame[-2, :, 3] == 255).any()


def test_stray_pixels_are_removed():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    mask[0, 7] = True       # alone
    mask[3, 3] = False      # a pinhole
    clean = po.cleanup(mask)
    assert not clean[0, 7] and clean[3, 3]


def test_despeckle_recolours_noise_but_keeps_details():
    rgb = np.zeros((7, 7, 3), dtype=np.uint8)
    rgb[...] = [124, 74, 42]          # wood
    rgb[2, 2] = [138, 79, 58]         # a near colour alone: noise
    rgb[4, 4] = [255, 244, 224]       # a bright highlight alone: a detail
    mask = np.ones((7, 7), dtype=bool)
    out = po.despeckle(rgb, mask)
    assert tuple(out[2, 2]) == (124, 74, 42)
    assert tuple(out[4, 4]) == (255, 244, 224)


def test_pixelize_drops_a_stray_speck():
    """A speck of the source that lands on one lone cell pixel (a leftover of the cut-out) is dropped."""
    img = character()
    img[40:48, 40:48] = [0.9, 0.9, 0.1, 1.0]
    out, _ = po.pixelize_array(img, 64, fit="frame", outline=False, blur=0)
    assert (out[2:8, 2:8, 3] == 0).all()
    assert (out[..., 3] == 255).sum() > 500


def test_bad_options_rejected():
    with pytest.raises(ValueError):
        po.pixelize_array(character(), 32, fit="cover")
    with pytest.raises(ValueError):
        po.pixelize_array(character(), 32, anchor="top")
    for kw in ({"size": 4}, {"max_colors": 1}, {"outline_color": "red"}, {"shading_steps": 0}, {"blur": 9},
               {"saturation": -1}):
        with pytest.raises(ValueError):
            sprites.check_pixel_options(**kw)


# -- palettes ----------------------------------------------------------------


def test_shipped_palette():
    hexes = sprites.read_text_palette(PALETTE)
    pal = po.read_palette(hexes)
    assert len(pal) == 32
    lab = po.oklab(pal.astype(float))
    assert lab[:, 0].argmin() == 0  # the darkest (outline) colour is listed first
    assert "moriveil" in sprites.shipped_palettes()


def test_palette_sources(tmp_path):
    hexes = ["#102030", "#a0b0c0", "#ff0000"]
    assert po.read_palette(hexes).tolist() == [[16, 32, 48], [160, 176, 192], [255, 0, 0]]
    txt = tmp_path / "p.hex"
    txt.write_text("; a comment\n102030\n#a0b0c0\n")
    assert sprites.read_text_palette(txt) == ["#102030", "#a0b0c0"]
    img = np.zeros((4, 4, 4), dtype=np.uint8)
    img[..., 3] = 255
    img[:2, :, :3] = [10, 20, 30]
    img[2:, :, :3] = [200, 100, 50]
    png = tmp_path / "p.png"
    Image.fromarray(img, "RGBA").save(png)
    assert len(po.read_palette(png)) == 2
    with pytest.raises(ValueError):
        po.read_palette(["#000000"])
    with pytest.raises(ValueError):
        po.read_palette(txt)  # the helper reads only lists and PNGs


def test_resolve_palette(tmp_path, settings):
    assert sprites.resolve_palette(settings, None) is None and sprites.resolve_palette(settings, "auto") is None
    assert sprites.resolve_palette(settings, "moriveil") == sprites.read_text_palette(PALETTE)
    assert sprites.resolve_palette(settings, ["a83232", "#000000"]) == ["#a83232", "#000000"]
    for bad in ("no_such_palette", ["#zzzzzz", "#000000"]):
        with pytest.raises(ValueError):
            sprites.resolve_palette(settings, bad)
    staged = settings.root / "tmp"
    staged.mkdir(parents=True, exist_ok=True)
    good = staged / "mine.gpl"
    good.write_text("GIMP Palette\nName: mine\nColumns: 2\n#\n 10 20 30 dark\n200 100 50\tlight\n")
    assert sprites.resolve_palette(settings, str(good)) == ["#0a141e", "#c86432"]
    outside = tmp_path / "elsewhere.gpl"
    outside.write_text(good.read_text())
    with pytest.raises(ValueError, match="must be under"):
        sprites.resolve_palette(settings, str(outside))
    wrong_suffix = staged / "blob.bin"
    wrong_suffix.write_bytes(bytes(range(256)) * 4)
    with pytest.raises(ValueError):
        sprites.resolve_palette(settings, str(wrong_suffix))
    fake_png = staged / "fake.png"
    fake_png.write_text("#000000\n#ffffff\n")
    with pytest.raises(ValueError, match="not a PNG"):
        sprites.resolve_palette(settings, str(fake_png))


SECRET = "-----BEGIN OPENSSH PRIVATE KEY----- b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ"


@pytest.mark.parametrize("suffix,body", [
    (".txt", f"#102030\n{SECRET}\n#ffffff\n"),
    (".gpl", f"GIMP Palette\nName: x\n 1 2 3\n{SECRET}\n"),
    (".hex", f"{SECRET}\n"),
    (".txt", "token=" + SECRET + "\n"),
])
def test_non_palette_file_rejected_without_echoing_it(settings, suffix, body):
    """A file that is not a palette is refused, and nothing of its contents reaches the error."""
    staged = settings.root / "tmp"
    staged.mkdir(parents=True, exist_ok=True)
    path = staged / f"notes{suffix}"
    path.write_text(body)
    with pytest.raises(ValueError) as info:
        sprites.resolve_palette(settings, str(path))
    msg = str(info.value)
    assert "line" in msg or "colours" in msg
    for fragment in (SECRET[:12], "OPENSSH", "b3BlbnNz", "token"):
        assert fragment not in msg


def test_secret_outside_the_allowed_folders_is_never_read(settings, tmp_path, monkeypatch):
    key = tmp_path / "id_ed25519.txt"
    key.write_text(SECRET + "\n")
    reads = []
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: reads.append(self) or real(self))
    with pytest.raises(ValueError) as info:
        sprites.resolve_palette(settings, str(key))
    assert key not in reads and "OPENSSH" not in str(info.value)


def test_helper_errors_never_echo_colour_text():
    with pytest.raises(ValueError) as info:
        po.read_palette(["#000000", SECRET])
    assert "OPENSSH" not in str(info.value) and "entry 2" in str(info.value)
    with pytest.raises(ValueError) as info:
        po.hex_colour(SECRET)
    assert "OPENSSH" not in str(info.value)


def test_helper_failure_details_stay_in_the_server_log(settings, tmp_path, caplog):
    """A helper crash returns a short error; its traceback goes to the server log only."""
    bogus = tmp_path / "broken.png"
    bogus.write_bytes(b"\x89PNG\r\n\x1a\n" + SECRET.encode())
    spec = {"op": "pixelize", "size": 16, "palette": None, "items": [{"input": str(bogus), "output": str(tmp_path / "o.png")}]}
    with caplog.at_level("ERROR", logger="localgen_mcp.sprites"):
        with pytest.raises(sprites.PixelError) as info:
            asyncio.run(sprites.run_ops(settings, spec, tmp_path / "o.png"))
    assert "server log" in str(info.value) and "Traceback" not in str(info.value)
    assert "Traceback" in caplog.text
    # A ValueError raised inside the helper comes back as its own short message.
    spec = {"op": "pack", "images": [], "sheet": str(tmp_path / "s.png"), "atlas": str(tmp_path / "s.json"),
            "contact": str(tmp_path / "c.png")}
    with pytest.raises(sprites.PixelError, match="no images to pack"):
        asyncio.run(sprites.run_ops(settings, spec, tmp_path / "s.png"))


# -- sheets ------------------------------------------------------------------


def _solid(tmp_path, i, w=64, h=64):
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[...] = [(37 * i) % 256, (91 * i) % 256, (53 * i + 20) % 256, 255]
    arr[0, 0] = [i, 255 - i, 7, 255]  # a marker pixel in the top left corner
    return save(arr, tmp_path / f"cell{i}.png")


@pytest.mark.parametrize("n,columns,expect", [(5, 3, (256, 128)), (4, None, (128, 128)), (6, 6, (512, 64)),
                                              (9, 4, (256, 256))])
def test_pack_atlas_matches_pixels_and_power_of_two(tmp_path, n, columns, expect):
    paths = [_solid(tmp_path, i) for i in range(n)]
    spec = {"op": "pack", "images": [{"name": f"c{i}", "path": p} for i, p in enumerate(paths)], "cell": [64, 64],
            "columns": columns, "power_of_two": True, "sheet": str(tmp_path / "s.png"),
            "atlas": str(tmp_path / "s.json"), "contact": str(tmp_path / "s_contact.png")}
    res = po.op_pack(spec)
    sheet = np.asarray(Image.open(res["sheet"]).convert("RGBA"))
    atlas = json.loads(Path(res["atlas"]).read_text())
    h, w = sheet.shape[:2]
    assert (w, h) == expect == (atlas["width"], atlas["height"])
    assert w & (w - 1) == 0 and h & (h - 1) == 0
    assert len(atlas["cells"]) == n
    for c, p in zip(atlas["cells"], paths):
        src = np.asarray(Image.open(p).convert("RGBA"))
        assert (sheet[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]] == src).all()
        assert c["unity_y"] == h - c["y"] - c["h"]
    # Unused space is transparent.
    covered = np.zeros((h, w), dtype=bool)
    for c in atlas["cells"]:
        covered[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]] = True
    assert (sheet[~covered][:, 3] == 0).all()
    assert Path(res["contact_sheet"]).is_file()


def test_pack_padding_and_alignment(tmp_path):
    big = _solid(tmp_path, 1)
    small = _solid(tmp_path, 2, 32, 16)
    res = po.op_pack({"op": "pack", "images": [{"name": "a", "path": big}, {"name": "b", "path": small}],
                      "cell": None, "columns": 2, "padding": 2, "align": "bottom", "power_of_two": False,
                      "sheet": str(tmp_path / "s.png"), "atlas": str(tmp_path / "s.json"),
                      "contact": str(tmp_path / "c.png")})
    a = res["atlas_data"]
    assert (a["width"], a["height"]) == (130, 64)
    assert a["cells"][1]["x"] == 66
    sheet = np.asarray(Image.open(res["sheet"]))
    # The small image sits at the bottom of its cell, centred horizontally: marker pixel at (66+16, 48).
    assert tuple(sheet[48, 82][:3]) == (2, 253, 7)
    with pytest.raises(ValueError):
        po.op_pack({"op": "pack", "images": [{"name": "a", "path": big}], "cell": [32, 32],
                    "sheet": str(tmp_path / "x.png"), "atlas": str(tmp_path / "x.json"),
                    "contact": str(tmp_path / "xc.png")})


def test_sheet_layout_maths():
    lay = po.sheet_layout(10, 64, 64, 4, 0, True)
    assert (lay["columns"], lay["rows"], lay["width"], lay["height"]) == (4, 3, 256, 256)
    assert lay["cells"][5] == (64, 64)
    assert po.next_pow2(1) == 1 and po.next_pow2(65) == 128 and po.next_pow2(128) == 128


# -- consistency metrics -----------------------------------------------------


def _cell(colour_a=(200, 50, 50), colour_b=(50, 50, 200), shift=0, split=0.5):
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[16:48, 16 + shift:48 + shift, :3] = colour_a
    rows = int(32 * split)
    arr[16 + rows:48, 16 + shift:48 + shift, :3] = colour_b
    arr[16:48, 16 + shift:48 + shift, 3] = 255
    return arr


def test_identical_frames_are_fully_consistent():
    a = _cell()
    assert po.histogram_distance(a, a) == 0.0
    assert po.silhouette_iou(a, a) == 1.0


def test_shifted_silhouette_iou():
    a, b = _cell(), _cell(shift=8)
    # Two 32x32 squares offset by 8: intersection 24*32, union 40*32.
    assert po.silhouette_iou(a, b) == round(24 / 40, 4)
    assert po.histogram_distance(a, b) == 0.0  # same colours in the same shares


def test_colour_drift_is_measured():
    a = _cell()
    assert po.histogram_distance(a, _cell((10, 200, 10), (200, 200, 10))) == 1.0
    assert po.histogram_distance(a, _cell(colour_b=(200, 200, 10))) == 0.5
    assert po.histogram_distance(a, _cell(split=0.75)) == 0.25


def test_metrics_op_reports_base_and_previous(tmp_path):
    base = save(_cell(), tmp_path / "base.png")
    f1 = save(_cell(shift=8), tmp_path / "f1.png")
    f2 = save(_cell(shift=8, colour_b=(200, 200, 10)), tmp_path / "f2.png")
    res = po.op_metrics({"base": base, "frames": [f1, f2]})["frames"]
    assert res[0]["iou"] == 0.6 and res[0]["iou_prev"] == 0.6
    assert res[1]["iou_prev"] == 1.0 and res[1]["hist_distance_prev"] == 0.5


# -- styles, actions and kinds -----------------------------------------------


def test_styles_are_parameters_with_preset_defaults():
    st = sprites.resolve_style(None)
    assert st["name"] == "cartoon" and st["outline"] and st["shading_steps"] == 3
    grim = sprites.resolve_style("grim", outline=False, saturation=0.5)
    assert grim["outline"] is False and grim["saturation"] == 0.5 and "roguelike" in grim["prompt"]
    custom = sprites.resolve_style("watercolour storybook art, soft edges", shading_steps=4)
    assert custom["name"] == "custom" and custom["prompt"].startswith("watercolour") and custom["shading_steps"] == 4
    assert "4 flat tones" in sprites.sprite_prompt("a lantern", custom)


def test_action_poses():
    poses, info = sprites.frame_poses("walk", 6, None)
    assert len(set(poses)) == 6 and info["loop"] and info["fps"] == 10
    poses, _ = sprites.frame_poses("idle", 4, None)
    assert len(poses) == 4
    poses, info = sprites.frame_poses("cast spell", 3, None)
    assert len(poses) == 3 and "cast spell" in poses[0]
    assert sprites.frame_poses("jump", 2, ["crouch", "leap"])[0] == ["crouch", "leap"]
    with pytest.raises(ValueError):
        sprites.frame_poses("jump", 3, ["crouch", "leap"])


def test_tile_kinds_are_caller_supplied():
    kinds = sprites.normalise_kinds(["Lava Pool", {"name": "sand", "seamless": True},
                                     {"name": "statue", "prompt": "a marble statue"},
                                     {"name": "field", "transparent": False}], None)
    assert [k["name"] for k in kinds] == ["lava_pool", "sand", "statue", "field"]
    assert kinds[0]["transparent"] and not kinds[0]["seamless"] and kinds[0]["prompt"] == "Lava Pool"
    assert kinds[1]["seamless"] and not kinds[1]["transparent"]
    assert kinds[2]["prompt"] == "a marble statue"
    assert not kinds[3]["transparent"]
    assert len(sprites.normalise_kinds(None, None)) == 16  # the "strategy" preset
    for bad in ([], ["a", "a"], [{"seamless": True}]):
        with pytest.raises(ValueError):
            sprites.normalise_kinds(bad, None)
    with pytest.raises(ValueError):
        sprites.normalise_kinds(None, "no_such_preset")


PROJECT_WORDS = ("moriveil", "tiny swords", "tiny_swords", "pixel frog", "dwarf", "goblin")


def test_no_project_words_in_tool_names_parameters_or_defaults():
    from localgen_mcp import server

    tools = asyncio.run(server.mcp.list_tools())
    for t in tools:
        if t.name not in ("generate_sprite", "generate_animation", "generate_tileset", "pack_sheet", "pixelize",
                          "import_sprite_sheet", "render_pose_guides"):
            continue
        schema = json.dumps(t.inputSchema).lower()
        for word in PROJECT_WORDS:
            assert word not in t.name.lower() and word not in schema, (t.name, word)
    text = json.dumps([sprites.STYLES, sprites.ACTIONS, sprites.TILE_PRESETS]).lower()
    for word in PROJECT_WORDS:
        assert word not in text


def test_server_process_imports_no_imaging_library():
    code = ("import sys; import localgen_mcp.server; "
            "print([m for m in ('numpy', 'PIL', 'torch') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(SRC)}, check=True)
    assert out.stdout.strip() == "[]"


# -- orchestration (the helper runs for real as a child process) --------------


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALGEN_ROOT", str(tmp_path / "root"))
    return config.load()


def test_pixelize_tool_end_to_end(tmp_path, settings):
    src = save(character(), tmp_path / "hero.png")
    res = asyncio.run(sprites.pixelize(settings, src, size=48, palette="moriveil", max_colors=16,
                                       output_dir=str(tmp_path / "out")))
    out = np.asarray(Image.open(res["sprite"]))
    assert out.shape == (48, 48, 4)
    assert len(colours(out)) <= 16 and res["colours_used"] == len(colours(out))
    assert Image.open(res["preview"]).size == (384, 384)
    assert not res["warnings"]
    again = asyncio.run(sprites.pixelize(settings, src, size=48, output_dir=str(tmp_path / "out")))
    assert again["sprite"] != res["sprite"]  # never overwrites
    assert again["palette_file"] and Path(again["palette_file"]).is_file()
    opaque = np.zeros((64, 64, 3), dtype=np.uint8)
    jpg = tmp_path / "flat.png"
    Image.fromarray(opaque, "RGB").save(jpg)
    warned = asyncio.run(sprites.pixelize(settings, str(jpg), size=16, output_dir=str(tmp_path / "out")))
    assert warned["warnings"]


def test_pixelize_tool_rejects_non_images(tmp_path, settings):
    fake = tmp_path / "x.png"
    fake.write_text("not an image")
    with pytest.raises(ValueError):
        asyncio.run(sprites.pixelize(settings, str(fake)))


def test_pack_sheet_tool(tmp_path, settings):
    paths = [_solid(tmp_path, i) for i in range(3)]
    res = asyncio.run(sprites.pack_sheet(settings, paths + [{"name": "extra", "path": paths[0]}],
                                         output_dir=str(tmp_path / "out"), name="units"))
    assert (res["width"], res["height"]) == (128, 128)
    assert [c["name"] for c in res["cells"]] == ["cell0", "cell1", "cell2", "extra"]
    assert res["unity"]["slice"] == "Grid By Cell Size"


class FakeGPU:
    """Stands in for the drawing and cut-out stages: each "drawing" is a fixture figure written as
    an RGBA PNG (so the cut-out is the drawing itself). bad lists (frame index, attempt) pairs that
    come out with a wrong silhouette and colours."""

    def __init__(self, tmp: Path, bad: set[tuple[int, int]], static_first: bool = False):
        # static_first: first attempts all repeat one pose (no motion); retries step sideways
        self.tmp, self.bad, self.calls, self.n, self.static_first = tmp, bad, [], 0, static_first
        self.attempts: dict[str, int] = {}

    async def draw(self, settings, prompts, seeds, outdir, idle, timeout, progress, refs=None, input_image=None):
        paths = []
        for prompt, seed in zip(prompts, seeds):
            self.calls.append({"prompt": prompt, "seed": seed, "refs": refs, "input_image": input_image})
            frame = None
            if "This is frame" in prompt:
                frame = int(prompt.split("This is frame ")[1].split(" ")[0]) - 1
            key = str(frame)
            attempt = self.attempts.get(key, 0)
            self.attempts[key] = attempt + 1
            img = character(shift=4 * (frame or 0) % 12)
            if self.static_first and frame is not None:
                img = character(stride=0 if attempt == 0 else (90 if frame % 2 else 30))
            if (frame, attempt) in self.bad:
                img = np.zeros_like(img)
                img[40:200, 300:500] = [0.1, 0.9, 0.2, 1.0]
            self.n += 1
            paths.append(save(img, Path(outdir) / f"draw_{self.n}.png"))
        return {"paths": paths, "outputs": [{"paths": [p]} for p in paths], "workflow": "fake"}

    async def cut(self, settings, paths, outdir, idle, timeout, progress):
        out = []
        for p in paths:
            self.n += 1
            dst = Path(outdir) / f"cut_{self.n}.png"
            shutil.copy(p, dst)
            out.append(str(dst))
        return out


def test_animation_retries_a_bad_frame_and_keeps_the_better_one(tmp_path, settings, monkeypatch):
    gpu = FakeGPU(tmp_path, bad={(2, 0)})
    monkeypatch.setattr(sprites, "draw", gpu.draw)
    monkeypatch.setattr(sprites, "cut", gpu.cut)
    res = asyncio.run(sprites.generate_animation(settings, "a test hero with a blue tunic", "walk", frames=4,
                                                 size=32, seed=7, min_motion=0.0, output_dir=str(tmp_path / "anim")))
    per = res["per_frame"]
    assert [f["attempts"] for f in per] == [1, 1, 2, 1]
    assert all(f["passed"] for f in per)
    assert per[2]["seed"] == 7 + 1 + 2 + 1000
    strip = np.asarray(Image.open(res["strip"]))
    assert strip.shape == (32, 128, 4)
    for i, f in enumerate(res["frames"]):
        assert (strip[:, 32 * i:32 * (i + 1)] == np.asarray(Image.open(f))).all()
    data = json.loads(Path(res["json"]).read_text())
    assert data["frame_count"] == 4 and data["frame_size"] == [32, 32] and data["fps"] == 10 and data["loop"]
    # One shared palette: every frame's colours come from the base sprite's palette.
    shared = {tuple(int(res["palette"][i][j:j + 2], 16) for j in (1, 3, 5)) for i in range(len(res["palette"]))}
    for f in res["frames"]:
        assert colours(np.asarray(Image.open(f))) <= shared
    # Frame prompts: the base drawing is image 1 every time, the previous frame image 2 after the first.
    frames = [c for c in gpu.calls if "This is frame" in c["prompt"]]
    assert all(c["input_image"] == res["base_reference"] for c in frames)
    first = frames[:4]  # first attempts, in frame order; the retry of frame 2 follows them
    assert first[0]["refs"] == [] and all(len(c["refs"]) == 1 for c in first[1:])
    assert frames[4]["refs"] == []  # a retry leaves out the previous frame
    assert Path(res["preview_gif"]).is_file()


def test_animation_reports_frames_that_stay_poor(tmp_path, settings, monkeypatch):
    gpu = FakeGPU(tmp_path, bad={(1, 0), (1, 1)})
    monkeypatch.setattr(sprites, "draw", gpu.draw)
    monkeypatch.setattr(sprites, "cut", gpu.cut)
    res = asyncio.run(sprites.generate_animation(settings, "a test hero", "idle", frames=3, size=32, seed=1,
                                                 retries=1, output_dir=str(tmp_path / "anim")))
    assert res["per_frame"][1]["passed"] is False and res["per_frame"][1]["attempts"] == 2
    assert res["consistency"]["frames_passed"] == 2


def test_tileset_orchestration(tmp_path, settings, monkeypatch):
    gpu = FakeGPU(tmp_path, bad=set())
    monkeypatch.setattr(sprites, "draw", gpu.draw)
    monkeypatch.setattr(sprites, "cut", gpu.cut)
    seamless_jobs = []

    async def fake_run(settings_, wf, job_list, outdir, *a, **kw):
        seamless_jobs.extend(job_list)
        paths = []
        for j in job_list:
            img = np.ones((256, 256, 4))
            img[..., 1] = 0.6
            paths.append(save(img, Path(outdir) / f"ground_{j['seed']}.png"))
        return {"outputs": [{"paths": [p]} for p in paths]}

    monkeypatch.setattr(sprites.jobs, "run_workflow", fake_run)
    res = asyncio.run(sprites.generate_tileset(settings, "a desert town", kinds=[
        "cactus", {"name": "sand", "seamless": True}, {"name": "plaza", "transparent": False}],
        size=32, seed=3, output_dir=str(tmp_path / "tiles")))
    assert [t["name"] for t in res["tiles"]] == ["cactus", "sand", "plaza"]
    assert len(seamless_jobs) == 1 and "sand" in seamless_jobs[0]["prompt"]
    sheet = np.asarray(Image.open(res["sheet"]))
    assert sheet.shape[:2] == (32, 128)
    sand = np.asarray(Image.open(res["tiles"][1]["path"]))
    assert (sand[..., 3] == 255).all()  # ground tiles are opaque
    cactus = np.asarray(Image.open(res["tiles"][0]["path"]))
    assert cactus[0, 0, 3] == 0  # objects keep a transparent background
    all_used = set().union(*(colours(np.asarray(Image.open(t["path"]))) for t in res["tiles"]))
    assert len(all_used) <= 32


# -- GPU smoke test ----------------------------------------------------------

GPU = shutil.which("nvidia-smi") is not None and os.environ.get("LOCALGEN_GPU_SMOKE") == "1"


@pytest.mark.skipif(not GPU, reason="needs the GPU, the real backend and models (set LOCALGEN_GPU_SMOKE=1)")
def test_gpu_smoke_sprite_and_idle(tmp_path):
    """One 64 px sprite and a 4 frame idle through the real shared backend."""
    real = config.load()
    out = real.root / "tmp" / f"sprite-smoke-{os.getpid()}"
    try:
        sp = asyncio.run(sprites.generate_sprite(real, "a stout bearded miner with a pickaxe", size=64,
                                                 palette="moriveil", seed=11, output_dir=str(out)))
        img = np.asarray(Image.open(sp["sprite"]))
        assert img.shape == (64, 64, 4) and img[0, 0, 3] == 0 and (img[..., 3] == 255).sum() > 300
        anim = asyncio.run(sprites.generate_animation(real, sp["raw"], "idle", frames=4, size=64, retries=0,
                                                      subject="a stout bearded miner", seed=12, output_dir=str(out)))
        assert np.asarray(Image.open(anim["strip"])).shape == (64, 256, 4)
        assert anim["consistency"]["frames"] == 4
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_animation_static_frames_fail_and_retry_without_the_previous_frame(tmp_path, settings, monkeypatch):
    """Frames that only repeat the pose before them fail the motion check; retries leave out image 2."""
    gpu = FakeGPU(tmp_path, bad=set(), static_first=True)
    monkeypatch.setattr(sprites, "draw", gpu.draw)
    monkeypatch.setattr(sprites, "cut", gpu.cut)
    res = asyncio.run(sprites.generate_animation(settings, "a test hero", "walk", frames=4, size=64, seed=2,
                                                 retries=1, output_dir=str(tmp_path / "anim")))
    per = res["per_frame"]
    assert res["thresholds"]["min_motion"] == 0.05
    assert all(f["attempts"] == 2 for f in per)       # every first attempt repeated the pose
    retries = [c for c in gpu.calls if "This is frame" in c["prompt"]][4:]
    assert len(retries) == 4 and all(c["refs"] == [] for c in retries)
    assert "previous frame" not in retries[1]["prompt"]
    assert all(f["motion"] >= 0.05 for f in per[1:]) and all(f["passed"] for f in per)
    assert res["consistency"]["static_frames"] == 0


def test_static_strip_is_reported_as_failing(tmp_path, settings, monkeypatch):
    gpu = FakeGPU(tmp_path, bad=set(), static_first=True)
    monkeypatch.setattr(sprites, "draw", gpu.draw)
    monkeypatch.setattr(sprites, "cut", gpu.cut)
    res = asyncio.run(sprites.generate_animation(settings, "a test hero", "walk", frames=3, size=64, seed=2,
                                                 retries=0, output_dir=str(tmp_path / "anim")))
    assert res["consistency"]["frames_passed"] == 0
    assert res["consistency"]["static_frames"] == 3
