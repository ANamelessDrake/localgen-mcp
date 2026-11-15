"""generate_turnaround: graph wiring, framing maths and validation (no GPU needed)."""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import struct
import subprocess
import zlib
from pathlib import Path

import pytest

from localgen_mcp import workflows
from localgen_mcp.workflows import turnaround as t

from test_config_workflows import _check_graph, _params


def _graph(**over):
    wf = workflows.get("turnaround")
    over = {"subject": "a robot", "images": {"concept": "localgen/c.png"}, **over}
    return wf.build(_params(wf, **over))


def _source(g, link):
    return g[link[0]]


def _klein_refs(g, view):
    """The IMAGE links a view's Klein chain takes as references, in order."""
    refs, node = [], g[f"{view}_guider"]["inputs"]["positive"][0]
    while g[node]["class_type"] == "ReferenceLatent":
        enc = _source(g, g[node]["inputs"]["latent"])
        refs.append(_source(g, enc["inputs"]["pixels"])["inputs"]["image"])
        node = g[node]["inputs"]["conditioning"][0]
    return refs[::-1]


def test_views_and_azimuths_follow_the_rig():
    # nodes_trellis2._VIEW_AZIMUTHS: front 0, left 90, back 180, right 270.
    assert t.RIG_VIEWS == {"front": 0, "left": 90, "back": 180, "right": 270}
    assert t.views_for(False) == ["front", "left", "back", "right"]
    assert t.ordered(t.views_for(True)) == ["front", "front_left", "left", "back_left", "back", "back_right",
                                            "right", "front_right"]
    # The left view sees the object's left side; the object faces image-left in it.
    assert "faces the left edge of the image" in t.WHERE["left"] and "own left side" in t.WHERE["left"]
    assert "faces the right edge of the image" in t.WHERE["right"]


@pytest.mark.parametrize("three_quarter", [False, True])
def test_graph_is_well_formed(three_quarter):
    g = _graph(three_quarter=three_quarter)
    _check_graph(g)
    views = t.views_for(three_quarter)
    saves = {n["inputs"]["filename_prefix"] for n in g.values() if n["class_type"] == "SaveImage"}
    assert saves == ({f"localgen/test/x_view_{v}" for v in views} | {f"localgen/test/x_raw_{v}" for v in views}
                     | {"localgen/test/x_sheet"})
    # Klein loads once and every view is sampled in this one graph.
    assert [n["class_type"] for n in g.values()].count("UNETLoader") == 1
    assert [n["class_type"] for n in g.values()].count("SamplerCustomAdvanced") == len(views)


def test_front_from_concept_then_views_from_front_and_concept():
    g = _graph()
    concept = ["concept", 0]
    assert _klein_refs(g, "front") == [concept]
    front = ["front_decode", 0]
    for view in ("left", "back", "right"):
        assert _klein_refs(g, view) == [front, concept]
        assert "EmptyFlux2LatentImage" == _source(g, g[f"{view}_sample"]["inputs"]["latent_image"])["class_type"]
    texts = {v: g[f"{v}_text"]["inputs"]["text"] for v in ("front", "left", "back", "right")}
    assert "front view" in texts["front"] and "No cast shadows".lower() in texts["front"].lower()
    assert "left side" in texts["left"] and "from directly behind" in texts["back"]
    assert all("a robot" in s for s in texts.values())


def test_symmetric_mirrors_the_right_views():
    g = _graph(three_quarter=True, symmetric=True)
    for view, partner in t.MIRROR_OF.items():
        assert f"{view}_sample" not in g  # not sampled
        assert g[f"{view}_mirror"]["inputs"]["image"] == [f"{partner}_decode", 0]
        assert g[f"{view}_mask"]["inputs"]["image"] == [f"{view}_mirror", 0]
    assert [n["class_type"] for n in g.values()].count("SamplerCustomAdvanced") == 5


def test_framed_views_carry_their_mask_as_alpha():
    g = _graph()
    for view in t.views_for(False):
        rgba = _source(g, g[f"{view}_save"]["inputs"]["images"])
        assert rgba["class_type"] == "JoinImageWithAlpha"
        # JoinImageWithAlpha inverts its alpha input, so it must get the background (inverted mask).
        inv = _source(g, rgba["inputs"]["alpha"])
        assert inv["class_type"] == "InvertMask"
        alpha = _source(g, inv["inputs"]["mask"])
        assert alpha["class_type"] == "ImageToMask" and alpha["inputs"]["image"] == [f"{view}_frame_mask_s", 0]
        # The mask is moved by exactly the same crops as the colour image.
        for a, b in (("canvas", "canvas_mask"), ("frame", "frame_mask")):
            ca, cb = g[f"{view}_{a}"]["inputs"], g[f"{view}_{b}"]["inputs"]
            assert {k: v for k, v in ca.items() if k not in ("images", "background")} == \
                   {k: v for k, v in cb.items() if k not in ("images", "background")}
        # RemoveBackground's output is the foreground mask: step 1 crops to it directly.
        assert _source(g, g[f"{view}_canvas"]["inputs"]["masks"])["class_type"] == "RemoveBackground"


# -- framing ---------------------------------------------------------------


def _crop_scale(node: dict, bw: float, bh: float, in_w: float, in_h: float) -> tuple[float, float]:
    """ImageCropToMask's rule (comfy_extras/nodes_images._crop_image_with_mask): the input is first
    shrunk to fit max(width, height), the box grows to the output aspect anchored on its larger
    relative side, times pad_factor, and the crop is resized to the output. Returns the scale from
    input pixels to output pixels, and the crop's width in input pixels."""
    w, h, pad = node["width"], node["height"], node["pad_factor"]
    pre = min(1.0, max(w, h) / max(in_w, in_h))
    bw, bh = bw * pre, bh * pre
    aspect = w / h
    crop_h = bw / aspect * pad if bw / bh >= aspect else bh * pad
    return pre * h / crop_h, crop_h * aspect / pre


def _union_inputs(g, link) -> set[str]:
    node = _source(g, link)
    if node["class_type"] == "MaskComposite":
        assert node["inputs"]["operation"] in ("add", "or")
        return _union_inputs(g, node["inputs"]["destination"]) | _union_inputs(g, node["inputs"]["source"])
    return {link[0]}


def simulate(g: dict, boxes: dict[str, tuple[float, float]], image: int = 1024) -> dict:
    """Follow the graph's framing nodes for silhouettes of the given (width, height) in each view."""
    views = list(boxes)
    canvas = {}
    for v in views:
        node = g[f"{v}_canvas"]["inputs"]
        s, _ = _crop_scale(node, *boxes[v], image, image)
        canvas[v] = (boxes[v][0] * s, boxes[v][1] * s, node["width"], node["height"])
    out = {}
    for v in views:
        node = g[f"{v}_frame"]["inputs"]
        members = _union_inputs(g, node["masks"])
        # which views' canvases the crop box is taken from
        used = [u for u in views if f"{u}_canvas_m" in members]
        assert used, f"{v}: the framing crop is not driven by any canvas mask"
        uw = max(canvas[u][0] for u in used)
        uh = max(canvas[u][1] for u in used)
        cw, ch = canvas[v][2], canvas[v][3]
        s, _ = _crop_scale(node, uw, uh, cw, ch)
        scale = g[f"{v}_frame_s"]["inputs"]["width"] / node["width"]
        out[v] = (canvas[v][0] * s * scale, canvas[v][1] * s * scale)
    return out


BOXES = {"front": (420, 760), "left": (900, 640), "back": (400, 800), "right": (300, 500)}


def test_every_view_framed_at_one_scale_with_the_widest_at_1_over_1_1():
    g = _graph()
    out = simulate(g, BOXES)
    heights = [h for _, h in out.values()]
    assert max(heights) - min(heights) < 1e-6  # same silhouette height: same scale in every view
    widest = max(max(w, h) for w, h in out.values())
    assert widest == pytest.approx(1024 / t.RIG_PAD, rel=1e-6)
    # the graph agrees with plan()
    planned = t.plan(BOXES)["views"]
    for v, (w, h) in out.items():
        assert (w, h) == pytest.approx((planned[v]["width"], planned[v]["height"]), rel=1e-6)


def test_wide_object_set_by_its_widest_view():
    boxes = {"front": (900, 300), "left": (300, 280), "back": (880, 290), "right": (320, 300)}
    out = simulate(_graph(), boxes)
    assert max(w for w, _ in out.values()) == pytest.approx(1024 / t.RIG_PAD, rel=1e-6)
    assert len({round(h, 6) for _, h in out.values()}) == 1
    # front width : left width keeps the object's true proportions (900/300 : 300/280)
    assert out["front"][0] / out["left"][0] == pytest.approx((900 / 300) / (300 / 280), rel=1e-6)


def test_plan_maths():
    p = t.plan({"a": (100, 200), "b": (300, 100)}, size=1100)
    # Normalised to one height, b is 6 times wider than a and sets the frame: 1100 / 1.1 = 1000.
    assert p["views"]["b"]["width"] == pytest.approx(1000)
    assert p["views"]["a"]["height"] == pytest.approx(p["views"]["b"]["height"])
    assert p["views"]["a"]["width"] == pytest.approx(1000 / 6)
    with pytest.raises(ValueError):
        t.plan({"a": (500, 100)})  # wider than CANVAS_ASPECT: the height anchor would not hold
    with pytest.raises(ValueError):
        t.plan({"a": (0, 100)})


def test_three_quarter_views_share_the_scale():
    g = _graph(three_quarter=True)
    boxes = {v: (400 + 20 * i, 700 - 10 * i) for i, v in enumerate(t.views_for(True))}
    out = simulate(g, boxes)
    assert len({round(h, 6) for _, h in out.values()}) == 1


# -- outputs and validation ------------------------------------------------


def test_classify_outputs():
    paths = ["/o/ab12_view_front_00001_.png", "/o/ab12_view_back_left_00001_.png", "/o/ab12_raw_left_00001_.png",
             "/o/ab12_sheet_00001_.png", "/o/unrelated.png", "/o/ab12_view_sideways_00001_.png"]
    c = t.classify(paths)
    assert c["views"] == {"front": paths[0], "back_left": paths[1]}
    assert list(c["views"]) == ["front", "back_left"]
    assert c["raw"] == {"left": paths[2]} and c["sheet"] == paths[3]


@pytest.mark.parametrize("bad", [{"background": "grey"}, {"background": "#12345"}, {"subject": "  "}])
def test_bad_options_rejected(bad):
    with pytest.raises(ValueError):
        _graph(**bad)


def test_limits():
    wf = workflows.get("turnaround")
    with pytest.raises(ValueError):
        wf.params({"view_size": 4096})


def test_tool_validates_before_any_work(tmp_path, monkeypatch):
    from localgen_mcp import server

    async def boom(*a, **k):
        raise AssertionError("no job should be submitted")

    monkeypatch.setattr(server.jobs, "run_workflow", boom)
    fake = tmp_path / "secret.png"
    fake.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    with pytest.raises(ValueError, match="image"):
        asyncio.run(server.generate_turnaround(concept_image=str(fake)))
    with pytest.raises(ValueError, match="concept_image or prompt"):
        asyncio.run(server.generate_turnaround())
    with pytest.raises(ValueError, match="background"):
        asyncio.run(server.generate_turnaround(prompt="a robot", background="red"))


# -- guide="mesh": the front stage and the paint stage ------------------------


GUIDED = {"concept": "localgen/c.png", "front": "localgen/f.png",
          **{f"guide_{v}": f"localgen/g_{v}.png" for v in t.ALL_VIEWS}}


def _paint(**over):
    return _graph(stage="paint", images=GUIDED, **over)


def test_front_stage_draws_only_the_clean_front():
    g = _graph(stage="front")
    _check_graph(g)
    assert [n["class_type"] for n in g.values()].count("SamplerCustomAdvanced") == 1
    assert _klein_refs(g, "front") == [["concept", 0]]
    saves = [n["inputs"]["filename_prefix"] for n in g.values() if n["class_type"] == "SaveImage"]
    assert saves == ["localgen/test/x_clean_front"]
    assert t.classify(["/o/ab_clean_front_00001_.png"])["clean_front"] == "/o/ab_clean_front_00001_.png"


@pytest.mark.parametrize("three_quarter", [False, True])
def test_paint_stage_is_well_formed(three_quarter):
    g = _paint(three_quarter=three_quarter)
    _check_graph(g)
    views = t.views_for(three_quarter)
    saves = {n["inputs"]["filename_prefix"] for n in g.values() if n["class_type"] == "SaveImage"}
    assert saves == {f"localgen/test/x_view_{v}" for v in views} | {"localgen/test/x_sheet"}
    assert [n["class_type"] for n in g.values()].count("UNETLoader") == 1
    assert [n["class_type"] for n in g.values()].count("SamplerCustomAdvanced") == len(views)
    # Rig framed already: no framing crops, the painted view itself is saved with its mask as alpha.
    assert not any(n["class_type"] == "ImageCropToMask" for n in g.values())
    for view in views:
        rgba = _source(g, g[f"{view}_save"]["inputs"]["images"])
        assert rgba["inputs"]["image"] == [f"{view}_decode", 0]
        inv = _source(g, rgba["inputs"]["alpha"])
        assert inv["class_type"] == "InvertMask"
        assert _source(g, inv["inputs"]["mask"])["class_type"] == "RemoveBackground"


def test_every_view_starts_from_its_own_render():
    g = _paint(three_quarter=True)
    for view in t.views_for(True):
        load = g[f"{view}_guide"]
        assert load["class_type"] == "LoadImage" and load["inputs"]["image"] == f"localgen/g_{view}.png"
        render = [f"{view}_guide", 0]
        # image to image: the start latent is the VAE encoding of this view's own render
        latent = _source(g, g[f"{view}_sample"]["inputs"]["latent_image"])
        assert latent["class_type"] == "VAEEncode" and latent["inputs"]["pixels"] == render
        sigmas = _source(g, g[f"{view}_sample"]["inputs"]["sigmas"])
        assert sigmas["class_type"] == "SplitSigmasDenoise" and g[f"{view}_sample"]["inputs"]["sigmas"][1] == 1
        assert _klein_refs(g, view) == [render, ["front", 0], ["concept", 0]]
        text = g[f"{view}_text"]["inputs"]["text"]
        assert t.WHERE[view] in text and "Keep exactly the same shape, silhouette, pose" in text


def test_guide_denoise_sets_the_schedule():
    g = _paint(guide_denoise=0.6, steps=4)
    split = g["left_split"]["inputs"]
    sched = _source(g, split["sigmas"])["inputs"]
    # schedule_steps: at least `steps` sampled steps, so 6 of 10 run, a real denoise of exactly 0.6
    assert (sched["steps"], split["denoise"]) == (10, pytest.approx(0.6))
    g = _paint(guide_denoise=0.75, steps=4)
    assert g["left_split"]["inputs"]["denoise"] == pytest.approx(0.75)


def test_paint_stage_symmetric_mirrors_without_rendering_the_right():
    g = _paint(three_quarter=True, symmetric=True)
    for view, partner in t.MIRROR_OF.items():
        assert f"{view}_guide" not in g and f"{view}_sample" not in g
        assert g[f"{view}_mirror"]["inputs"]["image"] == [f"{partner}_decode", 0]
    assert t.sampled(t.ordered(t.views_for(True)), True) == ["front", "front_left", "left", "back_left", "back"]


@pytest.mark.parametrize("bad", [{"guide": "depth"}, {"stage": "all"}, {"guide_denoise": 0.1},
                                 {"guide_denoise": 1.0}])
def test_bad_guide_options_rejected(bad):
    with pytest.raises(ValueError):
        _graph(**bad)


def test_guide_none_is_the_default():
    # guide="mesh" lost to "none" on a standing robot (the guide mesh copied its face onto its back)
    import inspect
    from localgen_mcp import server
    assert workflows.get("turnaround").defaults["guide"] == "none"
    assert inspect.signature(server.generate_turnaround).parameters["guide"].default == "none"


# -- the tool's stages, with the GPU and Blender faked -------------------------


class _Stages:
    """Fakes for jobs.run_workflow, generate_mesh and blender_run.run that record what the tool asks."""

    def __init__(self, tmp_path, fail_front_mesh=False):
        self.tmp, self.calls, self.fail_front_mesh = tmp_path, [], fail_front_mesh

    async def run_workflow(self, settings, wf, jobs_, output_dir, idle, images=None, timeout=0, progress=None,
                           mesh=None):
        p = wf.params(jobs_[0])
        self.calls.append(("workflow", p["stage"], dict(images or {}), p))
        stem = f"{self.tmp}/s{len(self.calls)}"
        if p["stage"] == "front":
            paths = [f"{stem}_clean_front_00001_.png"]
        else:
            views = t.views_for(p["three_quarter"])
            paths = [f"{stem}_view_{v}_00001_.png" for v in views] + [f"{stem}_sheet_00001_.png"]
        return {"paths": paths, "outputs": [{"seed": p.get("seed", 42)}],
                "output_dir": str(self.tmp), "total_seconds": 1.0}

    async def generate_mesh(self, **kw):
        self.calls.append(("mesh", kw))
        if self.fail_front_mesh and kw.get("face_count") and "clean_front" in kw["input_image"]:
            from localgen_mcp import jobs
            raise jobs.JobError("trellis2 failed: RemeshMesh (node 241): out of memory")
        return {"paths": [f"{self.tmp}/guide.glb"], "total_seconds": 2.0}

    async def blender(self, settings, script, args, marker, expect):
        spec = json.loads(args[0])
        self.calls.append(("blender", script, spec))
        return {"views": {v: f"{spec['prefix']}_{v}.png" for v in spec["views"]}, "bbox": [1, 0.3, 0.8],
                "boxes": {}, "seconds": 3.0}


def _run_tool(monkeypatch, stages, **kw):
    from localgen_mcp import server
    monkeypatch.setattr(server.jobs, "run_workflow", stages.run_workflow)
    monkeypatch.setattr(server, "generate_mesh", stages.generate_mesh)
    monkeypatch.setattr(server.blender_run, "run", stages.blender)
    return asyncio.run(server.generate_turnaround(**kw))


def _concept(tmp_path):
    concept = tmp_path / "c.png"
    concept.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 32)
    return str(concept)


def test_guided_tool_runs_front_mesh_render_paint(tmp_path, monkeypatch):
    concept = _concept(tmp_path)
    st = _Stages(tmp_path)
    r = _run_tool(monkeypatch, st, concept_image=concept, subject="a moth", guide="mesh", symmetric=True, seed=5,
                  background="#102030", mesh=True)
    assert [c[1] if c[0] == "workflow" else c[0] for c in st.calls] == ["front", "mesh", "blender", "paint", "mesh"]
    front_png = f"{tmp_path}/s1_clean_front_00001_.png"
    guide = st.calls[1][1]
    assert guide["input_image"] == front_png and guide["face_count"] == 50000 and guide["seed"] == 5
    spec = st.calls[2][2]
    assert st.calls[2][1] == "render_rig_views.py"
    assert spec["mesh"] == f"{tmp_path}/guide.glb" and spec["background"] == [16, 32, 48]
    assert spec["views"] == {"front": 0, "left": 90, "back": 180}  # right is mirrored from left
    assert spec["match"] == front_png  # the guide's facing is matched to the image it was made from
    images = st.calls[3][2]
    assert images["concept"] == concept and images["front"] == front_png
    assert {k for k in images if k.startswith("guide_")} == {"guide_front", "guide_left", "guide_back"}
    assert images["guide_left"] == spec["prefix"] + "_left.png"
    assert st.calls[3][3]["seed"] == 5
    final = st.calls[4][1]
    assert final["framing"] == "as_framed" and final["remove_background"] is False
    assert final["left_image"].endswith("_view_left_00001_.png")
    assert r["guide_mesh"] == f"{tmp_path}/guide.glb" and r["guide_source"] == "front" and not r["warnings"]
    assert set(r["guide_renders"]) == {"front", "left", "back"} and r["clean_front"] == front_png


def test_guide_mesh_falls_back_to_the_concept(tmp_path, monkeypatch):
    concept = _concept(tmp_path)
    st = _Stages(tmp_path, fail_front_mesh=True)
    r = _run_tool(monkeypatch, st, concept_image=concept, subject="a moth", guide="mesh")
    meshes = [c[1]["input_image"] for c in st.calls if c[0] == "mesh"]
    assert meshes == [f"{tmp_path}/s1_clean_front_00001_.png", concept]
    assert r["guide_source"] == "concept" and "out of memory" in r["warnings"][0]


def test_guide_none_is_the_old_single_graph(tmp_path, monkeypatch):
    st = _Stages(tmp_path)
    r = _run_tool(monkeypatch, st, concept_image=_concept(tmp_path), subject="a robot", guide="none")
    assert [c[1] for c in st.calls] == ["full"]
    assert "guide_mesh" not in r


def test_bad_guide_rejected_before_any_work(monkeypatch):
    from localgen_mcp import server

    async def boom(*a, **k):
        raise AssertionError("no job should be submitted")

    monkeypatch.setattr(server.jobs, "run_workflow", boom)
    with pytest.raises(ValueError, match="guide"):
        asyncio.run(server.generate_turnaround(prompt="a robot", guide="depth"))
    with pytest.raises(ValueError, match="guide_source"):
        asyncio.run(server.generate_turnaround(prompt="a robot", guide_source="back"))
    with pytest.raises(ValueError, match="guide_denoise"):
        asyncio.run(server.generate_turnaround(prompt="a robot", guide_denoise=0.99))


# -- the rig renders (Blender) -------------------------------------------------


BLENDER = shutil.which("blender")


def _png_first_pixel(path: Path) -> tuple:
    """The first pixel of an 8 bit RGB(A) PNG, standard library only (row 0's filter is None or Sub,
    both of which leave the first pixel as stored)."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    i, idat = 8, b""
    while i < len(data):
        n, kind = struct.unpack(">I4s", data[i:i + 8])
        if kind == b"IDAT":
            idat += data[i + 8:i + 8 + n]
        i += 12 + n
    raw = zlib.decompress(idat)
    assert raw[0] in (0, 1)
    return tuple(raw[1:4])


@pytest.mark.skipif(BLENDER is None, reason="Blender is not installed")
def test_rig_renders_frame_and_orient_like_the_rig(tmp_path, monkeypatch):
    """A 1.0 x 0.5 x 0.8 box with a nose at its front: one scale in every view, the widest side at
    1/1.1 of the frame, the nose at image-left in the left view, the background colour exact."""
    from localgen_mcp import blender_run, config
    monkeypatch.setenv("LOCALGEN_ROOT", str(tmp_path))
    monkeypatch.setenv("LOCALGEN_BLENDER_THREADS", "4")
    settings = config.load()
    glb = tmp_path / "box.glb"
    subprocess.run([BLENDER, "-b", "--factory-startup", "-P", str(Path(__file__).parent / "blender_make_box.py"),
                    "--", str(glb)], check=True, capture_output=True)
    size = 256
    spec = {"mesh": str(glb), "prefix": str(tmp_path / "r"), "size": size, "samples": 4,
            "views": dict(t.RIG_VIEWS), "background": [16, 32, 48]}
    rep = asyncio.run(blender_run.run(settings, "render_rig_views.py", [json.dumps(spec)], "RIG_VIEWS_RESULT",
                                      tmp_path / "r_right.png"))
    assert rep["bbox"] == pytest.approx([1.0, 0.5, 0.8], abs=1e-3)
    box = rep["boxes"]
    w = {v: b[2] - b[0] for v, b in box.items()}
    h = {v: b[3] - b[1] for v, b in box.items()}
    dist = 1.1 * 0.5 / math.tan(math.radians(10))

    def px(extent, near):
        """Pixels spanned by `extent` on a face `near` closer to the camera than the origin (pinhole:
        the frame is 1.1 wide at the origin)."""
        return extent * size / (1.1 * (dist - near) / dist)
    # Centred on its bounding box, the box spans y -0.05..0.25 and the nose reaches -0.25: the box's
    # front face is 0.05 nearer the front camera than the origin, its back face 0.25 nearer the back one.
    assert w["front"] == pytest.approx(px(1.0, 0.05), abs=2) and w["back"] == pytest.approx(px(1.0, 0.25), abs=2)
    assert h["front"] == pytest.approx(px(0.8, 0.05), abs=2) and h["left"] == pytest.approx(px(0.8, 0.5), abs=2)
    for v, b in box.items():  # centred vertically
        assert (b[1] + b[3]) / 2 == pytest.approx(size / 2, abs=2), v
    # Side views: the nose tip (0.25 in front of the centre, its sides 0.1 from it) and the box's
    # back (0.25 behind, its sides 0.5 from it) project at different scales. In the left view the
    # front points at image-left, in the right view at image-right.
    nose, tail = px(0.25, 0.1), px(0.25, 0.5)
    assert size / 2 - box["left"][0] == pytest.approx(nose, abs=2)
    assert box["left"][2] - size / 2 == pytest.approx(tail, abs=2)
    assert box["right"][2] - size / 2 == pytest.approx(nose, abs=2)
    assert size / 2 - box["right"][0] == pytest.approx(tail, abs=2)
    assert _png_first_pixel(Path(rep["views"]["front"])) == (16, 32, 48)


@pytest.mark.skipif(BLENDER is None, reason="Blender is not installed")
def test_rig_renders_turn_a_mesh_that_does_not_face_minus_y(tmp_path, monkeypatch):
    """The box turned 90 degrees (its front now faces +X): with "match" set to a front view of the
    unturned box, the renders find the yaw and the front view shows the box's wide front again."""
    from localgen_mcp import blender_run, config
    monkeypatch.setenv("LOCALGEN_ROOT", str(tmp_path))
    monkeypatch.setenv("LOCALGEN_BLENDER_THREADS", "4")
    settings = config.load()
    make = str(Path(__file__).parent / "blender_make_box.py")
    for name, yaw in (("straight", "0"), ("turned", "90")):
        subprocess.run([BLENDER, "-b", "--factory-startup", "-P", make, "--", str(tmp_path / f"{name}.glb"), yaw],
                       check=True, capture_output=True)

    def run(glb, prefix, match=None):
        spec = {"mesh": str(glb), "prefix": str(tmp_path / prefix), "size": 128, "samples": 4,
                "views": {"front": 0, "left": 90}, "background": [216, 216, 216]}
        if match:
            spec["match"] = match
        return asyncio.run(blender_run.run(settings, "render_rig_views.py", [json.dumps(spec)],
                                           "RIG_VIEWS_RESULT", tmp_path / f"{prefix}_left.png"))
    ref = run(tmp_path / "straight.glb", "ref")
    assert ref["yaw"] == 0 and ref["yaw_scores"] == {}
    plain = run(tmp_path / "turned.glb", "plain")
    turned = run(tmp_path / "turned.glb", "matched", match=ref["views"]["front"])
    width = {k: r["boxes"]["front"][2] - r["boxes"]["front"][0] for k, r in
             (("ref", ref), ("plain", plain), ("turned", turned))}
    assert width["plain"] < 0.7 * width["ref"]  # unmatched, the front camera sees the box's side
    assert turned["yaw"] in (90, 270) and width["turned"] == pytest.approx(width["ref"], abs=3)
    sc = turned["yaw_scores"]
    assert max(sc[str(turned["yaw"])], sc[str((turned["yaw"] + 180) % 360)]) < min(sc["0"], sc["180"])
