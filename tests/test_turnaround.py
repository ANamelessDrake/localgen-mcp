"""generate_turnaround: graph wiring, framing maths and validation (no GPU needed)."""

from __future__ import annotations

import asyncio

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
    g = _graph(ref_order="concept_first")
    assert _klein_refs(g, "left") == [concept, front]
    texts = {v: g[f"{v}_text"]["inputs"]["text"] for v in ("front", "left", "back", "right")}
    assert "front view" in texts["front"] and "No cast shadows".lower() in texts["front"].lower()
    assert "left side" in texts["left"] and "from directly behind" in texts["back"]
    assert all("a robot" in s for s in texts.values())


def test_back_can_start_from_the_mirrored_front():
    g = _graph(back_start="mirror", back_denoise=0.85)
    sample = g["back_sample"]["inputs"]
    init = _source(g, sample["latent_image"])
    assert init["class_type"] == "VAEEncode" and init["inputs"]["pixels"] == ["back_flip", 0]
    assert g["back_flip"]["inputs"] == {"image": ["front_decode", 0], "flip_method": "y-axis: horizontally"}
    assert g["back_split"]["inputs"]["denoise"] == 0.85
    assert "init" not in "".join(k for k in g if k.startswith("left_"))


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


@pytest.mark.parametrize("bad", [{"background": "grey"}, {"back_start": "flip"}, {"ref_order": "random"},
                                 {"back_denoise": 0.2}, {"subject": "  "}])
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
