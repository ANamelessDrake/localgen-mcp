from __future__ import annotations

import pytest

from localgen_mcp import config, workflows


def test_non_loopback_host_rejected(monkeypatch):
    monkeypatch.setenv("LOCALGEN_HOST", "0.0.0.0")
    with pytest.raises(ValueError, match="loopback"):
        config.load()


def test_defaults(monkeypatch):
    for k in ("LOCALGEN_HOST", "LOCALGEN_IDLE_TIMEOUT", "LOCALGEN_IDLE_TIMEOUT_MIN", "LOCALGEN_IDLE_TIMEOUT_MAX", "LOCALGEN_ROOT"):
        monkeypatch.delenv(k, raising=False)
    s = config.load()
    assert s.host == "127.0.0.1"
    assert s.idle_timeout == 300
    assert (s.idle_timeout_min, s.idle_timeout_max) == (60, 7200)
    assert str(s.output_dir).startswith("/ai/")
    assert s.clamp_timeout(5) == 60 and s.clamp_timeout(10**6) == 7200 and s.clamp_timeout(None) == 300


def test_one_default_per_modality():
    for modality in ("image", "video", "mesh"):
        assert sum(w.is_default for w in workflows.for_modality(modality)) == 1


def _check_graph(g: dict) -> None:
    """Every link points at a node that exists and every node has a class_type."""
    for nid, node in g.items():
        assert node.get("class_type"), nid
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in g, f"node {nid} links to missing node {v[0]}"


def _params(wf, **over):
    p = wf.params(over)
    p.setdefault("seed", 1)
    p["prefix"] = "localgen/test/x"
    if wf.needs_mesh:
        p.setdefault("mesh", "3d/localgen/m.glb")
        p.setdefault("placeholder", "localgen/placeholder.png")
    return p


@pytest.mark.parametrize("name", list(workflows.REGISTRY))
def test_default_graphs_are_well_formed(name):
    wf = workflows.REGISTRY[name]
    p = _params(wf, prompt="test", images={slot: f"localgen/{slot}.png" for slot in wf.image_slots})
    _check_graph(wf.build(p))
    if wf.min_images == 0:
        _check_graph(wf.build(_params(wf, prompt="test")))


def test_wan_frames_rounded_to_4n_plus_1_and_start_image():
    wf = workflows.get("video")
    g = wf.build(_params(wf, prompt="x", frames=50))
    latent = next(n for n in g.values() if n["class_type"] == "Wan22ImageToVideoLatent")
    assert latent["inputs"]["length"] == 49 and "start_image" not in latent["inputs"]
    g = wf.build(_params(wf, prompt="x", images={"start_image": "localgen/a.png"}))
    latent = next(n for n in g.values() if n["class_type"] == "Wan22ImageToVideoLatent")
    assert g[latent["inputs"]["start_image"][0]]["class_type"] == "LoadImage"


def test_trellis_without_background_removal_uses_alpha():
    wf = workflows.get("mesh")
    g = wf.build(_params(wf, images={"image": "localgen/a.png"}, remove_background=False))
    _check_graph(g)
    assert not any(n["class_type"].startswith("BiRefNet") or "BackgroundRemoval" in n["class_type"] for n in g.values())
    # LoadImage's mask output is 1 - alpha; the crop needs the foreground, so it must be inverted.
    mask = g["312"]["inputs"]["masks"]
    assert g[mask[0]]["class_type"] == "InvertMask"
    assert g[mask[0]]["inputs"]["mask"] == ["122", 1]


def test_limits_enforced():
    wf = workflows.get("image")
    with pytest.raises(ValueError):
        wf.params({"width": 10000})


def test_pick_by_image_slots():
    assert workflows.pick("image").name == "z-image-turbo"
    assert workflows.pick("image", slots={"ref1", "ref2"}).name == "flux2-klein-4b"
    assert workflows.pick("mesh", slots={"image"}).name == "trellis2"
    assert workflows.pick("mesh", slots={"front", "back"}).name == "pixal3d-multiview"
    assert workflows.pick("video", slots={"start_image"}).is_default
    with pytest.raises(ValueError):
        workflows.pick("image", slots={f"ref{i}" for i in range(1, 6)})  # five references
    with pytest.raises(ValueError):
        workflows.pick("mesh", "trellis2", {"front", "left"})


def _classes(g):
    return [n["class_type"] for n in g.values()]


def test_klein_chains_every_reference_into_both_conditionings():
    wf = workflows.get("image", "flux2-klein-4b")
    g = wf.build(_params(wf, prompt="x", images={"ref1": "a.png", "ref2": "b.png", "ref3": "c.png"}))
    _check_graph(g)
    assert _classes(g).count("LoadImage") == 3
    assert _classes(g).count("ReferenceLatent") == 6
    guider = next(n for n in g.values() if n["class_type"] == "CFGGuider")["inputs"]
    # The last references in each chain feed the guider, so all three are attached.
    depth = 0
    node = guider["positive"][0]
    while g[node]["class_type"] == "ReferenceLatent":
        depth += 1
        node = g[node]["inputs"]["conditioning"][0]
    assert depth == 3 and g[node]["class_type"] == "CLIPTextEncode"
    # No explicit size: the latent follows the first reference image.
    latent = next(n for n in g.values() if n["class_type"] == "EmptyFlux2LatentImage")["inputs"]
    assert g[latent["width"][0]]["class_type"] == "GetImageSize"
    g = wf.build(_params(wf, prompt="x", images={"ref1": "a.png"}, width=768, height=512))
    latent = next(n for n in g.values() if n["class_type"] == "EmptyFlux2LatentImage")["inputs"]
    assert (latent["width"], latent["height"]) == (768, 512)


def test_pixal3d_multiview_feeds_each_view():
    wf = workflows.get("mesh", "pixal3d-multiview")
    g = wf.build(_params(wf, images={"front": "f.png", "back": "b.png"}))
    _check_graph(g)
    cond = g["299"]
    assert cond["class_type"] == "Pixal3DMultiViewConditioning"
    assert set(k for k in ("front", "left", "back", "right") if k in cond["inputs"]) == {"front", "back"}
    assert g["40"]["inputs"]["unet_name"].startswith("pixal3d_multiview")
    assert "Trellis2Conditioning" not in _classes(g)
    assert _classes(g).count("RemoveBackground") == 2
    assert _classes(g).count("ImageCropToMask") == 2


def _source(g, ref):
    return g[ref[0]]


def test_pixal3d_as_framed_keeps_framing():
    wf = workflows.get("mesh", "pixal3d-multiview")
    for remove_bg in (True, False):
        g = wf.build(_params(wf, images={"front": "f.png", "left": "l.png"}, framing="as_framed",
                             remove_background=remove_bg))
        _check_graph(g)
        assert "ImageCropToMask" not in _classes(g)
        for slot in ("front", "left"):
            join = _source(g, g["299"]["inputs"][slot])
            assert join["class_type"] == "JoinImageWithAlpha"
            # JoinImageWithAlpha wants the background (it inverts), so the foreground is inverted first.
            inv = _source(g, join["inputs"]["alpha"])
            assert inv["class_type"] == "InvertMask"
            fg = _source(g, inv["inputs"]["mask"])
            if remove_bg:
                assert fg["class_type"] == "RemoveBackground"
            else:
                # alpha path: LoadImage mask (1 - alpha) inverted to the foreground, then back
                assert fg["class_type"] == "InvertMask"
                assert _source(g, fg["inputs"]["mask"])["class_type"] == "LoadImage"


def test_pixal3d_rejects_unknown_framing():
    wf = workflows.get("mesh", "pixal3d-multiview")
    with pytest.raises(ValueError):
        wf.build(_params(wf, images={"front": "f.png"}, framing="stretch"))


def test_only_real_images_are_uploaded(tmp_path):
    from localgen_mcp.jobs import is_image_file
    png = tmp_path / "a.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 32)
    webp = tmp_path / "b.webp"
    webp.write_bytes(b"RIFF\x10\0\0\0WEBPVP8 " + b"\0" * 16)
    secret = tmp_path / "id_rsa.png"  # an image extension does not make it an image
    secret.write_bytes(b"-----BEGIN OPENSSH PRIVATE KEY-----\n")
    assert is_image_file(png) and is_image_file(webp)
    assert not is_image_file(secret)


def test_texture_paints_each_view_from_its_own_clay_render():
    wf = workflows.get("texture")
    g = wf.build(_params(wf, prompt="a red robot", images={"style": "localgen/style.png"}, views=4))
    _check_graph(g)
    samplers = [k for k, n in g.items() if n["class_type"] == "SamplerCustomAdvanced"]
    assert len(samplers) == 4
    for v in range(4):
        sample = g[f"paint{v}_sample"]["inputs"]
        # image to image: the latent is the VAE encoding of this view's clay render, and only the
        # low (denoise) part of the schedule runs; from an empty latent Klein drifts to a front view
        init = g[sample["latent_image"][0]]
        assert init["class_type"] == "VAEEncode" and init["inputs"]["pixels"] == [f"view{v}", 0]
        assert g[sample["sigmas"][0]]["class_type"] == "SplitSigmasDenoise" and sample["sigmas"][1] == 1
        assert g[f"view{v}"]["inputs"]["batch_index"] == v
    # the style image guides the front view only
    style_users = [k for k, n in g.items() if n["inputs"].get("image") == ["style", 0]]
    assert len(style_users) == 1 and style_users[0].startswith("paint0_")
    texts = [g[f"paint{v}_text"]["inputs"]["text"] for v in range(4)]
    assert "clay render of the front" in texts[0] and "clay render of the back" in texts[2]
    assert "clay render of the side" in texts[1] and "clay render of the side" in texts[3]
    bake = g["bake"]["inputs"]
    assert bake["camera_info"] == ["orbit", 0] and bake["mask"] == ["clay", 1] and bake["fill_unseen"] is True
    assert g["load"]["inputs"]["model_file"] == "3d/localgen/m.glb"
    assert "UnwrapMesh" in _classes(g)
    assert "UnwrapMesh" not in _classes(wf.build(_params(wf, prompt="x", unwrap=False)))


def test_texture_view_names():
    from localgen_mcp.workflows.texture_mesh import view_name
    assert [view_name(a) for a in (0, 90, 180, 270)] == ["front", "side", "back", "side"]
    assert view_name(45) == "three-quarter front" or view_name(45).startswith("three-quarter")
    assert view_name(350) == "front"


def test_only_glb_files_are_uploaded(tmp_path):
    # texture_mesh re-serialises every input through Blender; run_workflow uploads only GLBs.
    from localgen_mcp.jobs import is_glb
    glb = tmp_path / "a.glb"
    glb.write_bytes(b"glTF" + b"\0" * 16)
    obj = tmp_path / "b.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    assert is_glb(glb) and not is_glb(obj)


def test_part_boxes_accept_fractions_or_pixels():
    from localgen_mcp.server import _part_boxes
    out = _part_boxes([{"name": "left arm!", "box": [0.5, 0.25, 0.75, 0.5]},
                       {"name": "leg", "box": [10, 20, 2000, 300], "parent": "left_arm"}], 1000, 800)
    assert out[0]["name"] == "left_arm" and out[0]["box"] == [500, 200, 750, 400]
    assert out[1]["box"] == [10, 20, 1000, 300] and out[1]["parent"] == "left_arm"  # clamped to the image
    with pytest.raises(ValueError):
        _part_boxes([{"name": "a", "box": [0, 0, 0.5, 0.5]}, {"name": "a", "box": [0, 0, 0.5, 0.5]}], 100, 100)
    with pytest.raises(ValueError):
        _part_boxes([{"name": "tiny", "box": [10, 10, 12, 12]}], 100, 100)
    with pytest.raises(ValueError):
        _part_boxes([], 100, 100)


def test_part_isolation_crops_then_cleans():
    wf = workflows.get("part")
    g = wf.build(_params(wf, prompt="x", images={"concept": "c.png"}, x=10, y=20, w=300, h=400))
    _check_graph(g)
    assert g["crop"]["inputs"] | {} == {"image": ["concept", 0], "x": 10, "y": 20, "width": 300, "height": 400}
    # Klein's only reference is the crop, which pins the part's identity and pose
    refs = [n for n in g.values() if n["class_type"] == "ReferenceLatent"]
    scales = {g[r["inputs"]["latent"][0]]["inputs"]["pixels"][0] for r in refs}
    assert {g[s]["inputs"]["image"][0] for s in scales} == {"crop"}


# -- image to image and masked editing ----------------------------------------


def _node(g, ref):
    return g[ref[0]]


def test_zimage_text_to_image_graph_unchanged_without_input_image():
    wf = workflows.get("image")
    g = wf.build(_params(wf, prompt="x"))
    assert sorted(g, key=int) == [str(i) for i in range(1, 11)]
    k = g["8"]["inputs"]
    assert _node(g, k["latent_image"])["class_type"] == "EmptySD3LatentImage" and k["denoise"] == 1
    assert g["10"]["inputs"]["images"] == ["9", 0]


def test_zimage_image_to_image_starts_from_the_encoded_input():
    wf = workflows.get("image")
    g = wf.build(_params(wf, prompt="x", denoise=0.3, images={"image": "localgen/in.png"}))
    _check_graph(g)
    assert "EmptySD3LatentImage" not in _classes(g) and "SetLatentNoiseMask" not in _classes(g)
    k = g["8"]["inputs"]
    assert k["steps"] / int(k["steps"] / k["denoise"]) == pytest.approx(0.3)  # what KSampler really applies
    enc = _node(g, k["latent_image"])
    assert enc["class_type"] == "VAEEncode"
    work = _node(g, enc["inputs"]["pixels"])
    # any input size: worked on at about 1 MP with both sides a multiple of 16
    assert work["class_type"] == "ImageScaleToTotalPixels" and work["inputs"]["resolution_steps"] == 16
    assert _node(g, work["inputs"]["image"])["inputs"]["image"] == "localgen/in.png"
    # the result goes back to the input's own size
    restore = _node(g, g["10"]["inputs"]["images"])
    assert restore["class_type"] == "ResizeImageMaskNode" and restore["inputs"]["input"] == ["9", 0]
    assert restore["inputs"]["resize_type"] == "match size"
    assert restore["inputs"]["resize_type.match"] == work["inputs"]["image"]
    k = wf.build(_params(wf, prompt="x", images={"image": "a.png"}))["8"]["inputs"]
    assert k["steps"] / int(k["steps"] / k["denoise"]) == pytest.approx(0.6)  # the default


def test_partial_denoise_is_honoured_with_few_steps():
    from localgen_mcp.workflows.image_edit import ksampler_steps, schedule_steps
    for d in (0.05, 0.1, 0.3, 0.45, 0.6, 0.75, 0.8, 0.9, 0.95, 0.99):
        steps, passed = ksampler_steps(8, d)
        total = int(steps / passed)  # KSampler's schedule length
        assert 8 <= steps < total, d  # never the full schedule, which would ignore the input image
        assert abs(steps / total - d) <= (0.01 if d <= 0.9 else 0.06), d
        t, passed = schedule_steps(4, d)  # SplitSigmasDenoise runs round(t * passed) of t
        run = round(t * passed)
        assert 0 < run < t and abs(run / t - d) <= 0.07, d
    assert ksampler_steps(8, 1.0) == (8, 1.0) and schedule_steps(4, 1.0) == (4, 1.0)
    # with 8 steps KSampler alone would turn 0.9 into a full schedule (int(8 / 0.9) == 8)
    assert int(8 / 0.9) == 8


def _check_masked(g, decoded):
    """Sampling is limited to the mask, and the decoded result is pasted over the original
    outside it, with the mask read from the mask image's red channel at the original's size."""
    original = ["edit_src", 0]
    assert g["edit_src"]["class_type"] == "LoadImage"
    comp = _node(g, next(n for n in g.values() if n["class_type"] == "SaveImage")["inputs"]["images"])
    assert comp["class_type"] == "ImageCompositeMasked"
    ci = comp["inputs"]
    assert ci["destination"] == original and ci["source"] == decoded
    assert ci["resize_source"] is True and (ci["x"], ci["y"]) == (0, 0)
    # the composite mask: red channel of the mask image (white = change), fitted to the original,
    # grown, then feathered; never LoadImage's own mask output (1 - alpha)
    mask = ci["mask"]
    chain = []
    while True:
        node = _node(g, mask)
        chain.append(node["class_type"])
        if node["class_type"] == "LoadImage":
            break
        mask = node["inputs"].get("mask") or node["inputs"].get("image") or node["inputs"].get("input")
        assert mask[1] == 0, f"{node['class_type']} reads output {mask[1]}"
    assert chain == ["ImageToMask", "ImageBlur", "MaskToImage", "GrowMask", "ResizeImageMaskNode", "ImageToMask",
                     "LoadImage"]
    assert g["edit_mask_red"]["inputs"]["channel"] == "red"
    assert g["edit_mask_fit"]["inputs"]["resize_type.match"] == original
    assert g["edit_mask_grow"]["inputs"]["expand"] > 0
    noise = next(n for n in g.values() if n["class_type"] == "SetLatentNoiseMask")["inputs"]
    assert noise["mask"] == ci["mask"] and _node(g, noise["samples"])["class_type"] == "VAEEncode"
    return noise


def test_zimage_masked_edit_composites_back_over_the_original():
    wf = workflows.get("image")
    g = wf.build(_params(wf, prompt="x", denoise=0.9, images={"image": "localgen/in.png", "mask": "localgen/m.png"}))
    _check_graph(g)
    _check_masked(g, ["9", 0])
    assert _node(g, g["8"]["inputs"]["latent_image"])["class_type"] == "SetLatentNoiseMask"
    assert g["edit_mask_load"]["inputs"]["image"] == "localgen/m.png"


def test_klein_edits_the_input_image_as_image_one():
    wf = workflows.get("image", "flux2-klein-4b")
    g = wf.build(_params(wf, prompt="x", images={"image": "in.png", "mask": "m.png", "ref1": "hat.png"}))
    _check_graph(g)
    noise = _check_masked(g, ["decode", 0])
    assert noise["samples"] == ["init", 0] and g["init"]["inputs"]["pixels"] == ["edit_work", 0]
    sample = g["sample"]["inputs"]
    assert sample["latent_image"] == ["noise_mask", 0]
    assert sample["sigmas"] == ["split", 1] and g["split"]["inputs"]["denoise"] == 1.0
    # image 1 is the edited image, then the references in order
    assert g["ref0_scale"]["inputs"]["image"] == ["edit_work", 0]
    assert g["ref1_scale"]["inputs"]["image"] == ["load0", 0] and g["load0"]["inputs"]["image"] == "hat.png"
    assert "EmptyFlux2LatentImage" not in _classes(g)
    assert g["sched"]["inputs"]["width"] == ["size", 0] and g["size"]["inputs"]["image"] == ["edit_work", 0]
    # without a mask: no noise mask, the result is resized back to the input's size
    g = wf.build(_params(wf, prompt="x", denoise=0.5, images={"image": "in.png"}))
    _check_graph(g)
    assert "SetLatentNoiseMask" not in _classes(g) and g["split"]["inputs"]["denoise"] == 0.5
    assert g[g["save"]["inputs"]["images"][0]]["class_type"] == "ResizeImageMaskNode"


def test_pick_with_input_image():
    assert workflows.pick("image", slots={"image"}).name == "z-image-turbo"
    assert workflows.pick("image", slots={"image", "mask"}).name == "z-image-turbo"
    assert workflows.pick("image", slots={"image", "mask", "ref1"}).name == "flux2-klein-4b"
    assert workflows.pick("image", "flux2-klein-4b", {"image"}).name == "flux2-klein-4b"


def test_edit_parameters_validated():
    from localgen_mcp.workflows.image_edit import edit_images
    assert edit_images(None, None, ["a.png"], None, None, None) == {"ref1": "a.png"}
    assert edit_images("in.png", "m.png", ["a.png"], 0.5, None, None) == {"ref1": "a.png", "image": "in.png",
                                                                          "mask": "m.png"}
    with pytest.raises(ValueError, match="mask_image needs input_image"):
        edit_images(None, "m.png", ["a.png"], None, None, None)
    with pytest.raises(ValueError, match="denoise needs input_image"):
        edit_images(None, None, [], 0.5, None, None)
    for bad in (0.0, 1.5, -1):
        with pytest.raises(ValueError, match="between"):
            edit_images("in.png", None, [], bad, None, None)
    with pytest.raises(ValueError, match="width and height"):
        edit_images("in.png", None, [], None, 512, None)
    with pytest.raises(ValueError, match="at most 3"):
        edit_images("in.png", None, ["a", "b", "c", "d"], None, None, None)
    with pytest.raises(ValueError, match="at most 4"):
        edit_images(None, None, ["a", "b", "c", "d", "e"], None, None, None)
