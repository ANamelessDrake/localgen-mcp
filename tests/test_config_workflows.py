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


def test_only_real_meshes_are_uploaded(tmp_path):
    from localgen_mcp.jobs import is_mesh_file
    glb = tmp_path / "a.glb"
    glb.write_bytes(b"glTF" + b"\0" * 16)
    obj = tmp_path / "b.obj"
    obj.write_text("# made by hand\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    stl = tmp_path / "c.stl"
    stl.write_bytes(b"\0" * 80 + (1).to_bytes(4, "little") + b"\0" * 50)
    assert is_mesh_file(glb) and is_mesh_file(obj) and is_mesh_file(stl)
    fake_glb = tmp_path / "secret.glb"
    fake_glb.write_text("not a model")
    fake_obj = tmp_path / "notes.obj"
    fake_obj.write_text("password=hunter2\n")
    fake_stl = tmp_path / "id_rsa.stl"
    fake_stl.write_bytes(b"-----BEGIN OPENSSH PRIVATE KEY-----" + b"x" * 200)
    assert not any(is_mesh_file(f) for f in (fake_glb, fake_obj, fake_stl))
