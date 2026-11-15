"""retopologize_mesh: request validation, the Blender command, and (when Blender
is installed) a real run on a textured fixture for both methods."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from localgen_mcp import config, retopo

HERE = Path(__file__).parent


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALGEN_ROOT", str(tmp_path))
    monkeypatch.setenv("LOCALGEN_BLENDER_THREADS", "4")
    return config.load()


def test_rejects_non_glb(tmp_path):
    fake = tmp_path / "secret.glb"
    fake.write_text("not a mesh")
    with pytest.raises(retopo.RetopoError, match="binary glTF"):
        retopo.validate(str(fake), 10000, "quad", 2048)


@pytest.mark.parametrize("faces, method, tex", [(50, "quad", 2048), (10000, "smooth", 2048),
                                                (10000, "quad", 100000)])
def test_rejects_bad_arguments(tmp_path, faces, method, tex):
    glb = tmp_path / "a.glb"
    glb.write_bytes(b"glTF" + b"\0" * 16)
    with pytest.raises(retopo.RetopoError):
        retopo.validate(str(glb), faces, method, tex)


def test_command_runs_the_bundled_script_with_this_interpreter(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(retopo.blender_run.shutil, "which", lambda name: "/usr/bin/true")
    cmd = retopo.command(settings, tmp_path / "in.glb", tmp_path / "out.glb", 8000, "quad", 1024)
    assert cmd[0] == "/usr/bin/true"
    assert cmd[cmd.index("-t") + 1] == "4"
    assert Path(cmd[cmd.index("-P") + 1]) == retopo.SCRIPT and retopo.SCRIPT.is_file()
    assert cmd[cmd.index("--") + 1:] == [str(tmp_path / "in.glb"), str(tmp_path / "out.glb"), "8000",
                                         "quad", "1024", sys.executable, "none", "1.0"]
    cmd = retopo.command(settings, tmp_path / "in.glb", tmp_path / "out.glb", 8000, "decimate", 1024)
    assert cmd[-2:] == ["auto", "1.0"]
    cmd = retopo.command(settings, tmp_path / "in.glb", tmp_path / "out.glb", 8000, "decimate", 1024, "paint", 0.3)
    assert cmd[-2:] == ["paint", "0.3"]


def test_report_is_parsed_from_the_last_marker_line():
    out = 'noise\nRETOPO_RESULT {"output_faces": 1}\nmore noise\nRETOPO_RESULT {"output_faces": 2}\n'
    assert retopo.parse_report(out) == {"output_faces": 2}
    with pytest.raises(retopo.RetopoError):
        retopo.parse_report("no report here")


BLENDER = shutil.which("blender")


@pytest.mark.skipif(BLENDER is None, reason="Blender is not installed")
@pytest.mark.parametrize("method", ["decimate", "quad"])
def test_real_retopology_keeps_colour_and_hits_the_budget(settings, tmp_path, method):
    src = tmp_path / "fixture.glb"
    subprocess.run([BLENDER, "-b", "--factory-startup", "-P", str(HERE / "blender_make_textured.py"),
                    "--", str(src)], check=True, capture_output=True, timeout=300)
    report = asyncio.run(retopo.retopologize(settings, str(src), 600, method, 256, str(tmp_path / "out")))
    assert Path(report["path"]).is_file()
    assert Path(report["fbx"]).is_file() and Path(report["fbx"]).stat().st_size > 10000
    assert report["input_faces"] > 3000
    assert 300 <= report["output_faces"] <= 900
    if method == "quad":
        assert report["method_used"].startswith("quad")
        assert report["output_quads"] > 0.9 * report["output_faces"]
    # The bake must carry the base colour across unchanged (sRGB 200, 60, 30):
    # a double sRGB encode gives about (228, 137, 101), a missing one about (148, 10, 3).
    assert report["base_color_median"]["source"] == [200, 60, 30]
    baked = report["base_color_median"]["baked"]
    assert all(abs(b - s) <= 6 for b, s in zip(baked, (200, 60, 30))), baked


@pytest.mark.skipif(BLENDER is None, reason="Blender is not installed")
def test_texture_input_is_reserialised_without_foreign_bytes(settings, tmp_path):
    """texture_mesh uploads Blender's re-export, never the caller's file: data hidden in an input
    (here past the first 64 KB of a valid OBJ) must not reach the GLB that gets uploaded."""
    from localgen_mcp import blender_run
    secret = "AKIA-NOT-A-REAL-KEY-0123456789"
    verts = "".join(f"v {i % 7} {i % 5} {i % 3}\n" for i in range(6000))  # well over 64 KB
    src = tmp_path / "model.obj"
    src.write_text(verts + "f 1 2 3\nf 4 5 6\n# " + secret + "\n")
    dst = tmp_path / "clean.glb"
    report = asyncio.run(blender_run.run(settings, "convert_mesh.py", [src, dst], "CONVERT_RESULT", dst))
    assert report["faces"] == 2
    data = dst.read_bytes()
    assert data[:4] == b"glTF" and secret.encode() not in data


@pytest.mark.skipif(BLENDER is None, reason="Blender is not installed")
def test_texture_input_drops_materials_and_referenced_images(settings, tmp_path):
    """An OBJ's .mtl can name any image on disk as a texture; the re-export must not carry it."""
    from localgen_mcp import blender_run
    elsewhere = tmp_path / "private"
    elsewhere.mkdir()
    marker = b"PRIVATE-PHOTO-BYTES-0123456789"
    # a valid PNG (so Blender would load it) carrying a marker in a text chunk
    import struct, zlib
    def chunk(k, d):
        return struct.pack(">I", len(d)) + k + d + struct.pack(">I", zlib.crc32(k + d) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
           + chunk(b"tEXt", b"Comment\0" + marker) + chunk(b"IDAT", zlib.compress(b"\0" + b"\xff" * 6 + b"\0" + b"\xff" * 6))
           + chunk(b"IEND", b""))
    (elsewhere / "photo.png").write_bytes(png)
    (tmp_path / "model.mtl").write_text(f"newmtl m\nmap_Kd {elsewhere / 'photo.png'}\n")
    src = tmp_path / "model.obj"
    src.write_text("mtllib model.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl m\nf 1/1 2/2 3/3\n")
    dst = tmp_path / "clean.glb"
    asyncio.run(blender_run.run(settings, "convert_mesh.py", [src, dst], "CONVERT_RESULT", dst))
    data = dst.read_bytes()
    assert marker not in data and b"\x89PNG" not in data
    doc = json.loads(data[20:20 + int.from_bytes(data[12:16], "little")])
    assert not doc.get("images") and not doc.get("materials")
    assert doc["meshes"][0]["primitives"][0]["attributes"].get("TEXCOORD_0") is not None  # UVs kept


@pytest.mark.skipif(BLENDER is None, reason="Blender is not installed")
def test_painted_detail_puts_the_faces_where_painted(settings, tmp_path):
    src = tmp_path / "painted.glb"
    subprocess.run([BLENDER, "-b", "--factory-startup", "-P", str(HERE / "blender_make_painted.py"), "--", str(src)],
                   check=True, capture_output=True, timeout=300)
    uniform = asyncio.run(retopo.retopologize(settings, str(src), 1500, "decimate", 256, str(tmp_path / "u"),
                                              detail="none"))
    painted = asyncio.run(retopo.retopologize(settings, str(src), 1500, "decimate", 256, str(tmp_path / "p"),
                                              detail="paint"))
    assert "detail" not in uniform
    d = painted["detail"]
    assert d["source"] == "paint" and 0.4 < d["share_of_vertices_marked_detailed"] < 0.6
    # Half the surface is painted white; uniform decimation would put about half the faces there.
    assert d["share_of_faces_in_detailed_areas"] > 0.65, d
    assert abs(painted["output_faces"] - 1500) <= 0.05 * 1500


@pytest.mark.skipif(BLENDER is None, reason="Blender is not installed")
def test_parts_are_placed_by_their_boxes_and_rigged_at_the_joint(settings, tmp_path):
    from localgen_mcp import blender_run
    subprocess.run([BLENDER, "-b", "--factory-startup", "-P", str(HERE / "blender_make_parts.py"), "--", str(tmp_path)],
                   check=True, capture_output=True, timeout=300)
    spec = {"concept": str(tmp_path / "concept.png"), "full_mesh": str(tmp_path / "full.glb"), "rig": True,
            "out_glb": str(tmp_path / "out.glb"), "out_fbx": str(tmp_path / "out.fbx"),
            "parts": [{"name": "body", "mesh": str(tmp_path / "body.glb"), "box": [100, 50, 300, 450]},
                      {"name": "arm", "mesh": str(tmp_path / "arm.glb"), "box": [300, 90, 380, 290]}]}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    rep = asyncio.run(blender_run.run(settings, "assemble_parts.py", [tmp_path / "spec.json"], "ASSEMBLE_RESULT",
                                      tmp_path / "out.glb"))
    assert rep["root"] == "body" and rep["parts"]["arm"]["parent"] == "body" and not rep["warnings"]
    (ax0, ay0, az0), (ax1, ay1, az1) = rep["parts"]["arm"]["bbox"]
    # placed where the box says, within a pixel or two of the 400 px per unit drawing
    for got, want in ((ax0, 0.25), (ax1, 0.45), (az0, 0.4), (az1, 0.9)):
        assert abs(got - want) < 0.015, rep["parts"]["arm"]
    assert abs((ay0 + ay1) / 2) < 0.03  # depth from the full mesh
    px, _, pz = rep["parts"]["arm"]["pivot"]
    assert abs(px - 0.25) < 0.03 and 0.4 < pz < 0.9  # the joint is the shoulder face
    assert (tmp_path / "out.fbx").stat().st_size > 10000


# -- LOD chains -----------------------------------------------------------


@pytest.mark.parametrize("lods", [[], [3000, 3000], [3000, 8000], [10000, 100], [10000, 5000.5], "10000,3000",
                                  [True], list(range(9000, 0, -1000)), [600_000, 3000]])
def test_rejects_bad_lods(lods):
    with pytest.raises(retopo.RetopoError):
        retopo.validate_lods(lods)


def test_accepts_lods():
    assert retopo.validate_lods(None) is None
    assert retopo.validate_lods([10000, 3000.0, 800]) == [10000, 3000, 800]


def test_lod_object_names_are_safe():
    assert retopo.lod_object_name(Path("/x/199b74d60b_00001_.glb")) == "199b74d60b_00001"
    assert retopo.lod_object_name(Path("/x/my robot (v2).glb")) == "my_robot_v2"
    assert retopo.lod_object_name(Path("/x/---.glb")) == "mesh"


def test_command_passes_lods_and_the_object_name(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(retopo.blender_run.shutil, "which", lambda name: "/usr/bin/true")
    cmd = retopo.command(settings, tmp_path / "bot.glb", tmp_path / "out.glb", 8000, "quad", 1024,
                         lods=[8000, 2000, 500])
    assert cmd[-4:] == ["none", "1.0", "8000,2000,500", "bot"]


def test_generate_mesh_checks_lods_before_any_gpu_work(monkeypatch):
    from localgen_mcp import server

    async def boom(*a, **k):
        raise AssertionError("the workflow ran before lods were checked")

    monkeypatch.setattr(server.jobs, "run_workflow", boom)
    with pytest.raises(retopo.RetopoError):
        asyncio.run(server.generate_mesh(input_image="/nonexistent.png", retopo_lods=[3000, 9000]))


def glb_json(path: Path) -> dict:
    data = path.read_bytes()
    return json.loads(data[20:20 + int.from_bytes(data[12:16], "little")])


def inspect_lods(path: Path) -> dict:
    proc = subprocess.run([BLENDER, "-b", "--factory-startup", "-P", str(HERE / "blender_inspect_lods.py"), "--",
                           str(path)], check=True, capture_output=True, text=True, timeout=300)
    return retopo.blender_run.parse_report(proc.stdout, "LODS")


@pytest.mark.skipif(BLENDER is None, reason="Blender is not installed")
@pytest.mark.parametrize("method", ["decimate", "quad"])
def test_lod_chain_shares_one_uv_layout_and_material(settings, tmp_path, method):
    src = tmp_path / "crate.glb"
    subprocess.run([BLENDER, "-b", "--factory-startup", "-P", str(HERE / "blender_make_textured.py"),
                    "--", str(src)], check=True, capture_output=True, timeout=300)
    lods = [1500, 500, 200]
    report = asyncio.run(retopo.retopologize(settings, str(src), 99999, method, 256, str(tmp_path / "out"),
                                             lods=lods))
    glb, fbx = Path(report["path"]), Path(report["fbx"])
    assert glb.name == f"crate_retopo_{method}_lods_1500_500_200.glb" and fbx.is_file()
    names = [f"crate_LOD{i}" for i in range(3)]
    assert [e["name"] for e in report["lods"]] == names
    assert report["target_faces"] == 1500 and report["materials"] == ["retopo"]
    assert abs(report["output_faces"] - 1500) <= 0.1 * 1500  # LOD0 is the retopology itself
    for entry, budget in zip(report["lods"][1:], lods[1:]):
        assert abs(entry["triangles"] - budget) <= 0.05 * budget, entry
        fit = entry["texture_fit"]  # LOD0's layout, not a new one (a fresh unwrap measures about 100)
        assert fit["uv_drift_texels_median"] < 3 and fit["colour_mismatch_share"] < 0.02, entry
    # LOD0 keeps the base colour bake (the single output's colour check still holds)
    baked = report["base_color_median"]["baked"]
    assert all(abs(b - s) <= 6 for b, s in zip(baked, (200, 60, 30))), baked

    # The GLB: one node per LOD, one material and one texture set referenced by all of them.
    doc = glb_json(glb)
    assert sorted(n["name"] for n in doc["nodes"]) == names
    assert len(doc["materials"]) == 1 and len(doc["images"]) <= 4
    assert all(p["material"] == 0 for m in doc["meshes"] for p in m["primitives"])
    tris = {n["name"]: sum(doc["accessors"][p["indices"]]["count"] // 3 for p in doc["meshes"][n["mesh"]]["primitives"])
            for n in doc["nodes"]}
    assert [tris[n] for n in names] == [e["triangles"] for e in report["lods"]]

    # Re-imported from both files: the LOD objects, one shared material, and LOD0's UVs on every LOD.
    for path in (glb, fbx):
        seen = inspect_lods(path)
        objs = {o["name"]: o for o in seen["objects"]}
        assert sorted(objs) == names, seen
        assert seen["materials"] == ["retopo"] and all(o["materials"] == ["retopo"] for o in objs.values())
        for name in names[1:]:
            assert objs[name]["uv_error_median"] * 256 < 3, (path.suffix, objs[name])
        if path == fbx and method == "quad":
            assert objs[names[0]]["faces"] == report["lods"][0]["faces"]  # LOD0 keeps its quads


@pytest.mark.skipif(BLENDER is None, reason="Blender is not installed")
def test_lods_follow_the_painted_detail_map(settings, tmp_path):
    src = tmp_path / "painted.glb"
    subprocess.run([BLENDER, "-b", "--factory-startup", "-P", str(HERE / "blender_make_painted.py"), "--", str(src)],
                   check=True, capture_output=True, timeout=300)
    report = asyncio.run(retopo.retopologize(settings, str(src), 0, "decimate", 256, str(tmp_path / "p"),
                                             detail="paint", lods=[3000, 1000, 300]))
    # Half the surface is painted white. LOD0 puts about 0.9 of its faces there; a uniform decimation
    # of LOD0 drifts back toward a half (measured 0.78 and 0.63), the weighted one holds (0.91, 0.81).
    shares = [e["share_of_faces_in_detailed_areas"] for e in report["lods"][1:]]
    assert shares[0] > 0.85 and shares[1] > 0.72, shares
