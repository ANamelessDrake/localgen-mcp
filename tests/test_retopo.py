"""retopologize_mesh: request validation, the Blender command, and (when Blender
is installed) a real run on a textured fixture for both methods."""

from __future__ import annotations

import asyncio
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
                                         "quad", "1024", sys.executable]


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
