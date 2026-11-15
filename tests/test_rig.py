"""rig_creature: request validation and, when Blender is installed, real rigs of synthetic creatures
checked by posing bones and measuring the deformed mesh (Armature modifier applied)."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from localgen_mcp import config, rig

HERE = Path(__file__).parent
BLENDER = shutil.which("blender")
needs_blender = pytest.mark.skipif(BLENDER is None, reason="Blender is not installed")


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALGEN_ROOT", str(tmp_path))
    monkeypatch.setenv("LOCALGEN_BLENDER_THREADS", "4")
    return config.load()


def glb_stub(path: Path, doc: dict | None = None) -> Path:
    raw = json.dumps(doc or {"asset": {"version": "2.0"}}).encode()
    raw += b" " * (-len(raw) % 4)
    path.write_bytes(b"glTF" + (2).to_bytes(4, "little") + (20 + len(raw)).to_bytes(4, "little")
                     + len(raw).to_bytes(4, "little") + b"JSON" + raw)
    return path


def test_rejects_files_that_are_not_self_contained_glbs(tmp_path):
    fake = tmp_path / "secret.glb"
    fake.write_text("not a mesh")
    with pytest.raises(rig.RigError, match="binary glTF"):
        rig.build_spec(str(fake))
    renamed = glb_stub(tmp_path / "mesh.obj")
    with pytest.raises(rig.RigError, match="binary glTF"):
        rig.build_spec(str(renamed))
    linked = glb_stub(tmp_path / "linked.glb", {"asset": {"version": "2.0"},
                                                "images": [{"uri": "/etc/passwd"}]})
    with pytest.raises(rig.RigError, match="external files"):
        rig.build_spec(str(linked))
    with pytest.raises(rig.RigError, match="not found"):
        rig.build_spec(str(tmp_path / "missing.glb"))


@pytest.mark.parametrize("kwargs, match", [
    ({"preset": "skeleton"}, "preset"),
    ({"weights": "heat"}, "weights"),
    ({"bones": 1}, "bones"),
    ({"bones": 65}, "bones"),
    ({"bones": True}, "bones"),
    ({"wing_chains": 0}, "wing_chains"),
    ({"wing_bones": 17}, "wing_bones"),
    ({"axis": "w"}, "axis"),
    ({"axis": [0, 0, 0]}, "zero"),
    ({"axis": [1, "a", 0]}, "three numbers"),
    ({"root_end": "top"}, "root_end"),
    ({"root_end": [0, float("nan"), 0]}, "finite"),
    ({"preset": "wings", "name": "fin"}, "chain"),
    ({"name": "!!!"}, "letters"),
])
def test_rejects_bad_arguments(tmp_path, kwargs, match):
    glb = glb_stub(tmp_path / "a.glb")
    with pytest.raises(rig.RigError, match=match):
        rig.build_spec(str(glb), **kwargs)


def test_spec_normalises_arguments(tmp_path):
    glb = glb_stub(tmp_path / "a.glb")
    spec = rig.build_spec(str(glb), "chain", 5, "Z", "auto", name="vine 1")
    assert spec["axis"] == "z" and spec["root_end"] is None and spec["name"] == "vine_1" and spec["bones"] == 5
    assert rig.build_spec(str(glb), root_end=[1, 2, 3])["root_end"] == [1.0, 2.0, 3.0]
    glb_out, fbx_out = rig.output_paths(glb, "wings", str(tmp_path / "out"))
    assert glb_out == tmp_path / "out" / "a_rig_wings.glb" and fbx_out.suffix == ".fbx"


# -- real Blender runs ---------------------------------------------------------


@pytest.fixture(scope="module")
def creatures(tmp_path_factory):
    if BLENDER is None:
        pytest.skip("Blender is not installed")
    out = tmp_path_factory.mktemp("creatures")
    subprocess.run([BLENDER, "-b", "--factory-startup", "-P", str(HERE / "blender_make_creatures.py"), "--", str(out)],
                   check=True, capture_output=True, timeout=300)
    return out


def pose(model: str, out: Path, bones: dict) -> dict:
    """Pose bones (degrees about local X) in a fresh Blender and return the measured skin."""
    subprocess.run([BLENDER, "-b", "--factory-startup", "-P", str(HERE / "blender_pose_check.py"), "--",
                    model, str(out), json.dumps(bones)], check=True, capture_output=True, timeout=300)
    res = json.loads(out.read_text())
    if "rest" in res:
        res["rest"], res["posed"], res["moved"] = (np.array(res[k]) for k in ("rest", "posed", "moved"))
    return res


def run_rig(settings, src: Path, out: Path, **kwargs) -> dict:
    return asyncio.run(rig.rig_creature(settings, str(src), output_dir=str(out), **kwargs))


def assert_bends_from_the_root(res: dict, bone: str, along: np.ndarray):
    """Posing `bone` leaves the mesh before its head in place and moves it more the further along it is."""
    head = np.array(res["bones"][bone]["head"])
    t = (res["rest"] - head) @ along  # position along the chain, from the posed bone's head
    length = t.max()
    before = res["moved"][t < -0.15 * length]
    far = res["moved"][t > 0.8 * length]
    assert before.max() < 0.02 * length, before.max()
    assert far.min() > 0.25 * length, far.min()
    # every slice moves at least as much as the one before it (allowing a little noise)
    edges = np.linspace(0, length, 9)
    means = [res["moved"][(t >= a) & (t < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
    assert all(b >= a - 0.01 * length for a, b in zip(means, means[1:])), means


def check_skin(res: dict, names: list[str]):
    assert res["armatures"] == 1 and res["meshes"] == 1
    assert sorted(res["bones"]) == sorted(names)
    assert res["armature_modifier"] and set(names) <= set(res["groups"])
    assert res["weighted_share"] == 1.0 and res["max_influences"] <= 4


@needs_blender
def test_tentacle_chain_bends_progressively_in_glb_and_fbx(settings, creatures, tmp_path):
    rep = run_rig(settings, creatures / "tentacle.glb", tmp_path)
    assert rep["preset"] == "chain" and "long and thin" in rep["auto_reason"]
    names = [f"tentacle_{i:02d}" for i in range(1, 9)]
    assert [b["name"] for b in rep["bones"]] == names
    assert [b["parent"] for b in rep["bones"]] == [None] + names[:-1]
    assert rep["root"]["position"] == [0.0, 0.0, 0.0]  # the thick end is the base
    assert abs(rep["bones"][-1]["tail"][0] - 2.0) < 1e-3
    assert rep["weights"]["coverage"] == 1.0 and rep["weights"]["max_influences"] <= 4
    assert rep["weights"]["method"] == "heat"
    for model in rep["paths"]:  # the GLB and the FBX, re-imported
        res = pose(model, tmp_path / (Path(model).suffix[1:] + ".json"), {"tentacle_03": 40})
        check_skin(res, names)
        assert_bends_from_the_root(res, "tentacle_03", np.array([1.0, 0.0, 0.0]))


@needs_blender
def test_manta_wings_flap_from_the_body(settings, creatures, tmp_path):
    rep = run_rig(settings, creatures / "manta.glb", tmp_path, wing_chains=3, wing_bones=4)
    assert rep["preset"] == "wings"
    assert rep["lateral_axis"] == [1.0, 0.0, 0.0] and rep["normal_axis"] == [0.0, 0.0, 1.0]
    names = [b["name"] for b in rep["bones"]]
    assert names[:6] == [f"spine_{i:02d}" for i in range(1, 7)]
    for side in "LR":
        assert [n for n in names if n.startswith(f"wing_{side}")] == \
            [f"wing_{side}{k}_{j:02d}" for k in (1, 2, 3) for j in (1, 2, 3, 4)]
    bones = {b["name"]: b for b in rep["bones"]}
    assert bones["wing_L2_04"]["tail"][0] == pytest.approx(1.0, abs=0.02)  # the wing tip
    assert bones["wing_R2_04"]["tail"][0] == pytest.approx(-1.0, abs=0.02)
    assert bones["wing_L2_01"]["parent"].startswith("spine_")
    assert rep["weights"]["coverage"] == 1.0
    res = pose(rep["paths"][1], tmp_path / "fbx.json", {"wing_L2_01": 30, "wing_R2_01": 30})
    check_skin(res, names)
    x, z = res["rest"][:, 0], res["rest"][:, 2]
    dz = res["posed"][:, 2] - z
    for sign in (1, -1):  # both wings flap up for a positive angle
        tip = dz[sign * x > 0.85]
        near = res["moved"][(sign * x > 0) & (sign * x < 0.1)]
        assert tip.min() > 0.25, tip.min()
        assert near.max() < 0.25 * tip.min(), (near.max(), tip.min())
    res = pose(rep["paths"][0], tmp_path / "glb.json", {"wing_L2_01": 30})
    left_tip = res["moved"][x > 0.85]
    right_wing = res["moved"][x < -0.3]
    assert left_tip.min() > 0.25 and right_wing.max() < 0.02  # the other wing stays put


@needs_blender
def test_failed_heat_weights_fall_back_to_distance_weights(settings, creatures, tmp_path):
    """Heat weighting fails outright on nested shells (as on real generated meshes): every vertex must still
    be weighted, and the chain must still bend from its root."""
    rep = run_rig(settings, creatures / "nested.glb", tmp_path, preset="chain")
    w = rep["weights"]
    assert w["heat"]["unweighted_share"] > 0.5, w
    assert w["method"].startswith("distance"), w
    assert w["coverage"] == 1.0 and w["unweighted_vertices"] == 0 and w["max_influences"] <= 4
    res = pose(rep["paths"][1], tmp_path / "fbx.json", {"tentacle_03": 40})
    check_skin(res, [b["name"] for b in rep["bones"]])
    tube = np.abs(res["rest"][:, 1]) < 0.2  # leave out the spheres beside the tube
    res["rest"], res["moved"] = res["rest"][tube], res["moved"][tube]
    assert_bends_from_the_root(res, "tentacle_03", np.array([1.0, 0.0, 0.0]))


@needs_blender
def test_heat_weights_on_a_far_bone_are_rejected(settings, creatures, tmp_path):
    rep = run_rig(settings, creatures / "shelled.glb", tmp_path, preset="chain")
    w = rep["weights"]
    assert w["heat"]["misassigned_share"] > 0.1 and w["method"].startswith("distance"), w
    res = pose(rep["paths"][0], tmp_path / "glb.json", {"tentacle_03": 40})
    assert_bends_from_the_root(res, "tentacle_03", np.array([1.0, 0.0, 0.0]))


@needs_blender
def test_explicit_options(settings, creatures, tmp_path):
    rep = run_rig(settings, creatures / "tentacle.glb", tmp_path, preset="chain", bones=4, root_end="max",
                  weights="distance", name="vine")
    assert [b["name"] for b in rep["bones"]] == ["vine_01", "vine_02", "vine_03", "vine_04"]
    assert rep["bones"][0]["head"][0] == pytest.approx(2.0, abs=1e-3)  # rooted at the thin end
    assert rep["weights"]["method"] == "distance" and rep["weights"]["heat"] is None
    rep = run_rig(settings, creatures / "manta.glb", tmp_path, preset="spine", axis="y", root_end=[0, 1, 0])
    assert rep["bones"][0]["head"][1] == pytest.approx(0.6, abs=1e-3)
    assert rep["bones"][-1]["tail"][1] == pytest.approx(-0.6, abs=1e-3)
