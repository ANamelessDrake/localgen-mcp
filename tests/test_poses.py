"""Pose library and pose guides: skeleton mapping, clip building, retargeting, projection, key
frames, guide drawing, the Blender baker and the generate_animation wiring.

Every clip here is synthetic (pose_fixtures.py, or tests/blender_make_walker.py inside Blender).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pose_fixtures as pf
from localgen_mcp import config, pixel_ops as po, pose_math as pm, poses, sprites

HERE = Path(__file__).resolve().parent
BLENDER = shutil.which("blender")
needs_blender = pytest.mark.skipif(BLENDER is None, reason="Blender is not installed")


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALGEN_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("LOCALGEN_POSE_DIR", str(tmp_path / "poses"))
    return config.load()


def save_clip(clip: dict, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clip))
    return str(path)


def library_with(settings, clip: dict) -> Path:
    """A pose library holding one clip, indexed as bake_pose_library would."""
    lib = settings.pose_library
    lib.mkdir(parents=True, exist_ok=True)
    save_clip(clip, lib / f"{clip['name']}.json")
    m = clip["metadata"]
    poses.write_index(lib, {"format": poses.LIBRARY_FORMAT, "clips": {clip["name"]: {
        "file": f"{clip['name']}.json", "action": m["action"], "loop": m["loop"], "frames": clip["frame_count"],
        "fps": clip["fps"], "impact_frame": m["impact_frame"], "weapon_hand": m["weapon_hand"],
        "key_frames": m["key_frames"], "source_file": "synthetic"}}})
    return lib


# -- skeleton mapping --------------------------------------------------------

LIMBS = {
    "B- prefix": ("B-hips", "B-spine", "B-chest", "B-neck", "B-head", "B-shoulder.{S}", "B-upperArm.{S}",
                  "B-forearm.{S}", "B-hand.{S}", "B-thigh.{S}", "B-shin.{S}", "B-foot.{S}", "B-toe.{S}"),
    "unreal": ("pelvis", "spine_01", "spine_03", "neck_01", "head", "clavicle_{s}", "upperarm_{s}", "lowerarm_{s}",
               "hand_{s}", "thigh_{s}", "calf_{s}", "foot_{s}", "ball_{s}"),
    "mixamo": ("mixamorig:Hips", "mixamorig:Spine", "mixamorig:Spine2", "mixamorig:Neck", "mixamorig:Head",
               "mixamorig:{Side}Shoulder", "mixamorig:{Side}Arm", "mixamorig:{Side}ForeArm", "mixamorig:{Side}Hand",
               "mixamorig:{Side}UpLeg", "mixamorig:{Side}Leg", "mixamorig:{Side}Foot", "mixamorig:{Side}ToeBase"),
    "unity": ("Hips", "Spine", "UpperChest", "Neck", "Head", "{Side}Shoulder", "{Side}UpperArm", "{Side}LowerArm",
              "{Side}Hand", "{Side}UpperLeg", "{Side}LowerLeg", "{Side}Foot", "{Side}Toes"),
    "biped": ("Bip01 Pelvis", "Bip01 Spine", "Bip01 Spine2", "Bip01 Neck", "Bip01 Head", "Bip01 {S} Clavicle",
              "Bip01 {S} UpperArm", "Bip01 {S} Forearm", "Bip01 {S} Hand", "Bip01 {S} Thigh", "Bip01 {S} Calf",
              "Bip01 {S} Foot", "Bip01 {S} Toe0"),
}


def skeleton(names: tuple, extra_spine: str | None = None) -> list[tuple[str, str | None]]:
    """(name, parent) pairs of a humanoid in one naming, with twist and finger bones as distractors."""
    pelvis, spine, chest, neck, head, clav, up, fore, hand, thigh, shin, foot, toe = names
    out = [("Root", None), (pelvis, "Root"), (spine, pelvis)]
    top = spine
    if extra_spine:
        out.append((extra_spine, spine))
        top = extra_spine
    out += [(chest, top), (neck, chest), (head, neck)]
    for side, s, S in (("Left", "l", "L"), ("Right", "r", "R")):
        f = lambda n: n.format(Side=side, s=s, S=S)  # noqa: E731
        out += [(f(clav), chest), (f(up), f(clav)), (f(fore), f(up)), (f(hand), f(fore)),
                (f(up) + "_twist", f(up)), (f(hand) + "_index1", f(hand)),
                (f(thigh), pelvis), (f(shin), f(thigh)), (f(foot), f(shin)), (f(toe), f(foot))]
    return out


@pytest.mark.parametrize("scheme", sorted(LIMBS))
def test_every_naming_maps_to_all_joints(scheme):
    names = LIMBS[scheme]
    m = pm.map_skeleton(skeleton(names, extra_spine="SpineMid"))
    assert list(m["joints"]) == pm.NAMES
    b = m["bones"]
    fmt = lambda n, side: n.format(Side={"l": "Left", "r": "Right"}[side], s=side, S=side.upper())  # noqa: E731
    for side in "lr":
        assert b[f"elbow_{side}"] == fmt(names[7], side)
        assert b[f"knee_{side}"] == fmt(names[10], side)
        assert b[f"toe_{side}"] == fmt(names[12], side)
        # the shoulder point is the upper arm's head; the clavicle turns the segment that ends there
        assert m["joints"][f"shoulder_{side}"]["edge"] == fmt(names[5], side)
    assert b["pelvis"] == names[0] and b["chest"] == names[2] and b["spine"] == names[1]
    assert m["joints"]["head_top"]["point"][0] == "offset"


def test_cmu_bvh_names_map():
    bones = [("Hips", None), ("LHipJoint", "Hips"), ("LeftUpLeg", "LHipJoint"), ("LeftLeg", "LeftUpLeg"),
             ("LeftFoot", "LeftLeg"), ("LeftToeBase", "LeftFoot"), ("RHipJoint", "Hips"), ("RightUpLeg", "RHipJoint"),
             ("RightLeg", "RightUpLeg"), ("RightFoot", "RightLeg"), ("RightToeBase", "RightFoot"),
             ("LowerBack", "Hips"), ("Spine", "LowerBack"), ("Spine1", "Spine"), ("Neck", "Spine1"),
             ("Neck1", "Neck"), ("Head", "Neck1"), ("LeftShoulder", "Spine1"), ("LeftArm", "LeftShoulder"),
             ("LeftForeArm", "LeftArm"), ("LeftHand", "LeftForeArm"), ("LThumb", "LeftHand"),
             ("RightShoulder", "Spine1"), ("RightArm", "RightShoulder"), ("RightForeArm", "RightArm"),
             ("RightHand", "RightForeArm")]
    m = pm.map_skeleton(bones)
    assert m["bones"]["hip_l"] == "LeftUpLeg" and m["bones"]["neck"] == "Neck"
    assert m["bones"]["spine"] == "LowerBack" and m["bones"]["chest"] == "Spine1"
    assert m["joints"]["hand_l"]["point"] == ["tail", "LeftHand"]  # no middle finger: the hand bone's tail


def test_split_name_sides():
    assert pm.split_name("B-upperArm.R") == ("r", "upperarm")
    assert pm.split_name("mixamorig:LeftForeArm") == ("l", "forearm")
    assert pm.split_name("calf_l") == ("l", "calf")
    assert pm.split_name("Bip01 L Thigh") == ("l", "thigh")
    assert pm.split_name("LHipJoint") == ("l", "hipjoint")
    assert pm.split_name("neck_01") == (None, "neck01")


def test_missing_limbs_are_named():
    bones = [b for b in skeleton(LIMBS["unity"]) if not b[0].startswith("Right")]
    with pytest.raises(pm.PoseError, match=r"upper arm \(right\).*thigh \(right\)"):
        pm.map_skeleton(bones)


# -- building clips ------------------------------------------------------------


def test_clip_records_the_rotations_it_was_made_from():
    clip = pf.synthetic_clip()
    local = np.asarray(clip["frames"]["local_rotations"])
    truth = pf.local_rotations()
    same = np.abs(np.sum(local * truth, -1))  # |dot| = 1 for the same rotation
    assert same[:, 1:].min() > 1 - 1e-6
    assert pm.qangle(local[pf.ELBOW_FRAME, pm.INDEX["wrist_l"]]) == pytest.approx(90.0, abs=0.01)
    assert pm.qangle(local[0, pm.INDEX["wrist_l"]]) == pytest.approx(0.0, abs=0.01)


def test_forward_kinematics_gives_back_the_positions():
    clip = pf.synthetic_clip()
    a = pm.arrays(clip)
    pos, _ = pm.forward(a["offsets"], a["local"], a["root_position"])
    assert np.abs(pos - np.asarray(clip["frames"]["positions"])).max() < 1e-4


@pytest.mark.parametrize("yaw", [90.0, 180.0, -35.0])
def test_facing_floor_and_root_motion_are_normalised(yaw):
    base = pf.synthetic_clip()
    turned = pf.synthetic_clip(yaw=yaw, travel=1.2, unit=0.01)  # turned, walking, in centimetres
    p0, p1 = np.asarray(base["frames"]["positions"]), np.asarray(turned["frames"]["positions"])
    assert np.abs(p0 - p1).max() < 1e-3
    hip = p1[0, pm.INDEX["hip_l"]]
    assert hip[0] > 0.05 and abs(hip[1]) < 1e-3                      # left side at +X
    assert np.abs(p1[:, 0, :2]).max() < 1e-9                            # pelvis pinned over the origin
    assert turned["metadata"]["root_travel"] == pytest.approx(1.2, abs=1e-3)
    assert p1[:, [pm.INDEX[j] for j in pm.FOOT_JOINTS], 2].min() == pytest.approx(0.0, abs=1e-6)
    assert turned["rest"]["stature"] == pytest.approx(base["rest"]["stature"], rel=1e-4)


def test_root_motion_can_be_kept():
    clip = pm.build_clip(pf.synthetic_raw(travel=1.0), name="walker", root_motion="keep")
    pel = np.asarray(clip["frames"]["positions"])[:, 0]
    assert pel[0, 1] == pytest.approx(0.0, abs=1e-9) and pel[-1, 1] == pytest.approx(-1.0, abs=1e-6)


def test_metadata_finds_the_strike():
    meta = pf.synthetic_clip()["metadata"]
    assert meta["action"] == "attack" and meta["loop"] is False
    assert meta["weapon_hand"] == "right" and meta["weapon_hand_source"] == "motion"
    assert abs(meta["impact_frame"] - pf.STRIKE) <= 1
    labels = {k["label"]: k["frame"] for k in meta["key_frames"]}
    assert 2 <= labels["wind_up"] < meta["impact_frame"] and labels["recover"] > meta["impact_frame"]


def test_names_set_action_loop_and_hand():
    meta = pf.synthetic_clip(name="HammeringGround01_L - Loop")["metadata"]
    assert (meta["action"], meta["loop"], meta["weapon_hand"], meta["weapon_hand_source"]) == \
        ("hammer", True, "left", "name")
    assert pm.action_from_name("BlockShield01 - Hit") == "block"
    assert pm.action_from_name("HumanM@Death01") == "die"
    assert pm.action_from_name("Loco_Walk_Fwd") == "walk"


# -- retargeting ---------------------------------------------------------------


def test_retarget_keeps_rotations_and_changes_lengths():
    clip = pf.synthetic_clip()
    human = pm.retarget(clip, "human")
    stocky = pm.retarget(clip, "stocky")
    assert np.abs(np.abs(np.sum(human["world"] * stocky["world"], -1)) - 1).max() < 1e-9
    factors = pm.BODIES["stocky"]
    for group, joints in pm.GROUPS.items():
        for j in joints:
            i, p = pm.INDEX[j], pm.PARENTS[pm.INDEX[j]]
            a = np.linalg.norm(human["positions"][:, i] - human["positions"][:, p], axis=-1)
            b = np.linalg.norm(stocky["positions"][:, i] - stocky["positions"][:, p], axis=-1)
            assert np.allclose(b, a * factors.get(group, 1.0), rtol=1e-6), j
            # same direction every frame
            da = (human["positions"][:, i] - human["positions"][:, p]) / a[:, None]
            db = (stocky["positions"][:, i] - stocky["positions"][:, p]) / b[:, None]
            assert np.abs(da - db).max() < 1e-6, j
    feet = [pm.INDEX[j] for j in pm.FOOT_JOINTS]
    assert stocky["positions"][:, feet, 2].min() == pytest.approx(0.0, abs=1e-9)
    assert stocky["stature"] < human["stature"]


def test_custom_body_and_bad_bodies():
    b = pm.resolve_body({"base": "tall", "head": 2.0})
    assert b["head"] == 2.0 and b["legs"] == pm.BODIES["tall"]["legs"]
    for bad in ("giant", {"wings": 1.0}, {"legs": 9.0}, {"legs": True}, 3):
        with pytest.raises(pm.PoseError):
            pm.resolve_body(bad)


def test_exaggerate_scales_rotation_away_from_the_average():
    clip = pf.synthetic_clip()
    local = pm.arrays(clip)["local"]
    assert pm.exaggerate(local, 1.0) is local
    mean = pm.qmean(local, axis=0)
    twice = pm.exaggerate(local, 2.0)
    j = pm.INDEX["wrist_l"]
    before = pm.qangle(pm.qmul(pm.qconj(mean[j]), local[:, j]))
    after = pm.qangle(pm.qmul(pm.qconj(mean[j]), twice[:, j]))
    small = before < 80  # doubling stays under 180 degrees
    assert np.allclose(after[small], 2 * before[small], atol=1e-6)
    assert np.allclose(twice[:, 0], local[:, 0])  # root rotation kept


# -- camera --------------------------------------------------------------------


def test_projection_front_side_and_top_down():
    origin, s = (100.0, 200.0), 50.0
    pts = np.array([[0.1, 0.0, 1.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
    front = pm.project(pts, pm.camera("front", 0), s, origin)
    assert np.allclose(front[0, :2], [105.0, 150.0])          # left side (+X) on the image right
    assert front[1, 2] == pytest.approx(1.0)                   # forward is towards the camera
    left = pm.project(pts, pm.camera("left", 0), s, origin)
    assert np.allclose(left[1, :2], [50.0, 200.0])             # seen from its left, it faces image left
    right = pm.project(pts, pm.camera("right", 0), s, origin)
    assert np.allclose(right[1, :2], [150.0, 200.0])
    top = pm.project(pts, pm.camera("front", 90), s, origin)
    assert np.allclose(top[1, :2], [100.0, 250.0])             # straight down: forward is image down
    assert np.allclose(top[0, :2], [105.0, 200.0])             # height goes into depth only
    assert top[0, 2] == pytest.approx(1.0)
    rts = pm.project(pts, pm.camera("front", 45), s, origin)
    h = np.sqrt(0.5)
    assert np.allclose(rts[0, :2], [105.0, 200.0 - 50 * h])    # 1 m up shows as cos(45) m
    assert np.allclose(pm.camera(45.0, 10), pm.camera("front_left", 10))
    with pytest.raises(pm.PoseError):
        pm.camera("sideways", 0)
    with pytest.raises(pm.PoseError):
        pm.camera("front", 120)


# -- frame selection -------------------------------------------------------------


@pytest.mark.parametrize("n", range(2, 9))
def test_key_selection_always_includes_the_impact(n):
    meta = pf.synthetic_clip()["metadata"]
    sel = pm.select_frames(meta, pf.FRAMES, n)
    assert len(sel["frames"]) == n and sel["frames"] == sorted(sel["frames"])
    assert sel["frames"][sel["impact_index"]] == meta["impact_frame"]
    if n >= len(meta["key_frames"]):
        assert {k["frame"] for k in meta["key_frames"]} <= set(sel["frames"])


def test_loops_are_spread_evenly_without_the_repeated_last_frame():
    meta = {"loop": True, "last_frame_repeats_first": True, "key_frames": [{"frame": 0, "label": "start"}]}
    assert pm.select_frames(meta, 25, 6)["frames"] == [0, 4, 8, 12, 16, 20]
    one_shot = {"loop": False, "key_frames": []}
    assert pm.select_frames(one_shot, 21, 5)["frames"] == [0, 5, 10, 15, 20]
    assert pm.select_frames(one_shot, 21, 3, "keys")["frames"] == [0, 10, 20]


# -- guide drawing -----------------------------------------------------------------


def guide_spec(clip_path: str, outdir: Path, n: int = 4, **kw) -> dict:
    return {"op": "pose_guides", "clip": clip_path, "frames": n, "size": [256, 256],
            "outputs": [str(outdir / f"g{i}.png") for i in range(n)], **kw}


def test_near_limb_is_drawn_over_the_far_one():
    """Two capsules crossing: the nearer one's colour wins where they overlap, whatever the order given."""
    cam = pm.camera("front", 0)
    near = {"kind": "capsule", "points": np.array([[-1.0, -0.5, 0.0], [1.0, -0.5, 0.0]]), "radius": 0.1,
            "colour": (0, 0, 255), "order": 0}     # y = -0.5: towards the front camera
    far = {"kind": "capsule", "points": np.array([[0.0, 0.5, -1.0], [0.0, 0.5, 1.0]]), "radius": 0.1,
           "colour": (255, 0, 0), "order": 0}
    for prims in ([near, far], [far, near]):
        im = np.asarray(po.draw_guide(prims, cam, 40.0, (64, 64), (128, 128), (0, 0, 0)))
        r, g, b = im[64, 64]
        assert b > 200 and r < 30
    # Depth shading: the far capsule is darker than its own colour, the near one full.
    im = np.asarray(po.draw_guide([near, far], cam, 40.0, (64, 64), (128, 128), (0, 0, 0)))
    assert im[64, 20, 2] > 240 and 140 < im[20, 64, 0] < 165


def test_side_view_draws_the_near_arm_over_the_body(tmp_path):
    """Seen from its left, the left arm (blue) covers the torso; from its right, the right arm (orange)."""
    clip = save_clip(pf.synthetic_clip(), tmp_path / "c.json")
    for view, side in (("left", "l"), ("right", "r")):
        res = po.op_pose_guides(guide_spec(clip, tmp_path, 1, view=view, elevation=0, clip_frames=[0],
                                           outputs=[str(tmp_path / f"{view}.png")]))
        im = np.asarray(Image.open(res["paths"][0])).astype(int)
        # the upper arm's middle, projected with the op's own scale and origin
        rt = pm.retarget(json.loads(Path(clip).read_text()))
        mid = (rt["positions"][0, pm.INDEX[f"shoulder_{side}"]] + rt["positions"][0, pm.INDEX[f"elbow_{side}"]]) / 2
        cam = pm.camera(view, 0)
        x0, y0, x1, y1 = res["box"]
        fx0, fx1, fy0, fy1 = po._extent([po.guide_primitives(rt["positions"][0], rt["world"][0], style="capsule",
                                                             stature=rt["stature"], thickness=1.0, weapons=[],
                                                             weapon_hand="right", rest_grip={})], cam)
        origin = ((x0 + x1) / 2 - (fx0 + fx1) / 2 * res["scale"], y1 + fy0 * res["scale"])
        px, py, _ = pm.project(mid, cam, res["scale"], origin)
        r, g, b = im[int(round(py)), int(round(px))]
        if side == "l":
            assert b > r + 40, (view, r, g, b)
        else:
            assert r > b + 40, (view, r, g, b)


def test_guides_are_deterministic_and_sized_to_the_match_image(tmp_path):
    clip = save_clip(pf.synthetic_clip(), tmp_path / "c.json")
    cut = np.zeros((300, 200, 4), dtype=np.uint8)
    cut[60:280, 70:130] = [200, 100, 50, 255]           # the subject: 220 px tall, feet at y = 280
    Image.fromarray(cut, "RGBA").save(tmp_path / "cut.png")
    spec = guide_spec(clip, tmp_path, 3, match_image=str(tmp_path / "cut.png"), view="front_right", elevation=30,
                      body="stocky", weapon=["sword", "shield"], contact=str(tmp_path / "sheet.png"))
    spec["size"] = None
    a = po.op_pose_guides(spec)
    first = [Path(p).read_bytes() for p in a["paths"]]
    b = po.op_pose_guides(spec)
    assert [Path(p).read_bytes() for p in b["paths"]] == first
    im = np.asarray(Image.open(a["paths"][0]))
    assert im.shape == (300, 200, 3)
    assert a["box"] == [70, 60, 130, 280]
    figure = np.nonzero(np.any(im != po.GUIDE_BACKGROUND["capsule"], axis=-1))
    if a["shrink"] == 1.0:  # the first frame stands on the subject's feet line
        assert abs(int(figure[0].max()) - 280) <= 2
    assert a["impact_index"] is not None and a["clip_frames"][a["impact_index"]] == a["clip_frames"][a["impact_index"]]
    assert Path(tmp_path / "sheet.png").is_file()
    assert len(a["durations_ms"]) == 3 and all(d >= 20 for d in a["durations_ms"])


def test_weapon_props_add_pixels_and_bad_options_fail(tmp_path):
    clip = save_clip(pf.synthetic_clip(), tmp_path / "c.json")
    plain = po.op_pose_guides(guide_spec(clip, tmp_path, 1, clip_frames=[10], view="left",
                                         outputs=[str(tmp_path / "a.png")]))
    armed = po.op_pose_guides(guide_spec(clip, tmp_path, 1, clip_frames=[10], view="left", weapon="spear",
                                         outputs=[str(tmp_path / "b.png")]))
    bg = np.array(po.GUIDE_BACKGROUND["capsule"])
    count = lambda p: int(np.any(np.asarray(Image.open(p)) != bg, -1).sum())  # noqa: E731
    assert count(armed["paths"][0]) > count(plain["paths"][0]) * 1.02 or armed["shrink"] < plain["shrink"]
    for bad in ({"weapon": "bazooka"}, {"weapon": ["sword", "axe"]}, {"style": "neon"}, {"view": "up"},
                {"body": "giant"}, {"exaggerate": 7}, {"selection": "random"}):
        with pytest.raises(ValueError):
            po.op_pose_guides(guide_spec(clip, tmp_path, 2, **bad))


def test_skeleton_style_uses_openpose_colours(tmp_path):
    clip = save_clip(pf.synthetic_clip(), tmp_path / "c.json")
    res = po.op_pose_guides(guide_spec(clip, tmp_path, 1, clip_frames=[0], style="skeleton",
                                       outputs=[str(tmp_path / "s.png")]))
    im = np.asarray(Image.open(res["paths"][0]))
    assert tuple(im[0, 0]) == (0, 0, 0)
    found = {tuple(int(v) for v in c) for c in im.reshape(-1, 3)}
    assert sum(c in found for c in po.OPENPOSE_COLOURS) >= 5


def test_helper_rejects_files_that_are_not_clips_without_echoing(tmp_path):
    secret = tmp_path / "x.json"
    secret.write_text('{"format": "something else", "token": "hunter2"}')
    with pytest.raises(ValueError) as err:
        po.op_pose_guides(guide_spec(str(secret), tmp_path, 2))
    assert "hunter2" not in str(err.value)


# -- library -------------------------------------------------------------------------


def test_resolve_and_list_clips(settings, tmp_path):
    clip = pf.synthetic_clip(name="fixture_attack")
    lib = library_with(settings, clip)
    assert poses.resolve_clip(settings, "fixture_attack") == (lib / "fixture_attack.json", "fixture_attack")
    assert poses.resolve_clip(settings, "attack")[1] == "fixture_attack"           # by action tag
    assert poses.resolve_clip(settings, str(lib / "fixture_attack.json"))[1] == "fixture_attack"
    outside = save_clip(clip, tmp_path / "elsewhere" / "c.json")
    with pytest.raises(poses.PoseLibraryError, match="must be under"):
        poses.resolve_clip(settings, outside)
    with pytest.raises(poses.PoseLibraryError, match="list_pose_clips"):
        poses.resolve_clip(settings, "moonwalk")
    listed = poses.list_pose_clips(settings)
    assert listed["count"] == 1 and listed["clips"][0]["impact_frame"] == clip["metadata"]["impact_frame"]
    assert poses.list_pose_clips(settings, action="walk")["count"] == 0


def test_bake_inputs_are_checked(tmp_path, settings):
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(poses.PoseLibraryError):
        poses.collect_inputs([str(tmp_path / "a.txt")])
    with pytest.raises(poses.PoseLibraryError):
        poses.collect_inputs([str(tmp_path / "missing.fbx")])
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "Walk A.FBX").write_text("x")
    (tmp_path / "d" / "walk-a.bvh").write_text("x")
    files = poses.collect_inputs([str(tmp_path / "d")])
    spec = poses.bake_spec(files, settings.pose_library, name=None, action=None, loop=None, weapon_hand=None,
                           take=None, root_motion="in_place")
    assert sorted(c["name"] for c in spec) == ["walk_a", "walk_a_2"]
    with pytest.raises(poses.PoseLibraryError):
        poses.bake_spec(files, settings.pose_library, name="x", action=None, loop=None, weapon_hand=None,
                        take=None, root_motion="in_place")
    with pytest.raises(poses.PoseLibraryError):
        poses.bake_spec(files[:1], settings.pose_library, name=None, action=None, loop=None, weapon_hand="both",
                        take=None, root_motion="in_place")


def test_default_library_is_outside_the_repository(monkeypatch):
    monkeypatch.delenv("LOCALGEN_POSE_DIR", raising=False)
    monkeypatch.setenv("LOCALGEN_ROOT", "/ai")
    lib = config.load().pose_library
    assert lib == Path("/ai/data/localgen/poses")
    assert not lib.resolve().is_relative_to(HERE.parent.resolve())


def test_no_project_words_in_pose_tools_or_presets():
    from localgen_mcp import server
    from test_sprites import PROJECT_WORDS

    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    for name in ("generate_animation", "bake_pose_library", "list_pose_clips"):
        text = (name + json.dumps(tools[name].inputSchema)).lower()
        for word in PROJECT_WORDS:
            assert word not in text, (name, word)
    presets = json.dumps([pm.BODIES, pm.VIEWS, list(po.WEAPONS), pm.ACTION_WORDS]).lower()
    for word in PROJECT_WORDS:
        assert word not in presets


# -- Blender baker -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def walker(tmp_path_factory):
    out = tmp_path_factory.mktemp("walker")
    subprocess.run([BLENDER, "-b", "--factory-startup", "-P", str(HERE / "blender_make_walker.py"), "--", str(out)],
                   check=True, capture_output=True, timeout=300)
    return out


@needs_blender
def test_baker_maps_grounds_and_records_rotations(walker, settings):
    res = asyncio.run(poses.bake_pose_library(settings, [str(walker)]))
    assert not res["failed"] and len(res["baked"]) == 2  # the FBX and the BVH
    index = json.loads((settings.pose_library / "index.json").read_text())
    assert set(index["clips"]) == {b["name"] for b in res["baked"]}
    for b in res["baked"]:
        assert b["joint_bones"]["elbow_l"] == "LeftLowerArm" and b["joint_bones"]["chest"] == "Chest"
        assert b["naming"].startswith("unity humanoid")
        clip = json.loads(Path(b["path"]).read_text())
        pos = np.asarray(clip["frames"]["positions"])
        local = np.asarray(clip["frames"]["local_rotations"])
        assert clip["frame_count"] == 21 and clip["fps"] == pytest.approx(24.0, abs=1e-3)
        # facing: the file faces +X, the clip faces -Y with its left at +X
        assert pos[0, pm.INDEX["hip_l"], 0] == pytest.approx(0.1, abs=1e-3)
        assert pos[0, pm.INDEX["toe_l"], 1] < -0.1
        # in place: the root bone walked 1 m, the pelvis stays over the origin
        assert clip["metadata"]["root_travel"] == pytest.approx(1.0, abs=1e-3)
        assert np.abs(pos[:, 0, :2]).max() < 1e-6
        # rotations: the elbow bend keyed on LeftLowerArm (frame 11 is index 10)
        assert pm.qangle(local[10, pm.INDEX["wrist_l"]]) == pytest.approx(90.0, abs=0.5)
        assert pm.qangle(local[0, pm.INDEX["wrist_l"]]) == pytest.approx(0.0, abs=0.5)
        assert pm.qangle(local[4, pm.INDEX["elbow_r"]]) == pytest.approx(100.0, abs=0.5)
        a = pm.arrays(clip)
        fk, _ = pm.forward(a["offsets"], a["local"], a["root_position"])
        assert np.abs(fk - pos).max() < 1e-3
        meta = clip["metadata"]
        assert meta["weapon_hand"] == "right" and meta["action"] == "attack"
        assert meta["impact_frame"] == 10  # the fastest swing ends at frame 11
        assert b["source_file"].startswith("fixture_attack_R.")


@needs_blender
def test_baker_reports_a_bad_file_and_keeps_going(walker, settings, tmp_path):
    broken = tmp_path / "broken.fbx"
    broken.write_bytes(b"not an fbx")
    res = asyncio.run(poses.bake_pose_library(settings, [str(broken), str(walker / "fixture_attack_R.fbx")]))
    assert [f["name"] for f in res["failed"]] == ["broken"] and len(res["baked"]) == 1


# -- generate_animation wiring --------------------------------------------------------------


def test_pose_clip_frames_use_the_guide_as_image_2(tmp_path, settings, monkeypatch):
    from test_sprites import FakeGPU, colours

    library_with(settings, pf.synthetic_clip(name="fixture_attack"))
    gpu = FakeGPU(tmp_path, bad={(1, 0)})
    monkeypatch.setattr(sprites, "draw", gpu.draw)
    monkeypatch.setattr(sprites, "cut", gpu.cut)
    res = asyncio.run(sprites.generate_animation(
        settings, "a test hero with a sword", "attack", frames=4, size=32, seed=5, min_motion=0.0,
        pose_clip="fixture_attack", body="stocky", view="left", elevation=20, weapon=["sword", "shield"],
        output_dir=str(tmp_path / "anim")))
    calls = [c for c in gpu.calls if "This is frame" in c["prompt"]]
    guides = res["pose"]["guides"]
    assert len(guides) == 4 and all(Path(g).is_file() for g in guides)
    ref = Image.open(res["base_reference"])
    assert all(Image.open(g).size == ref.size for g in guides)   # guides at the generation size
    for c in calls[:4]:
        i = int(c["prompt"].split("This is frame ")[1].split(" ")[0]) - 1
        assert c["refs"] == [guides[i]] and c["input_image"] == res["base_reference"]
        assert "Image 2 is a pose reference" in c["prompt"] and "previous frame" not in c["prompt"]
    assert calls[4]["refs"] == [guides[1]]                          # a retry keeps the guide
    per = res["per_frame"]
    assert per[1]["attempts"] == 2 and not any(f["previous_frame_reference"] for f in per)
    assert [f["clip_frame"] for f in per] == res["pose"]["clip_frames"]
    meta = pf.synthetic_clip()["metadata"]
    assert res["impact_frame"] is not None and res["pose"]["clip_frames"][res["impact_frame"]] == meta["impact_frame"]
    data = json.loads(Path(res["json"]).read_text())
    assert data["impact_frame"] == res["impact_frame"] and data["pose"]["body"]["legs"] == pm.BODIES["stocky"]["legs"]
    assert len(data["durations_ms"]) == 4 and data["loop"] is False
    gif = Image.open(res["preview_gif"])  # the clip's timing (Pillow merges identical frames, adding their times)
    total = 0
    for i in range(gif.n_frames):
        gif.seek(i)
        total += gif.info["duration"]
    assert abs(total - sum(data["durations_ms"])) <= 10 * 4  # GIF keeps centiseconds
    shared = {tuple(int(res["palette"][i][j:j + 2], 16) for j in (1, 3, 5)) for i in range(len(res["palette"]))}
    for f in res["frames"]:
        assert colours(np.asarray(Image.open(f))) <= shared


def test_pose_options_fail_before_any_drawing(tmp_path, settings, monkeypatch):
    from test_sprites import FakeGPU

    library_with(settings, pf.synthetic_clip(name="fixture_attack"))
    gpu = FakeGPU(tmp_path, bad=set())
    monkeypatch.setattr(sprites, "draw", gpu.draw)
    monkeypatch.setattr(sprites, "cut", gpu.cut)
    with pytest.raises(sprites.PixelError, match="view"):
        asyncio.run(sprites.generate_animation(settings, "a hero", "attack", frames=3, pose_clip="fixture_attack",
                                               view="sideways", output_dir=str(tmp_path / "a")))
    with pytest.raises(ValueError, match="not both"):
        asyncio.run(sprites.generate_animation(settings, "a hero", "attack", frames=2, pose_clip="fixture_attack",
                                               frame_prompts=["a", "b"], output_dir=str(tmp_path / "a")))
    assert gpu.calls == []


# -- GPU smoke test ----------------------------------------------------------------------

GPU = shutil.which("nvidia-smi") is not None and os.environ.get("LOCALGEN_GPU_SMOKE") == "1"


@pytest.mark.skipif(not GPU, reason="needs the GPU, the real backend and models (set LOCALGEN_GPU_SMOKE=1)")
def test_gpu_smoke_pose_driven_animation(tmp_path, monkeypatch):
    """A 4 frame pose driven strike from the synthetic clip through the real shared backend."""
    real = config.load()
    out = real.root / "tmp" / f"pose-smoke-{os.getpid()}"
    clip = save_clip(pf.synthetic_clip(name="fixture_attack"), out / "fixture_attack.json")
    try:
        anim = asyncio.run(sprites.generate_animation(
            real, "a knight in plain grey armour holding a sword", "attack", frames=4, size=64, retries=0,
            seed=21, pose_clip=clip, view="left", elevation=10, weapon="sword", output_dir=str(out)))
        assert np.asarray(Image.open(anim["strip"])).shape == (64, 256, 4)
        assert anim["impact_frame"] is not None and anim["consistency"]["frames"] == 4
    finally:
        shutil.rmtree(out, ignore_errors=True)
