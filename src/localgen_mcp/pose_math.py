"""Pose clips: skeleton mapping, clip building, retargeting, projection and key frame selection.

Pure numpy and the standard library, no imports from the package, so three places can load it:
the Blender baker (blender/bake_pose_clips.py, by file path, inside Blender's Python), the pixel
helper (pixel_ops.py draws pose guides with it, by file path) and the tests (as a module). The
MCP server process never imports it.

Canonical skeleton: JOINTS, about 20 humanoid points, each with a parent. Coordinates are metres,
Z up, the character facing -Y (its left side at +X), the floor at z = 0, and the pelvis over the
origin when root motion is removed. A camera at azimuth 0 sits at -Y and sees the character's
front.

A clip stores, per frame, every joint's position and every joint's local rotation: the rotation of
the segment parent -> joint relative to its parent segment, as a change from the rest pose
(identity at rest). Forward kinematics from the rest offsets and these rotations gives back the
positions, and retargeting to other proportions scales the rest offsets and keeps the rotations,
which is how one clip drives any body shape.
"""

from __future__ import annotations

import math
import re

import numpy as np

FORMAT = "localgen-pose-clip"
VERSION = 1

JOINTS: list[tuple[str, str | None]] = [
    ("pelvis", None), ("spine", "pelvis"), ("chest", "spine"), ("neck", "chest"), ("head", "neck"),
    ("head_top", "head"), ("face", "head"),
    ("shoulder_l", "chest"), ("elbow_l", "shoulder_l"), ("wrist_l", "elbow_l"), ("hand_l", "wrist_l"),
    ("shoulder_r", "chest"), ("elbow_r", "shoulder_r"), ("wrist_r", "elbow_r"), ("hand_r", "wrist_r"),
    ("hip_l", "pelvis"), ("knee_l", "hip_l"), ("ankle_l", "knee_l"), ("toe_l", "ankle_l"),
    ("hip_r", "pelvis"), ("knee_r", "hip_r"), ("ankle_r", "knee_r"), ("toe_r", "ankle_r"),
]
NAMES = [j for j, _ in JOINTS]
INDEX = {j: i for i, j in enumerate(NAMES)}
PARENTS = [INDEX[p] if p else -1 for _, p in JOINTS]
SYNTHETIC = ("head_top", "face")  # placed from the head bone; no source bone carries them
HEAD_TOP = 0.11                     # head joint to crown, as a share of stature
FACE = (0.075, 0.035)               # face point: forward and up from the head joint, share of stature
FOOT_JOINTS = ("ankle_l", "ankle_r", "toe_l", "toe_r")
CENTIMETRES = 20.0                  # a rest stature above this many units is read as centimetres

# Segments grouped for body proportions: each group's rest offsets scale by the body's factor.
GROUPS = {
    "legs": ("knee_l", "ankle_l", "knee_r", "ankle_r"),
    "feet": ("toe_l", "toe_r"),
    "arms": ("elbow_l", "wrist_l", "hand_l", "elbow_r", "wrist_r", "hand_r"),
    "torso": ("spine", "chest", "neck"),
    "neck": ("head",),
    "head": ("head_top", "face"),
    "shoulders": ("shoulder_l", "shoulder_r"),
    "hips": ("hip_l", "hip_r"),
}
BODY_KEYS = tuple(GROUPS) + ("thickness",)
# Presets are named by shape. stocky: short legs, broad, big head and torso. small: short, big
# head, long arms. tall: long limbs, a smaller head.
BODIES: dict[str, dict[str, float]] = {
    "human": {},
    "stocky": {"legs": 0.68, "feet": 1.1, "arms": 0.85, "torso": 1.05, "neck": 0.55, "head": 1.35,
               "shoulders": 1.35, "hips": 1.3, "thickness": 1.6},
    "small": {"legs": 0.72, "feet": 1.0, "arms": 1.12, "torso": 0.8, "neck": 0.8, "head": 1.55,
              "shoulders": 0.9, "hips": 0.9, "thickness": 0.85},
    "tall": {"legs": 1.15, "feet": 1.05, "arms": 1.1, "torso": 1.08, "neck": 1.1, "head": 0.9,
             "shoulders": 0.95, "hips": 0.95, "thickness": 0.85},
}
BODY_RANGE = (0.25, 3.0)

VIEWS = {"front": 0.0, "front_left": 45.0, "left": 90.0, "back_left": 135.0, "back": 180.0,
         "back_right": 225.0, "right": 270.0, "front_right": 315.0}
ELEVATION_RANGE = (-30.0, 90.0)
EXAGGERATE_RANGE = (0.0, 3.0)

ACTION_WORDS = (  # first match wins, so "BlockShield01 - Hit" is a block and not a hurt
    ("block", "block"), ("death", "die"), ("dying", "die"), ("die", "die"), ("damage", "hurt"),
    ("hurt", "hurt"), ("hit", "hurt"), ("chop", "chop"), ("mining", "chop"), ("dig", "chop"),
    ("hammer", "hammer"), ("attack", "attack"), ("slash", "attack"), ("stab", "attack"),
    ("punch", "attack"), ("kick", "attack"), ("swing", "attack"), ("strike", "attack"),
    ("sprint", "run"), ("run", "run"), ("jog", "run"), ("walk", "walk"), ("idle", "idle"),
    ("jump", "jump"),
)
STRIKES = ("attack", "chop", "hammer")
STOP_SHARE = 0.25   # a strike's hand "stops" when it moves under this share of its peak speed
STOP_WINDOW = 6     # frames after the peak in which to look for the stop
CYCLE_MATCH = 0.03  # a loop repeats when the pose comes back within this share of stature
CYCLE_MOTION = 0.05  # a loop moving less than this share of stature is never split into cycles
LOOPING = ("walk", "run", "idle")


class PoseError(ValueError):
    """A problem with a clip or a request; the message never quotes file contents."""


# -- quaternions (w, x, y, z), vectorised over leading axes ------------------------------------


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def qconj(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) * np.array([1.0, -1.0, -1.0, -1.0])


def qnorm(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(n > 1e-12, q / np.maximum(n, 1e-12), np.array([1.0, 0.0, 0.0, 0.0]))


def qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """v rotated by q (unit quaternions)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u, w = q[..., 1:], q[..., :1]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def qaxis(axis, degrees: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    h = math.radians(degrees) / 2
    return np.concatenate([[math.cos(h)], a * math.sin(h)])


def qbetween(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The smallest rotation taking direction u to direction v (any lengths; zero vectors give identity)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nu = np.linalg.norm(u, axis=-1, keepdims=True)
    nv = np.linalg.norm(v, axis=-1, keepdims=True)
    ok = (nu > 1e-9) & (nv > 1e-9)
    a = u / np.maximum(nu, 1e-12)
    b = v / np.maximum(nv, 1e-12)
    d = np.sum(a * b, -1, keepdims=True)
    q = np.concatenate([1.0 + d, np.cross(a, b)], -1)
    # Opposite directions: half a turn about any axis perpendicular to a.
    other = np.where(np.abs(a[..., :1]) < 0.9, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    perp = np.cross(a, other)
    perp = perp / np.maximum(np.linalg.norm(perp, axis=-1, keepdims=True), 1e-12)
    flip = np.concatenate([np.zeros_like(d), perp], -1)
    q = np.where(d < -0.999999, flip, q)
    return np.where(ok, qnorm(q), np.array([1.0, 0.0, 0.0, 0.0]))


def qangle(q: np.ndarray) -> np.ndarray:
    """Rotation angle in degrees (0..180)."""
    w = np.clip(np.abs(np.asarray(q, dtype=np.float64)[..., 0]), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(w))


def qpow(q: np.ndarray, k: float) -> np.ndarray:
    """The rotation q scaled to k times its angle about the same axis (shortest path)."""
    q = qnorm(q)
    q = np.where(q[..., :1] < 0, -q, q)
    half = np.arccos(np.clip(q[..., 0], -1.0, 1.0))
    s = np.sin(half)
    axis = np.where((s > 1e-9)[..., None], q[..., 1:] / np.maximum(s, 1e-12)[..., None], 0.0)
    return np.concatenate([np.cos(k * half)[..., None], axis * np.sin(k * half)[..., None]], -1)


def qmean(qs: np.ndarray, axis: int = 0) -> np.ndarray:
    """A mean rotation: signs aligned to the first, summed and normalised (good for nearby rotations)."""
    qs = np.asarray(qs, dtype=np.float64)
    first = np.take(qs, [0], axis=axis)
    sign = np.where(np.sum(qs * first, -1, keepdims=True) < 0, -1.0, 1.0)
    return qnorm(np.sum(qs * sign, axis=axis))


# -- skeleton mapping (bone names to canonical joints) --------------------------------------------

_SKIP = re.compile(r"twist|roll|leaf|(^|[^a-z])ik([^a-z]|$)|nub|proxy|prop|wep|weapon|scabbard|eye|jaw|"
                   r"index|ring|pinky|thumb|metacarpal|helper|socket|target|pole|(^|[^a-z])end$|_end|"
                   r"interaction|center_of_mass|camera|cloth|hair|breast|tail", re.I)
_PREFIX = re.compile(r"^(b-|bip0*1[ _]|bip[ _]|def-|org-|mch-|cc_base_|rig_|bone_)", re.I)
PART_ALIASES = {
    "pelvis": {"hips", "pelvis", "hip"},
    "neck": {"neck", "neck01", "neck1"},
    "head": {"head"},
    "clavicle": {"shoulder", "clavicle", "collar", "collarbone"},
    "upperarm": {"upperarm", "arm", "uparm"},
    "forearm": {"forearm", "lowerarm", "elbow"},
    "hand": {"hand", "wrist"},
    "middle1": {"middle1", "middle01", "middlefinger1", "middlefinger01", "handmiddle1", "middleproximal",
                "fingermiddle1", "middle"},
    "thigh": {"upleg", "thigh", "upperleg", "hip"},
    "shin": {"leg", "shin", "calf", "lowerleg", "knee"},
    "foot": {"foot", "ankle"},
    "toe": {"toebase", "toe", "toes", "ball", "toe0", "toe01"},
}
SIDED = ("clavicle", "upperarm", "forearm", "hand", "middle1", "thigh", "shin", "foot", "toe")


def split_name(name: str) -> tuple[str | None, str]:
    """(side "l", "r" or None, base) for a bone name: namespaces ("mixamorig:"), rig prefixes
    ("B-", "Bip01 ") and separators are dropped, the side is read from a Left/Right word or an
    l/r token at either end, and the base is lower case letters and digits."""
    n = re.split(r"[:|]", name)[-1].strip()
    n = _PREFIX.sub("", n)
    side, rest = None, n
    m = re.match(r"^(left|right)[ _.\-]?(.+)$", n, re.I)
    if m:
        side, rest = m.group(1)[0].lower(), m.group(2)
    else:
        m = re.match(r"^(.+?)[ _.\-](l|r|left|right)$", n, re.I)
        if m:
            side, rest = m.group(2)[0].lower(), m.group(1)
        else:
            m = re.match(r"^(l|r)[ _.\-](.+)$", n, re.I) or re.match(r"^([LR])(?=[A-Z][a-z])(.+)$", n)
            if m:
                side, rest = m.group(1).lower(), m.group(2)
            else:
                m = re.match(r"^(.+?)(Left|Right)$", n)
                if m:
                    side, rest = m.group(2)[0].lower(), m.group(1)
    return side, re.sub(r"[^a-z0-9]", "", rest.lower())


def naming_scheme(names: list[str]) -> str:
    """A label for the bone naming, for the report."""
    joined = " ".join(names)
    if any(n.lower().startswith("mixamorig") for n in names):
        return "mixamo (mixamorig: prefix)"
    if any(n.startswith("B-") for n in names):
        return "B- prefix with .L/.R suffixes"
    if re.search(r"\b(upperarm|lowerarm|thigh|calf)_[lr]\b", joined):
        return "unreal mannequin (upperarm_l, calf_l)"
    if re.search(r"\bLeft(UpperArm|LowerArm|UpperLeg|LowerLeg)\b", joined):
        return "unity humanoid (LeftUpperArm)"
    if re.search(r"\bLeft(Arm|ForeArm|UpLeg)\b", joined):
        return "mixamo style without prefix (LeftArm, LeftUpLeg)"
    if any(n.lower().startswith("bip") for n in names):
        return "3ds max biped (Bip01 L Thigh)"
    return "other"


def map_skeleton(bones: list[tuple[str, str | None]]) -> dict:
    """Map a source skeleton, given as (name, parent name) pairs, to the canonical joints.

    Returns {"joints": {canonical: spec}, "bones": {canonical: source bone}, "naming": label}.
    A spec says where the joint sits and which source bone turns the segment that ends there:
    {"point": ["head", bone] | ["tail", bone] | ["lerp", bone_a, bone_b, t] | ["offset", bone, [fwd, up]],
     "carrier": bone the point moves with, "edge": bone whose rotation drives parent -> joint}.
    "offset" points (the crown and the face) are placed from the head bone by build_clip.
    Raises PoseError naming the parts that were not found."""
    parent = {n: p for n, p in bones}
    order = [n for n, _ in bones]

    def depth(n: str) -> int:
        d = 0
        while parent.get(n):
            n = parent[n]
            d += 1
        return d

    def ancestors(n: str) -> list[str]:
        out = []
        while parent.get(n):
            n = parent[n]
            out.append(n)
        return out

    found: dict[tuple[str, str | None], list[str]] = {}
    for n in order:
        side, base = split_name(n)
        mid = base in PART_ALIASES["middle1"] or re.fullmatch(r"(hand)?middle(finger)?0?1|middleproximal", base)
        if _SKIP.search(n) and not mid:
            continue
        for part, aliases in PART_ALIASES.items():
            if base in aliases or (part == "middle1" and mid):
                if part in SIDED and side is None:
                    continue
                if part not in SIDED and side is not None:
                    continue
                found.setdefault((part, side), []).append(n)
    pick = {k: min(v, key=lambda n: (depth(n), order.index(n))) for k, v in found.items()}

    def get(part, side=None):
        return pick.get((part, side))

    missing = []
    thighs = [get("thigh", s) for s in "lr"]
    pelvis = get("pelvis")
    if pelvis is None and all(thighs):
        common = [a for a in ancestors(thighs[0]) if a in ancestors(thighs[1])]
        pelvis = common[0] if common else None
    neck, head = get("neck"), get("head")
    if neck is None and head is not None:
        neck = parent.get(head)
    for label, value in (("pelvis or hips", pelvis), ("neck", neck), ("head", head)):
        if value is None:
            missing.append(label)
    joints: dict[str, dict] = {}
    bone_of: dict[str, str] = {}

    def at_head(j: str, b: str) -> None:
        joints[j] = {"point": ["head", b], "carrier": b, "edge": parent.get(b) or b}
        bone_of[j] = b

    if not missing:
        if pelvis not in ancestors(neck):
            missing.append("a spine from the pelvis to the neck")
        else:
            joints["pelvis"] = {"point": ["head", pelvis], "carrier": pelvis, "edge": pelvis}
            bone_of["pelvis"] = pelvis
            chain = list(reversed(ancestors(neck)))
            chain = chain[chain.index(pelvis) + 1:]  # bones strictly between the pelvis and the neck
            if len(chain) >= 2:
                at_head("spine", chain[0])
                at_head("chest", chain[-1])
            elif len(chain) == 1:
                at_head("spine", chain[0])
                joints["chest"] = {"point": ["lerp", chain[0], neck, 0.5], "carrier": chain[0], "edge": chain[0]}
                bone_of["chest"] = chain[0]
            else:
                for j, t in (("spine", 1 / 3), ("chest", 2 / 3)):
                    joints[j] = {"point": ["lerp", pelvis, neck, t], "carrier": pelvis, "edge": pelvis}
                    bone_of[j] = pelvis
            at_head("neck", neck)
            if head not in ancestors(head) and neck in ancestors(head):
                at_head("head", head)
            else:
                missing.append("head below the neck")
            joints["head_top"] = {"point": ["offset", head, [0.0, HEAD_TOP]], "carrier": head, "edge": head}
            joints["face"] = {"point": ["offset", head, list(FACE)], "carrier": head, "edge": head}
    for s in "lr":
        arm = [get(p, s) for p in ("upperarm", "forearm", "hand")]
        leg = [get(p, s) for p in ("thigh", "shin", "foot")]
        names = ("upper arm", "forearm", "hand", "thigh", "shin", "foot")
        for label, b in zip(names, arm + leg):
            if b is None:
                missing.append(f"{label} ({'left' if s == 'l' else 'right'})")
        if None in arm or None in leg or "pelvis" not in joints:
            continue
        if not (arm[0] in ancestors(arm[1]) and arm[1] in ancestors(arm[2]) and
                leg[0] in ancestors(leg[1]) and leg[1] in ancestors(leg[2]) and pelvis in ancestors(leg[0])):
            missing.append(f"connected {'left' if s == 'l' else 'right'} arm and leg chains")
            continue
        at_head(f"shoulder_{s}", arm[0])
        at_head(f"elbow_{s}", arm[1])
        at_head(f"wrist_{s}", arm[2])
        mid = get("middle1", s)
        if mid and arm[2] in ancestors(mid):
            joints[f"hand_{s}"] = {"point": ["head", mid], "carrier": arm[2], "edge": arm[2]}
        else:
            joints[f"hand_{s}"] = {"point": ["tail", arm[2]], "carrier": arm[2], "edge": arm[2]}
        bone_of[f"hand_{s}"] = arm[2]
        at_head(f"hip_{s}", leg[0])
        at_head(f"knee_{s}", leg[1])
        at_head(f"ankle_{s}", leg[2])
        toe = get("toe", s)
        if toe and leg[2] in ancestors(toe):
            at_head(f"toe_{s}", toe)
        else:
            joints[f"toe_{s}"] = {"point": ["tail", leg[2]], "carrier": leg[2], "edge": leg[2]}
            bone_of[f"toe_{s}"] = leg[2]
    if missing:
        raise PoseError("could not map the skeleton to a humanoid: no " + ", ".join(missing))
    return {"joints": {j: joints[j] for j in NAMES}, "bones": bone_of, "naming": naming_scheme(order)}


# -- building a clip from sampled bones -------------------------------------------------------------


def _r(a: np.ndarray, nd: int = 5) -> list:
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def stature_of(rest: dict[str, np.ndarray]) -> float:
    """Standing height estimate from the rest pose: head joint height over the feet, plus the crown."""
    foot = min(rest[j][2] for j in FOOT_JOINTS if j in rest)
    return float((rest["head"][2] - foot) / (1.0 - HEAD_TOP))


def build_clip(raw: dict, *, name: str, action: str | None = None, loop: bool | None = None,
               weapon_hand: str | None = None, root_motion: str = "in_place", source: dict | None = None) -> dict:
    """A clip from bones sampled in Blender (or built by hand in tests).

    raw: {"fps", "joints": {canonical: spec from map_skeleton}, "rest_points": {canonical: [x, y, z]},
    "points": [[[x, y, z] per real joint, in rest_points order] per frame], "bones": [names],
    "rest_rotations": [[w, x, y, z] per bone], "rotations": [[[w, x, y, z] per bone] per frame]}
    (world space, any facing). Real joints are every joint except the "offset" ones.
    root_motion: "in_place" pins the pelvis over the origin every frame (heights kept); "keep"
    keeps its horizontal travel relative to the first frame."""
    if root_motion not in ("in_place", "keep"):
        raise PoseError('root_motion must be "in_place" or "keep"')
    specs = raw["joints"]
    real = list(raw["rest_points"])
    bones = list(raw["bones"])
    bi = {b: i for i, b in enumerate(bones)}
    pts = np.asarray(raw["points"], dtype=np.float64)          # F, Jr, 3
    rots = qnorm(np.asarray(raw["rotations"], dtype=np.float64))  # F, B, 4
    rest_rot = qnorm(np.asarray(raw["rest_rotations"], dtype=np.float64))
    if pts.ndim != 3 or pts.shape[0] < 1 or pts.shape[1] != len(real):
        raise PoseError("sampled points do not match the joints")
    frames = pts.shape[0]
    delta = qmul(rots, qconj(rest_rot)[None])                   # world rotation from rest, per bone
    rest = {j: np.asarray(raw["rest_points"][j], dtype=np.float64) for j in real}

    # Upright: a file read with the wrong up axis (a Z up BVH read as Y up) has the rest figure
    # lying down; turn it so feet to head points up. Upright figures are left exactly as they are.
    feet = (rest["ankle_l"] + rest["ankle_r"]) / 2
    up = rest["head"] - feet
    if np.linalg.norm(up) > 1e-9 and up[2] < math.cos(math.radians(45)) * np.linalg.norm(up):
        stand = qbetween(up, np.array([0.0, 0.0, 1.0]))
        rest = {j: qrot(stand, p) for j, p in rest.items()}
        pts = qrot(stand, pts)
        delta = qmul(qmul(stand, delta), qconj(stand))
    # Facing: turn about Z so the character's left (hips and shoulders, right to left) is +X.
    left = (rest["hip_l"] - rest["hip_r"]) + (rest["shoulder_l"] - rest["shoulder_r"])
    yaw = -math.atan2(left[1], left[0])
    turn = qaxis([0, 0, 1], math.degrees(yaw))
    rest = {j: qrot(turn, p) for j, p in rest.items()}
    pts = qrot(turn, pts)
    delta = qmul(qmul(turn, delta), qconj(turn))
    stature = stature_of(rest)
    if stature <= 0:
        raise PoseError("the rest pose is not upright (the head is not above the feet)")
    unit = 0.01 if stature > CENTIMETRES else 1.0  # a figure over 20 units tall is in centimetres
    if unit != 1.0:
        rest = {j: p * unit for j, p in rest.items()}
        pts = pts * unit
        stature *= unit

    pos = np.zeros((frames, len(NAMES), 3))
    rest_all = np.zeros((len(NAMES), 3))
    for k, j in enumerate(real):
        pos[:, INDEX[j]] = pts[:, k]
        rest_all[INDEX[j]] = rest[j]
    for j in SYNTHETIC:
        fwd, up = specs[j]["point"][2]
        off = np.array([0.0, -fwd * stature, up * stature])
        h = INDEX["head"]
        rest_all[INDEX[j]] = rest_all[h] + off
        pos[:, INDEX[j]] = pos[:, h] + qrot(delta[:, bi[specs[j]["edge"]]], off)

    # Floor at the lowest the feet ever reach; root motion off.
    floor = float(pos[:, [INDEX[j] for j in FOOT_JOINTS], 2].min())
    pos[..., 2] -= floor
    rest_all[:, 2] -= float(min(rest_all[INDEX[j], 2] for j in FOOT_JOINTS))
    pel = pos[:, INDEX["pelvis"], :2].copy()
    travel = float(np.max(np.linalg.norm(pel - pel[0], axis=-1)))
    shift = pel if root_motion == "in_place" else np.broadcast_to(pel[0], pel.shape)
    pos[..., :2] -= shift[:, None, :]
    rest_all[:, :2] -= rest_all[INDEX["pelvis"], :2]

    # Segment rotations: the edge bone's rotation, then the smallest swing that lines the rest
    # segment up with the sampled one (exact positions, the bone's own twist).
    world = np.zeros((frames, len(NAMES), 4))
    world[:, 0] = delta[:, bi[specs["pelvis"]["edge"]]]
    for j, p in enumerate(PARENTS):
        if p < 0:
            continue
        d = delta[:, bi[specs[NAMES[j]]["edge"]]]
        e_rest = rest_all[j] - rest_all[p]
        e_now = pos[:, j] - pos[:, p]
        if np.linalg.norm(e_rest) < 1e-6:
            world[:, j] = world[:, p]
            continue
        world[:, j] = qmul(qbetween(qrot(d, e_rest), e_now), d)
    local = np.zeros_like(world)
    local[:, 0] = world[:, 0]
    for j, p in enumerate(PARENTS):
        if p >= 0:
            local[:, j] = qnorm(qmul(qconj(world[:, p]), world[:, j]))
    offsets = np.array([rest_all[j] - rest_all[p] if p >= 0 else np.zeros(3) for j, p in enumerate(PARENTS)])
    fps = float(raw.get("fps") or 30.0)
    clip = {
        "format": FORMAT, "version": VERSION, "name": name, "fps": fps, "frame_count": frames,
        "joints": NAMES, "parents": PARENTS,
        "rest": {"positions": _r(rest_all), "offsets": _r(offsets),
                 "bone_lengths": _r(np.linalg.norm(offsets, axis=-1)), "stature": round(stature, 5)},
        "frames": {"root_position": _r(pos[:, 0]), "root_rotation": _r(local[:, 0], 6),
                   "local_rotations": _r(local, 6), "positions": _r(pos)},
        "source": {**(source or {}), "joint_bones": raw.get("joint_bones") or {}},
    }
    clip["metadata"] = analyse(clip, name=name, action=action, loop=loop, weapon_hand=weapon_hand)
    clip["metadata"]["root_travel"] = round(travel, 4)
    clip["metadata"]["root_motion"] = root_motion
    return clip


# -- clip access, forward kinematics, retargeting -----------------------------------------------------


def check_clip(clip: dict) -> None:
    if not isinstance(clip, dict) or clip.get("format") != FORMAT:
        raise PoseError("not a pose clip file")
    if clip.get("joints") != NAMES:
        raise PoseError("the pose clip has a different joint set; bake it again")
    f = clip.get("frames") or {}
    n = int(clip.get("frame_count") or 0)
    shapes = (("root_position", (n, 3)), ("root_rotation", (n, 4)), ("local_rotations", (n, len(NAMES), 4)))
    for key, shape in shapes:
        if np.asarray(f.get(key, []), dtype=np.float64).shape != shape:
            raise PoseError(f"the pose clip's {key} has the wrong shape")
    if n < 1:
        raise PoseError("the pose clip has no frames")


def arrays(clip: dict) -> dict:
    check_clip(clip)
    f = clip["frames"]
    return {"offsets": np.asarray(clip["rest"]["offsets"], dtype=np.float64),
            "root_position": np.asarray(f["root_position"], dtype=np.float64),
            "local": qnorm(np.asarray(f["local_rotations"], dtype=np.float64)),
            "stature": float(clip["rest"]["stature"])}


def forward(offsets: np.ndarray, local: np.ndarray, root_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions (F, J, 3) and world segment rotations (F, J, 4) from rest offsets, local rotations
    (the root's entry is the root rotation) and root positions."""
    frames = local.shape[0]
    pos = np.zeros((frames, len(NAMES), 3))
    world = np.zeros((frames, len(NAMES), 4))
    pos[:, 0] = root_position
    world[:, 0] = local[:, 0]
    for j, p in enumerate(PARENTS):
        if p < 0:
            continue
        world[:, j] = qmul(world[:, p], local[:, j])
        pos[:, j] = pos[:, p] + qrot(world[:, j], offsets[j])
    return pos, world


def resolve_body(body) -> dict[str, float]:
    """Proportion factors from a preset name or a dict of factors (keys BODY_KEYS, optional "base" preset)."""
    if body is None:
        body = "human"
    if isinstance(body, str):
        if body not in BODIES:
            raise PoseError(f"body must be one of {sorted(BODIES)} or a dict of proportions")
        base, custom = body, {}
    elif isinstance(body, dict):
        base = body.get("base", "human")
        if base not in BODIES:
            raise PoseError(f"body base must be one of {sorted(BODIES)}")
        custom = {k: v for k, v in body.items() if k != "base"}
    else:
        raise PoseError("body must be a preset name or a dict of proportions")
    out = {k: 1.0 for k in BODY_KEYS}
    out.update(BODIES[base])
    for k, v in custom.items():
        if k not in BODY_KEYS:
            raise PoseError(f"unknown body proportion {k!r}; use {list(BODY_KEYS)}")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not BODY_RANGE[0] <= v <= BODY_RANGE[1]:
            raise PoseError(f"body proportion {k} must be a number from {BODY_RANGE[0]} to {BODY_RANGE[1]}")
        out[k] = float(v)
    return out


def exaggerate(local: np.ndarray, k: float) -> np.ndarray:
    """Scale every joint's rotation away from the clip's average pose by k (1 keeps the clip, 0 holds
    the average pose). Relative to the average, not the skeleton's bind pose: the bind pose is often
    a T pose, and pushing away from it would spread the arms instead of exaggerating the motion. The
    root rotation is kept, so a fall does not over-rotate."""
    if abs(k - 1.0) < 1e-9:
        return local
    mean = qmean(local, axis=0)[None]
    out = qmul(mean, qpow(qmul(qconj(mean), local), k))
    out[:, 0] = local[:, 0]
    return qnorm(out)


def retarget(clip: dict, body=None, exaggeration: float = 1.0) -> dict:
    """The clip on another body: rest offsets scaled per segment group, rotations kept (optionally
    exaggerated), the root height scaled with the legs, and the whole clip lifted or lowered so
    the lowest the feet reach is the floor again. Returns positions, world rotations, offsets and
    the body."""
    a = arrays(clip)
    factors = resolve_body(body)
    if not EXAGGERATE_RANGE[0] <= float(exaggeration) <= EXAGGERATE_RANGE[1]:
        raise PoseError(f"exaggerate must be from {EXAGGERATE_RANGE[0]} to {EXAGGERATE_RANGE[1]}")
    scale = np.ones(len(NAMES))
    for group, joints in GROUPS.items():
        for j in joints:
            scale[INDEX[j]] = factors[group]
    offsets = a["offsets"] * scale[:, None]
    root = a["root_position"].copy()
    root *= factors["legs"]  # pelvis height (and any kept travel) goes with the leg length
    local = exaggerate(a["local"], float(exaggeration))
    pos, world = forward(offsets, local, root)
    floor = float(pos[:, [INDEX[j] for j in FOOT_JOINTS], 2].min())
    pos[..., 2] -= floor
    rest_pos, _ = forward(offsets, np.tile(np.array([1.0, 0, 0, 0]), (1, len(NAMES), 1)),
                          np.array([[0.0, 0.0, 0.0]]))
    stature = float((rest_pos[0, INDEX["head_top"], 2] - rest_pos[0, [INDEX[j] for j in FOOT_JOINTS], 2].min()))
    return {"positions": pos, "world": world, "offsets": offsets, "body": factors, "stature": stature}


# -- camera ---------------------------------------------------------------------------------------


def camera(view, elevation: float) -> np.ndarray:
    """Rows right, up, back (towards the camera) of an orthographic camera. view: a VIEWS name or
    an azimuth in degrees (0 sees the front, 90 the character's left side, as _rig.py's cameras);
    elevation: degrees above the horizon (0 side on, 90 straight down)."""
    if isinstance(view, str):
        if view not in VIEWS:
            raise PoseError(f"view must be one of {list(VIEWS)} or an azimuth in degrees")
        az = VIEWS[view]
    elif isinstance(view, (int, float)) and not isinstance(view, bool) and math.isfinite(view):
        az = float(view)
    else:
        raise PoseError(f"view must be one of {list(VIEWS)} or an azimuth in degrees")
    if not isinstance(elevation, (int, float)) or not ELEVATION_RANGE[0] <= float(elevation) <= ELEVATION_RANGE[1]:
        raise PoseError(f"elevation must be from {ELEVATION_RANGE[0]} to {ELEVATION_RANGE[1]} degrees")
    a, e = math.radians(az), math.radians(float(elevation))
    back = np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])
    right = np.array([math.cos(a), math.sin(a), 0.0])
    up = np.cross(back, right)
    return np.stack([right, up, back])


def project(points: np.ndarray, cam: np.ndarray, scale: float, origin: tuple[float, float]) -> np.ndarray:
    """(..., 3) world points to (..., 3): pixel x, pixel y (down) and depth (larger is nearer)."""
    p = np.asarray(points, dtype=np.float64) @ cam.T
    return np.stack([origin[0] + p[..., 0] * scale, origin[1] - p[..., 1] * scale, p[..., 2]], -1)


# -- analysis: action, loop, weapon hand, key frames -----------------------------------------------


def action_from_name(name: str) -> str | None:
    low = name.lower()
    for word, tag in ACTION_WORDS:
        if word in low:
            return tag
    return None


def hand_from_name(name: str) -> str | None:
    m = re.search(r"(?:^|[_\-. @])([LR])(?=$|[_\-. ])", name, re.I)
    return {"L": "left", "R": "right"}[m.group(1).upper()] if m else None


def find_cycle(pos: np.ndarray, stature: float, usable: int) -> int | None:
    """The length in frames of a cycle that repeats inside a looping clip (a walk recorded as two
    strides), or None when the clip is one cycle. A lag counts when the pose, relative to the
    pelvis, comes back within CYCLE_MATCH of stature on average and that is a local minimum well
    below the clip's largest change; a clip that barely moves is never split."""
    rel = pos[:usable] - pos[:usable, :1]
    lags = list(range(max(2, usable // 6), usable - 3))
    if not lags:
        return None
    d = np.array([float(np.mean(np.linalg.norm(rel[:usable - L] - rel[L:usable], axis=-1))) for L in lags])
    if d.max() < CYCLE_MOTION * stature:
        return None
    for i, L in enumerate(lags):
        left = d[i - 1] if i > 0 else np.inf
        right = d[i + 1] if i + 1 < len(d) else np.inf
        if d[i] < CYCLE_MATCH * stature and d[i] < 0.2 * d.max() and d[i] <= left and d[i] <= right:
            return L
    return None


def _speed(p: np.ndarray, fps: float) -> np.ndarray:
    if len(p) < 2:
        return np.zeros(len(p))
    v = np.gradient(p, axis=0) * fps
    return np.linalg.norm(v, axis=-1)


def analyse(clip: dict, *, name: str, action: str | None = None, loop: bool | None = None,
            weapon_hand: str | None = None) -> dict:
    """Metadata for a clip: action tag, loop flag, weapon hand and suggested key frames.

    Strikes (attack, chop, hammer): impact is the frame the weapon hand all but stops after its
    fastest motion (the blow lands), or, when it never stops, its most extended frame (distance
    from the chest) near the fastest; wind-up is the frame before impact where the
    hand is pulled back furthest against the strike direction; recover is where it comes back
    closest to its start.
    Hurt: impact is the largest recoil of the head from the first frame. Die: impact is the first
    frame the pelvis is down near its lowest. Walk and run: the two foot contacts (feet farthest
    apart, either foot forward) and the passing frames between them."""
    pos = np.asarray(clip["frames"]["positions"], dtype=np.float64)
    frames = len(pos)
    fps = float(clip["fps"])
    stature = float(clip["rest"]["stature"])
    tag = (action or action_from_name(name) or "motion").lower().strip()
    ends = float(np.max(np.linalg.norm(pos[0] - pos[-1], axis=-1))) / stature if frames > 1 else 0.0
    named_loop = "loop" in name.lower() or "cycle" in name.lower()
    is_loop = bool(loop) if loop is not None else (named_loop or tag in LOOPING)
    repeats = bool(is_loop and frames > 2 and ends < 0.02)
    usable = frames - 1 if repeats else frames
    cycle = find_cycle(pos, stature, usable) if is_loop and frames >= 8 else None
    if cycle:
        usable = cycle  # key frames and frame selection use the first cycle
    J = INDEX
    hand_src = "given"
    if weapon_hand is None:
        weapon_hand = hand_from_name(name)
        hand_src = "name"
    if weapon_hand is None:
        paths = {s: float(np.sum(np.linalg.norm(np.diff(pos[:, J[f"hand_{s[0]}"]] - pos[:, J["chest"]], axis=0),
                                                axis=-1))) for s in ("left", "right")}
        weapon_hand = max(paths, key=paths.get) if frames > 1 else "right"
        hand_src = "motion"
    if weapon_hand not in ("left", "right"):
        raise PoseError('weapon_hand must be "left" or "right"')
    keys: list[tuple[int, str]] = [(0, "start")]
    impact = None
    last = usable - 1
    if frames >= 3 and tag in STRIKES:
        hand = pos[:, J[f"hand_{weapon_hand[0]}"]]
        speed = _speed(hand, fps)[:usable]
        peak = int(np.argmax(speed))
        # The hit: the first frame from the fastest on where the hand all but stops (its step to
        # the next frame under STOP_SHARE of the peak speed), within STOP_WINDOW frames. A swing
        # that never stops (a slash through the air) takes its most extended frame near the peak.
        step = np.linalg.norm(np.diff(hand, axis=0), axis=-1) * fps
        stops = [t for t in range(peak, min(last, peak + STOP_WINDOW) + 1)
                 if t < len(step) and step[t] < STOP_SHARE * speed[peak]]
        if stops:
            impact = stops[0]
        else:
            reach = np.linalg.norm(hand - pos[:, J["chest"]], axis=-1)
            lo, hi = max(0, peak - 1), min(last, peak + 3)
            impact = lo + int(np.argmax(reach[lo:hi + 1]))
        if impact > 0:
            # Wind-up: the hand pulled back furthest against the strike (hand from chest, along the
            # direction it points at impact).
            rel = hand - pos[:, J["chest"]]
            aim = rel[impact] / max(float(np.linalg.norm(rel[impact])), 1e-9)
            wind = int(np.argmin(rel[:impact] @ aim))
            if wind > 0:
                keys.append((wind, "wind_up"))
        keys.append((impact, "impact"))
        if impact < last:
            back = impact + 1 + int(np.argmin(np.linalg.norm(hand[impact + 1:usable] - hand[0], axis=-1)))
            mid = (impact + back) // 2
            if impact < mid < back:
                keys.append((mid, "follow_through"))
            keys.append((back, "recover"))
    elif frames >= 3 and tag == "hurt":
        head = pos[:usable, J["head"]]
        impact = int(np.argmax(np.linalg.norm(head - head[0], axis=-1)))
        keys.append((impact, "impact"))
        if impact < last:
            back = impact + 1 + int(np.argmin(np.linalg.norm(head[impact + 1:] - head[0], axis=-1)))
            if back > impact + 1:
                keys.append(((impact + back) // 2, "staggered"))
            keys.append((back, "recover"))
    elif frames >= 3 and tag == "die":
        z = pos[:usable, J["pelvis"], 2]
        low = z.min()
        down = np.nonzero(z <= low + 0.15 * max(z[0] - low, 1e-6))[0]
        impact = int(down[0]) if len(down) else last
        if impact > 1:
            keys.append((impact // 2, "falling"))
        keys.append((impact, "ground"))
        if impact < last:
            keys.append((last, "lying"))
    elif frames >= 3 and tag in ("walk", "run"):
        sep = pos[:usable, J["toe_l"], 1] - pos[:usable, J["toe_r"], 1]  # forward is -Y
        keys.append((int(np.argmin(sep)), "contact_left"))
        keys.append((int(np.argmax(sep)), "contact_right"))
        cross = np.nonzero(np.diff(np.sign(sep)) != 0)[0]
        for c in cross[:2]:
            keys.append((int(c), "passing"))
    elif frames >= 4:
        for q in (1, 2, 3):
            keys.append((q * usable // 4, f"quarter_{q}"))
    if not is_loop and last > 0 and all(f != last for f, _ in keys):
        keys.append((last, "end"))
    seen, ordered = set(), []
    for f, label in sorted(keys):
        if f not in seen:
            seen.add(f)
            ordered.append({"frame": f, "label": label})
    return {"action": tag, "loop": is_loop, "ends_match": round(ends, 4), "last_frame_repeats_first": repeats,
            "cycle_frames": usable,
            "impact_frame": impact, "key_frames": ordered, "weapon_hand": weapon_hand,
            "weapon_hand_source": hand_src, "stature": round(stature, 4), "duration": round(frames / fps, 4)}


def cycle_length(meta: dict, frame_count: int) -> int:
    """Frames that make one pass of the clip: one cycle of a loop (without a last frame that repeats
    the first), or every frame of a one shot."""
    if meta.get("cycle_frames"):
        return max(1, min(int(meta["cycle_frames"]), frame_count))
    return max(1, frame_count - 1 if meta.get("last_frame_repeats_first") else frame_count)


KEY_PRIORITY = ("impact", "ground", "wind_up", "start", "recover", "contact_left", "contact_right",
                "falling", "follow_through", "staggered", "lying", "end", "passing")


def select_frames(meta: dict, frame_count: int, n: int, mode: str = "auto") -> dict:
    """Which clip frames to draw for an n frame animation.

    "even": n frames evenly spaced over one cycle (a loop leaves out a last frame that repeats the
    first, and a clip holding several cycles uses the first; a one shot runs first to last). "keys": the clip's key frames, the impact always among
    them: with more frames than keys the largest gaps are split, with fewer the keys are kept in
    priority order (impact first). "auto": even for loops, keys otherwise.
    Returns {"frames": [clip frame per output frame], "labels": [...], "impact_index": output index or None}."""
    if n < 1:
        raise PoseError("need at least one frame")
    if mode not in ("auto", "even", "keys"):
        raise PoseError('frame selection must be "auto", "even" or "keys"')
    loop = bool(meta.get("loop"))
    usable = cycle_length(meta, frame_count)
    if mode == "auto":
        mode = "even" if loop or not meta.get("key_frames") else "keys"
    labels_at = {k["frame"]: k["label"] for k in meta.get("key_frames") or []}
    if mode == "even":
        if loop or n == 1:
            picked = [int(math.floor(i * usable / n)) for i in range(n)]
        else:
            picked = [int(round(i * (usable - 1) / (n - 1))) for i in range(n)]
    else:
        keys = sorted({k["frame"] for k in meta.get("key_frames") or [] if 0 <= k["frame"] < usable} | {0})
        impact = meta.get("impact_frame")
        if len(keys) >= n:
            rank = {label: i for i, label in enumerate(KEY_PRIORITY)}
            ordered = sorted(keys, key=lambda f: (rank.get(labels_at.get(f, ""), len(rank)), f))
            if impact is not None and impact in ordered:
                ordered.remove(impact)
                ordered.insert(0, impact)
            picked = sorted(ordered[:n])
        else:
            picked = list(keys)
            end = usable - 1
            if not loop and end not in picked:
                picked.append(end)  # a one shot action shows where it ends
            while len(picked) < n:
                bounds = picked + ([end] if picked[-1] < end else [])
                gaps = [(bounds[i + 1] - bounds[i], i) for i in range(len(bounds) - 1)]
                if not gaps or max(gaps)[0] < 2:
                    if picked[-1] < end:
                        picked.append(end)
                        continue
                    break  # every clip frame is used
                width, i = max(gaps, key=lambda g: (g[0], -g[1]))
                picked.append(bounds[i] + width // 2)
                picked.sort()
            while len(picked) < n:  # a clip shorter than n frames: repeat the last
                picked.append(picked[-1])
    impact = meta.get("impact_frame")
    idx = None
    if impact is not None and impact in picked:
        idx = picked.index(impact)
    return {"frames": picked, "labels": [labels_at.get(f, "") for f in picked], "impact_index": idx}
