"""
Procedural multi-view skeleton action generator (NumPy only).

Produces skeletons in the NTU RGB+D 25-joint layout so that the same model can later be
trained on real NTU data.  Each action is a small kinematic program (limb directions as a
function of time).  A *view* is a virtual camera orbiting the person (yaw rotation); every
view gets its own sensor noise and self-occlusion (joints on the far side of the body are
dropped), so no single view sees everything -- that is what makes multi-view fusion useful.

Coordinates (body frame): x = person's left, y = up, z = facing direction, units = metres.
"""
from collections import OrderedDict

import numpy as np

NUM_JOINTS = 25
D2R = np.pi / 180.0

# NTU joint indices (0-based)
SPINE_BASE, SPINE_MID, NECK, HEAD = 0, 1, 2, 3
L_SHOULDER, L_ELBOW, L_WRIST, L_HAND = 4, 5, 6, 7
R_SHOULDER, R_ELBOW, R_WRIST, R_HAND = 8, 9, 10, 11
L_HIP, L_KNEE, L_ANKLE, L_FOOT = 12, 13, 14, 15
R_HIP, R_KNEE, R_ANKLE, R_FOOT = 16, 17, 18, 19
SPINE_SHOULDER, L_TIP, L_THUMB, R_TIP, R_THUMB = 20, 21, 22, 23, 24

# Bones used for drawing (also used as the skeleton graph by the model).
EDGES = [(0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
         (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
         (0, 12), (12, 13), (13, 14), (14, 15), (0, 16), (16, 17), (17, 18), (18, 19)]

LEFT, RIGHT = 1.0, -1.0


# ----------------------------------------------------------------------------- geometry
def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _arr(x, T):
    return np.broadcast_to(np.asarray(x, dtype=np.float64), (T,)).copy()


def limb_dir(side, a, f, T):
    """Arm direction: a = abduction from hanging-down (0 down, 90 sideways, 180 up),
    f = azimuth of the arm plane (0 = in the coronal plane, 90 = forward)."""
    a, f = _arr(a, T), _arr(f, T)
    return np.stack([side * np.sin(a) * np.cos(f), -np.cos(a), np.sin(a) * np.sin(f)], -1)


def sag(side, theta, abd, T):
    """Limb swung in the sagittal plane: theta>0 forward, theta<0 backward (rad from hanging down)."""
    theta, abd = _arr(theta, T), _arr(abd, T)
    return _unit(np.stack([side * np.sin(abd), -np.cos(theta) * np.cos(abd),
                           np.sin(theta) * np.cos(abd)], -1))


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def _drop(th1, th2):
    """How far the pelvis must drop so that both feet stay on the floor (leg = 0.45 + 0.45 m)."""
    return -(0.9 - 0.45 * np.cos(th1) - 0.45 * np.cos(th2))


def _rest(T):
    return dict(
        root=np.zeros((T, 3)), lean=np.zeros(T), twist=np.zeros(T),
        la_u=limb_dir(LEFT, 8 * D2R, 0, T), la_f=sag(LEFT, 25 * D2R, 0.15, T),
        ra_u=limb_dir(RIGHT, 8 * D2R, 0, T), ra_f=sag(RIGHT, 25 * D2R, 0.15, T),
        ll_u=sag(LEFT, 0, 0.05, T), ll_f=sag(LEFT, 0, 0.05, T),
        rl_u=sag(RIGHT, 0, 0.05, T), rl_f=sag(RIGHT, 0, 0.05, T))


def build_pose(P, scale=1.0):
    """Forward kinematics: parameter dict -> joints [T, 25, 3]."""
    T = P["lean"].shape[0]
    R = _rot_y(P["twist"]) @ _rot_x(P["lean"])                  # torso rotation [T,3,3]
    base = P["root"] + np.array([0.0, 0.95, 0.0])
    J = np.zeros((T, NUM_JOINTS, 3))

    def up(off):                                                  # torso-attached point
        return base + np.einsum("tij,j->ti", R, np.asarray(off, dtype=np.float64))

    def rdir(d):                                                  # torso-local direction -> world
        return np.einsum("tij,tj->ti", R, d)

    J[:, SPINE_BASE] = base
    J[:, SPINE_MID] = up((0, .15, 0))
    J[:, SPINE_SHOULDER] = up((0, .40, 0))
    J[:, NECK] = up((0, .50, 0))
    J[:, HEAD] = up((0, .65, 0))
    J[:, L_SHOULDER] = up((.20, .40, 0))
    J[:, R_SHOULDER] = up((-.20, .40, 0))

    for sh, el, wr, hd, tip, th, u, f, side in (
            (L_SHOULDER, L_ELBOW, L_WRIST, L_HAND, L_TIP, L_THUMB, P["la_u"], P["la_f"], LEFT),
            (R_SHOULDER, R_ELBOW, R_WRIST, R_HAND, R_TIP, R_THUMB, P["ra_u"], P["ra_f"], RIGHT)):
        J[:, el] = J[:, sh] + 0.30 * rdir(u)
        d2 = rdir(f)
        J[:, wr] = J[:, el] + 0.27 * d2
        J[:, hd] = J[:, wr] + 0.08 * d2
        J[:, tip] = J[:, wr] + 0.16 * d2
        J[:, th] = J[:, wr] + 0.06 * d2 + np.array([side * 0.03, 0, 0.03])

    for hp, kn, an, ft, u, f, side in (
            (L_HIP, L_KNEE, L_ANKLE, L_FOOT, P["ll_u"], P["ll_f"], LEFT),
            (R_HIP, R_KNEE, R_ANKLE, R_FOOT, P["rl_u"], P["rl_f"], RIGHT)):
        J[:, hp] = base + np.array([side * 0.10, -0.05, 0.0])
        J[:, kn] = J[:, hp] + 0.45 * u
        J[:, an] = J[:, kn] + 0.45 * f
        J[:, ft] = J[:, an] + np.array([0.0, -0.03, 0.12])
    return J * scale


# ----------------------------------------------------------------------------- actions
# Each action: fn(p, amp, T) -> parameter dict.  p = phase [T] (rad), amp ~ U(0.85, 1.15).
def _wave(side):
    def fn(p, amp, T):
        d = _rest(T)
        a2 = (150 + 35 * amp * np.sin(p)) * D2R
        k = "la" if side == LEFT else "ra"
        d[k + "_u"] = limb_dir(side, 140 * D2R, 0.15, T)
        d[k + "_f"] = limb_dir(side, a2, 0.2, T)
        return d
    return fn


def _clap(p, amp, T):
    d = _rest(T)
    s = 0.5 + 0.5 * np.sin(p)
    for side, k in ((LEFT, "la"), (RIGHT, "ra")):
        d[k + "_u"] = limb_dir(side, 60 * D2R, 80 * D2R, T)
        d[k + "_f"] = limb_dir(side, 90 * D2R, (95 + 62 * min(amp, 1.1) * s) * D2R, T)
    return d


def _jumping_jack(p, amp, T):
    d = _rest(T)
    s = np.clip((0.5 + 0.5 * np.sin(p)) * amp, 0, 1)
    a = (12 + 160 * s) * D2R
    for side, k in ((LEFT, "la"), (RIGHT, "ra")):
        d[k + "_u"] = limb_dir(side, a, 0.1, T)
        d[k + "_f"] = limb_dir(side, a, 0.1, T)
    abd = (3 + 24 * s) * D2R
    for side, k in ((LEFT, "ll"), (RIGHT, "rl")):
        d[k + "_u"] = sag(side, 0, abd, T)
        d[k + "_f"] = sag(side, 0, abd, T)
    d["root"][:, 1] = 0.06 * s
    return d


def _squat(p, amp, T):
    d = _rest(T)
    s = 0.5 - 0.5 * np.cos(p)
    th = 65 * D2R * amp * s
    for side, k in ((LEFT, "l"), (RIGHT, "r")):
        d[k + "l_u"] = sag(side, th, 0.05, T)
        d[k + "l_f"] = sag(side, -th, 0.05, T)
        d[k + "a_u"] = limb_dir(side, 25 * D2R + 60 * D2R * s, 80 * D2R, T)
        d[k + "a_f"] = limb_dir(side, 25 * D2R + 62 * D2R * s, 85 * D2R, T)
    d["root"][:, 1] = _drop(th, -th)
    d["lean"] = 0.45 * th
    return d


def _sit_stand(p, amp, T):
    d = _rest(T)
    s = np.clip(1.5 * (0.5 - 0.5 * np.cos(p)) - 0.25, 0, 1)
    th = 90 * D2R * min(amp, 1.05) * s
    for side, k in ((LEFT, "l"), (RIGHT, "r")):
        d[k + "l_u"] = sag(side, th, 0.05, T)
        d[k + "l_f"] = sag(side, 0 * th, 0.05, T)
        d[k + "a_u"] = sag(side, 20 * D2R * s, 0.15, T)
        d[k + "a_f"] = sag(side, 55 * D2R * s + 25 * D2R, 0.15, T)
    d["root"][:, 1] = _drop(th, 0 * th)
    d["lean"] = 0.35 * s
    return d


def _jump(p, amp, T):
    d = _rest(T)
    air = np.maximum(0, np.sin(p))
    cr = np.maximum(0, -np.sin(p))
    thc = 50 * D2R * cr * amp
    for side, k in ((LEFT, "l"), (RIGHT, "r")):
        d[k + "l_u"] = sag(side, thc + 15 * D2R * air, 0.05, T)
        d[k + "l_f"] = sag(side, -thc - 25 * D2R * air, 0.05, T)
        arm = -50 * D2R * cr + 120 * D2R * air
        d[k + "a_u"] = sag(side, arm, 0.2, T)
        d[k + "a_f"] = sag(side, arm, 0.2, T)
    d["root"][:, 1] = 0.30 * amp * air + _drop(thc, -thc)
    d["lean"] = 0.35 * cr
    return d


def _kick(p, amp, T):
    d = _rest(T)
    k = np.maximum(0, np.sin(p))
    th1 = 70 * D2R * amp * k
    th2 = th1 - 70 * D2R * np.sin(np.pi * k)
    d["rl_u"] = sag(RIGHT, th1, 0.05, T)
    d["rl_f"] = sag(RIGHT, th2, 0.05, T)
    d["ll_u"] = sag(LEFT, 8 * D2R * k, 0.05, T)
    d["ll_f"] = sag(LEFT, 8 * D2R * k, 0.05, T)
    d["lean"] = -0.25 * k
    d["root"][:, 1] = _drop(8 * D2R * k, 8 * D2R * k) * 0.5
    for side, kk in ((LEFT, "la"), (RIGHT, "ra")):
        d[kk + "_u"] = limb_dir(side, 35 * D2R + 25 * D2R * k, 0.0, T)
        d[kk + "_f"] = limb_dir(side, 35 * D2R + 25 * D2R * k, 0.0, T)
    return d


def _walk(p, amp, T):
    d = _rest(T)
    for side, k, ph in ((LEFT, "l", p), (RIGHT, "r", p + np.pi)):
        th1 = 30 * D2R * amp * np.sin(ph)
        th2 = th1 - 45 * D2R * np.maximum(0, np.cos(ph))
        d[k + "l_u"] = sag(side, th1, 0.05, T)
        d[k + "l_f"] = sag(side, th2, 0.05, T)
        arm = -25 * D2R * amp * np.sin(ph)
        d[k + "a_u"] = sag(side, arm, 0.12, T)
        d[k + "a_f"] = sag(side, arm + 30 * D2R, 0.12, T)
    d["root"][:, 1] = -0.025 * np.abs(np.sin(p))
    d["lean"] = np.full(T, 0.05)
    return d


def _bow(p, amp, T):
    d = _rest(T)
    s = 0.5 - 0.5 * np.cos(p)
    d["lean"] = 65 * D2R * amp * s
    return d


def _drink(p, amp, T):
    d = _rest(T)
    s = np.clip(0.5 + 0.8 * np.sin(p), 0, 1)
    d["ra_u"] = sag(RIGHT, (10 + 45 * amp * s) * D2R, 0.1, T)
    d["ra_f"] = sag(RIGHT, (25 + 140 * s) * D2R, 0.2 - 0.9 * s, T)   # forearm swings in towards the face
    d["lean"] = -0.12 * s
    return d


def _punch(p, amp, T):
    d = _rest(T)
    s = np.maximum(0, np.sin(p)) ** 0.7
    d["ra_u"] = sag(RIGHT, (30 + 60 * amp * s) * D2R, 0.2, T)
    d["ra_f"] = sag(RIGHT, (150 - 60 * amp * s) * D2R, 0.2, T)
    d["la_u"] = sag(LEFT, 30 * D2R, 0.2, T)
    d["la_f"] = sag(LEFT, 150 * D2R, 0.2, T)
    d["twist"] = -0.3 * s
    d["lean"] = 0.1 * s
    return d


# name -> (program, base number of cycles in a clip)
ACTIONS = OrderedDict([
    ("wave_right", (_wave(RIGHT), 4.0)),
    ("wave_left", (_wave(LEFT), 4.0)),
    ("clap", (_clap, 5.0)),
    ("jumping_jack", (_jumping_jack, 3.0)),
    ("squat", (_squat, 2.0)),
    ("sit_stand", (_sit_stand, 1.5)),
    ("jump", (_jump, 2.5)),
    ("kick", (_kick, 2.5)),
    ("walk", (_walk, 3.0)),
    ("bow", (_bow, 1.5)),
    ("drink", (_drink, 1.5)),
    ("punch", (_punch, 3.0)),
])
CLASS_NAMES = list(ACTIONS.keys())
NUM_CLASSES = len(CLASS_NAMES)


def generate_sequence(label, T, rng, cycles=None):
    """One clean, randomised performance of an action: joints [T, 25, 3] in the body frame."""
    fn, base_cycles = ACTIONS[CLASS_NAMES[label]]
    cyc = base_cycles * rng.uniform(0.8, 1.25) if cycles is None else cycles
    p = 2 * np.pi * cyc * np.linspace(0, 1, T, endpoint=False) + rng.uniform(0, 2 * np.pi)
    P = fn(p, rng.uniform(0.85, 1.15), T)
    for key in ("la_u", "la_f", "ra_u", "ra_f", "ll_u", "ll_f", "rl_u", "rl_f"):   # per-subject style
        P[key] = _unit(P[key] + rng.normal(0, 0.06, size=3))
    P["lean"] = P["lean"] + rng.normal(0, 0.03)
    P["twist"] = P["twist"] + rng.normal(0, 0.05)
    return build_pose(P, scale=rng.uniform(0.9, 1.1))


def generate_stream(labels, T_per_action, rng):
    """Several different actions back to back -> (joints [T_total,25,3], per-frame labels)."""
    segs = [generate_sequence(l, T_per_action, rng) for l in labels]
    return np.concatenate(segs, 0), np.repeat(np.asarray(labels), T_per_action)


# ----------------------------------------------------------------------------- cameras
def sample_yaws(rng, k, mode="seen"):
    """seen: U(-60,60) deg.   unseen: side/back views, |yaw| in U(70,120) deg."""
    if mode == "seen":
        return rng.uniform(-60, 60, size=k)
    return rng.choice([-1.0, 1.0], size=k) * rng.uniform(70, 120, size=k)


_TORSO = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]


def make_views(joints, yaws_deg, rng, noise=0.03, occlusion=True):
    """Project body-frame joints [T,25,3] into K camera views -> float32 [K, 3, T, 25].

    Each view: yaw rotation about the vertical axis, random placement in front of the camera,
    Gaussian sensor noise, and self-occlusion.  Occluded joints are exactly (0, 0, 0)."""
    T = joints.shape[0]
    out = np.zeros((len(yaws_deg), 3, T, NUM_JOINTS), dtype=np.float32)
    for k, yaw in enumerate(yaws_deg):
        phi = yaw * D2R
        c, s = np.cos(phi), np.sin(phi)
        X, Y, Z = joints[..., 0], joints[..., 1], joints[..., 2]
        cam = np.stack([c * X + s * Z, Y, -s * X + c * Z], 0)           # [3,T,V]; z = depth
        if occlusion:
            behind = cam[2] - cam[2][:, SPINE_MID:SPINE_MID + 1]       # [T,V]  (>0 = behind torso)
            p_hide = np.clip((behind - 0.04) / 0.18, 0, 1) * 0.9
            p_hide[:, [SPINE_BASE, SPINE_MID, SPINE_SHOULDER]] = 0.0
            p_hide[:, _TORSO] *= 0.25                                   # torso is usually inferred
            chunks = rng.random((int(np.ceil(T / 4)), NUM_JOINTS))      # contiguous 4-frame dropouts
            hidden = np.repeat(chunks, 4, axis=0)[:T] < p_hide
            hidden |= rng.random((T, NUM_JOINTS)) < 0.05                # random flicker
        else:
            hidden = np.zeros((T, NUM_JOINTS), dtype=bool)
        cam = cam + rng.normal(0, noise, size=cam.shape)
        cam = cam + np.array([rng.uniform(-.4, .4), rng.uniform(-.2, .2), rng.uniform(2.5, 4.5)])[:, None, None]
        cam[:, hidden] = 0.0
        out[k] = cam
    return out


def to_model_input(views, num_person=1):
    """[K,3,T,V] -> [K,3,T,V,M] (extra person slots are all-zero, like single-person NTU clips)."""
    K, C, T, V = views.shape
    x = np.zeros((K, C, T, V, num_person), dtype=np.float32)
    x[..., 0] = views
    return x
