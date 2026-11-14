"""Visualisation helpers: multi-view skeleton GIFs, streaming timeline, confusion matrix.
Needs matplotlib (+ pillow for GIFs).  No torch import here."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

from .synth import EDGES, CLASS_NAMES, NUM_JOINTS, make_views

_BONE_COLORS = {"torso": "#444444", "left": "#1f77b4", "right": "#d62728"}
_LEFT_J = {4, 5, 6, 7, 21, 22, 12, 13, 14, 15}
_RIGHT_J = {8, 9, 10, 11, 23, 24, 16, 17, 18, 19}


def _edge_color(i, j):
    if i in _LEFT_J or j in _LEFT_J:
        return _BONE_COLORS["left"]
    if i in _RIGHT_J or j in _RIGHT_J:
        return _BONE_COLORS["right"]
    return _BONE_COLORS["torso"]


class _SkeletonPanel:
    """One camera view; artists are created once and updated per frame."""

    def __init__(self, ax, view, title):
        self.view = view                                   # [3, T, V]; hidden joints == 0
        self.vis = np.abs(view).sum(0) > 0                 # [T, V]
        pts = view[:, self.vis]
        cx, cy = (pts[0].min() + pts[0].max()) / 2, (pts[1].min() + pts[1].max()) / 2
        half = max(1.0, (pts[1].max() - pts[1].min()) / 2 + 0.1)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#bbbbbb")
        ax.set_title(title, fontsize=9)
        self.lines = [ax.plot([], [], "-", color=_edge_color(i, j), lw=2.4, solid_capstyle="round")[0]
                      for i, j in EDGES]
        self.dots, = ax.plot([], [], "o", color="#222222", ms=3.2)
        self.txt = ax.text(0.03, 0.03, "", transform=ax.transAxes, fontsize=7.5, color="#666666")

    def update(self, t):
        v, vis = self.view[:, t], self.vis[t]
        for ln, (i, j) in zip(self.lines, EDGES):
            if vis[i] and vis[j]:
                ln.set_data([v[0, i], v[0, j]], [v[1, i], v[1, j]])
            else:
                ln.set_data([], [])
        self.dots.set_data(v[0, vis], v[1, vis])
        self.txt.set_text(f"{int(vis.sum())}/{NUM_JOINTS} joints visible")


def _save(anim, path, fps):
    anim.save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close("all")


def render_clip_gif(views, yaws, probs, true_label, path, class_names=CLASS_NAMES,
                    view_weights=None, fps=10):
    """views [K,3,T,V], probs [C] (softmax of the multi-view prediction)."""
    K, _, T, _ = views.shape
    fig = plt.figure(figsize=(3.1 * K + 3.4, 3.9))
    gs = fig.add_gridspec(1, K + 1, width_ratios=[1] * K + [1.15], wspace=0.12)
    panels = []
    for k in range(K):
        ax = fig.add_subplot(gs[0, k])
        w = "" if view_weights is None else f"   (fusion weight {view_weights[k]:.2f})"
        panels.append(_SkeletonPanel(ax, views[k], f"view {k + 1}: yaw {yaws[k]:+.0f}°{w}"))
    ax = fig.add_subplot(gs[0, K])
    top = np.argsort(-probs)[:5][::-1]
    cols = ["#2ca02c" if i == int(np.argmax(probs)) else "#c7c7c7" for i in top]
    ax.barh([class_names[i] for i in top], probs[top], color=cols)
    ax.set_xlim(0, 1); ax.tick_params(labelsize=8)
    ax.set_title("MVTHA prediction", fontsize=9)
    for y, i in enumerate(top):
        ax.text(min(probs[i] + 0.02, 0.8), y, f"{probs[i] * 100:.0f}%", va="center", fontsize=8)
    pred = int(np.argmax(probs))
    ok = pred == true_label
    fig.suptitle(f"true: {class_names[true_label]}    predicted: {class_names[pred]}  "
                 f"({probs[pred] * 100:.0f}%)  {'✓' if ok else '✗'}",
                 fontsize=11, color="#1a7f37" if ok else "#b42318")
    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.04, right=0.97)
    anim = FuncAnimation(fig, lambda t: [p.update(t) for p in panels], frames=T, interval=1000 / fps)
    _save(anim, path, fps)


def render_stream_gif(views, yaws, true_frames, pred_frames, conf_frames, path,
                      class_names=CLASS_NAMES, fps=10, step=2):
    """A long multi-action recording classified with a sliding window."""
    K, _, T, _ = views.shape
    cmap = ListedColormap(plt.get_cmap("tab20")(np.arange(len(class_names))))
    fig = plt.figure(figsize=(3.1 * K, 5.6))
    gs = fig.add_gridspec(2, K, height_ratios=[3.2, 1], hspace=0.25, wspace=0.08)
    panels = [_SkeletonPanel(fig.add_subplot(gs[0, k]), views[k], f"view {k + 1}: yaw {yaws[k]:+.0f}°")
              for k in range(K)]
    ax = fig.add_subplot(gs[1, :])
    ax.imshow(np.vstack([true_frames, pred_frames]), aspect="auto", cmap=cmap, vmin=-0.5,
              vmax=len(class_names) - 0.5, interpolation="nearest")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["truth", "MVTHA"], fontsize=8)
    ax.set_xlabel("frame", fontsize=8); ax.tick_params(labelsize=8)
    cursor = ax.axvline(0, color="k", lw=1.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(len(class_names))]
    ax.legend(handles, class_names, ncol=6, fontsize=6.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.45), frameon=False)
    title = fig.suptitle("", fontsize=11)
    fig.subplots_adjust(top=0.9, bottom=0.2, left=0.04, right=0.98)

    def draw(f):
        t = f * step
        for p in panels:
            p.update(t)
        cursor.set_xdata([t, t])
        ok = pred_frames[t] == true_frames[t]
        title.set_text(f"now doing: {class_names[true_frames[t]]}   →   MVTHA says: "
                       f"{class_names[pred_frames[t]]} ({conf_frames[t] * 100:.0f}%)")
        title.set_color("#1a7f37" if ok else "#b42318")

    anim = FuncAnimation(fig, draw, frames=T // step, interval=1000 / fps)
    _save(anim, path, fps)


def save_confusion(cm, path, class_names=CLASS_NAMES, title="Confusion matrix"):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    n = len(class_names)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def contact_sheet(path, generate_sequence, num_classes, T=32, frames=(0, 8, 16, 24), seed=0):
    """Frontal-view snapshots of every action (what the synthetic dataset looks like)."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(num_classes, len(frames), figsize=(1.6 * len(frames), 1.55 * num_classes))
    for c in range(num_classes):
        J = generate_sequence(c, T, rng)
        v = make_views(J, [0.0], rng, noise=0.0, occlusion=False)[0]
        for k, f in enumerate(frames):
            ax = axes[c, k]
            _SkeletonPanel(ax, v, "").update(f)
            ax.set_title("")
            for t in ax.texts:
                t.set_text("")
            if k == 0:
                ax.set_ylabel(CLASS_NAMES[c], rotation=0, ha="right", va="center", fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=90); plt.close(fig)
