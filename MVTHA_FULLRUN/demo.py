#!/usr/bin/env python
"""
MVTHA demo: recognise actions from multiple camera views of a skeleton.

    python demo.py                      # full demo (trains a small model first if no checkpoint exists)
    python demo.py --showcase 6         # more example GIFs
    python demo.py --actions clap kick  # showcase specific actions
    python demo.py --file my_skel.npy   # classify your own skeleton, [T,25,3] or [K,T,25,3]
    python demo.py --retrain            # force re-training

Everything is written to ./demo_outputs/ .
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from mvtha_demo import synth
from mvtha_demo.dataset import SyntheticMultiViewDataset
from mvtha_demo.engine import (confusion_matrix, evaluate, load_checkpoint, pick_device, predict_probs)

NAMES = synth.CLASS_NAMES


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/mvtha_demo.pt")
    p.add_argument("--out", default="demo_outputs")
    p.add_argument("--device", default="auto")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--showcase", type=int, default=4, help="number of example GIFs")
    p.add_argument("--actions", nargs="*", choices=NAMES, help="actions for the showcase GIFs")
    p.add_argument("--yaws", type=float, nargs="+", default=[-85, 10, 100],
                   help="camera yaws (deg) for the showcase / stream; two of the defaults are side views")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-gifs", action="store_true", help="skip GIF/PNG rendering")
    p.add_argument("--file", help=".npy skeleton to classify")
    return p.parse_args()


def hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# --------------------------------------------------------------------------- evaluation
def run_evaluation(model, cfg, device, out_dir, make_png):
    hr("1. Accuracy: held-out clips, fresh camera positions, noise and occlusion")
    K = cfg["views"]
    rows, cms = {}, {}
    for mode, label in (("seen", "camera angles like training (|yaw| <= 60°)"),
                        ("unseen", "UNSEEN side/back views (70° <= |yaw| <= 120°)")):
        ds = SyntheticMultiViewDataset(25, seed=2024, num_views=K, frames=cfg["frames"], yaw_mode=mode,
                                       fixed=True, num_person=cfg["num_person"])
        dl = DataLoader(ds, 64, shuffle=False)
        rows[label] = []
        for v in range(1, K + 1):
            acc, preds, labels = evaluate(model, dl, device, num_views=v)
            rows[label].append(acc)
            if v == K:
                cms[mode] = confusion_matrix(preds, labels, len(NAMES))
    print(f"{'':50s}" + "".join(f"{v} view{'s' if v > 1 else ' ':<3s}  " for v in range(1, K + 1)))
    for label, accs in rows.items():
        print(f"{label:50s}" + "".join(f"{a * 100:6.1f}%    " for a in accs))
    print("\n  Each view is independently noisy and self-occluded (far-side limbs vanish).  Compare the columns to\n"
          "  see what extra views buy you; the 'unseen' row tests viewpoints the model never trained on.")

    cm = cms["unseen"]
    per_class = cm.diagonal() / np.maximum(cm.sum(1), 1)
    print("\nPer-class accuracy (all views, unseen angles):")
    for i in range(0, len(NAMES), 4):
        print("   " + "   ".join(f"{NAMES[j]:13s}{per_class[j] * 100:5.0f}%" for j in range(i, min(i + 4, len(NAMES)))))
    off = [(cm[i, j], NAMES[i], NAMES[j]) for i in range(len(NAMES)) for j in range(len(NAMES)) if i != j and cm[i, j]]
    off.sort(reverse=True)
    if off:
        print("Most frequent confusions: " + ", ".join(f"{a}→{b} ({n}x)" for n, a, b in off[:3]))
    if make_png:
        from mvtha_demo import viz
        path = os.path.join(out_dir, "confusion_unseen_views.png")
        viz.save_confusion(cm, path, title="MVTHA on unseen camera angles (all views)")
        print(f"[saved] {path}")
    return {"accuracy": {k: v for k, v in rows.items()}, "per_class_unseen": dict(zip(NAMES, per_class.tolist()))}


# --------------------------------------------------------------------------- showcase
def run_showcase(model, cfg, device, args):
    hr("2. Showcase: fresh performances seen from three cameras (two are side views)")
    rng = np.random.default_rng(args.seed)
    labels = [NAMES.index(a) for a in args.actions] if args.actions else \
        list(rng.choice(len(NAMES), size=min(args.showcase, len(NAMES)), replace=False))
    yaws = list(args.yaws)[:cfg["views"]]
    hits = 0
    for n, lab in enumerate(labels):
        J = synth.generate_sequence(int(lab), cfg["frames"], rng)
        views = synth.make_views(J, yaws, rng)
        probs = predict_probs(model, synth.to_model_input(views, cfg["num_person"]), device)[0]
        w = model.view_weights[0].mean(dim=(0, 1)).cpu().numpy() if model.view_weights is not None else None
        top = np.argsort(-probs)[:3]
        ok = int(np.argmax(probs)) == int(lab)
        hits += ok
        vis = [(np.abs(v).sum(0) > 0).mean() * 100 for v in views]
        print(f"  true={NAMES[lab]:13s} pred={NAMES[top[0]]:13s} {'OK ' if ok else 'ERR'}  "
              f"top3: " + ", ".join(f"{NAMES[i]} {probs[i] * 100:.0f}%" for i in top) +
              f"   | joints visible per view: " + "/".join(f"{v:.0f}%" for v in vis) +
              (f" | fusion weights: " + "/".join(f"{x:.2f}" for x in w) if w is not None else ""))
        if not args.no_gifs:
            from mvtha_demo import viz
            path = os.path.join(args.out, f"showcase_{n + 1}_{NAMES[lab]}.gif")
            viz.render_clip_gif(views, yaws, probs, int(lab), path, view_weights=w)
            print(f"      [saved] {path}")
    print(f"  -> {hits}/{len(labels)} correct")


# --------------------------------------------------------------------------- streaming
def run_stream(model, cfg, device, args):
    hr("3. Streaming recognition: six different actions back-to-back, sliding window")
    rng = np.random.default_rng(args.seed + 1)
    W = cfg["frames"]
    seq = [int(i) for i in rng.choice(len(NAMES), size=6, replace=False)]
    J, truth = synth.generate_stream(seq, W, rng)
    yaws = list(args.yaws)[:cfg["views"]]
    views = synth.make_views(J, yaws, rng)                                  # [K,3,Ttot,V]
    Ttot, stride = views.shape[2], 4
    starts = list(range(0, Ttot - W + 1, stride))
    batch = np.stack([synth.to_model_input(views[:, :, s:s + W], cfg["num_person"]) for s in starts])
    probs = np.concatenate([predict_probs(model, batch[i:i + 64], device) for i in range(0, len(batch), 64)])
    centres = np.array(starts) + W / 2
    nearest = np.abs(np.arange(Ttot)[:, None] - centres[None]).argmin(1)    # window centred closest to each frame
    pred_frames = probs.argmax(1)[nearest]
    conf_frames = probs.max(1)[nearest]
    acc = float((pred_frames == truth).mean())
    print("  script : " + " -> ".join(NAMES[i] for i in seq))
    seg = [pred_frames[i * W:(i + 1) * W] for i in range(len(seq))]
    print("  MVTHA  : " + " -> ".join(NAMES[np.bincount(s, minlength=len(NAMES)).argmax()] for s in seg)
          + "   (majority label per segment)")
    print(f"  frame-level accuracy over the stream: {acc * 100:.1f}%  (frames near a switch are inherently ambiguous)")
    if not args.no_gifs:
        from mvtha_demo import viz
        path = os.path.join(args.out, "stream_demo.gif")
        viz.render_stream_gif(views, yaws, truth, pred_frames, conf_frames, path)
        print(f"  [saved] {path}")
    return acc


# --------------------------------------------------------------------------- user file
def run_file(model, cfg, device, path):
    hr(f"Classifying {path}")
    a = np.load(path).astype(np.float32)
    if a.ndim == 3:                       # [T,V,C] single view
        a = a[None]
    assert a.ndim == 4 and a.shape[2] == 25 and a.shape[3] == 3, "expected [T,25,3] or [K,T,25,3]"
    a = a.transpose(0, 3, 1, 2)           # [K,3,T,V]
    idx = np.linspace(0, a.shape[2] - 1, cfg["frames"]).round().astype(int)   # resample to training length
    x = synth.to_model_input(a[:, :, idx], cfg["num_person"])
    probs = predict_probs(model, x, device)[0]
    print(f"  {a.shape[0]} view(s), {a.shape[2]} frames (resampled to {cfg['frames']}); all-zero joints = missing")
    for i in np.argsort(-probs)[:5]:
        print(f"   {NAMES[i]:14s} {probs[i] * 100:5.1f}%  " + "#" * int(probs[i] * 40))


# --------------------------------------------------------------------------- main
def main():
    args = parse()
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    if args.retrain or not os.path.exists(args.ckpt):
        hr("No checkpoint found - training a small MVTHA model on the synthetic dataset"
           if not args.retrain else "Re-training")
        import train_demo
        targs = train_demo.get_parser().parse_args(["--out", args.ckpt, "--device", args.device])
        train_demo.train(targs)

    model, cfg = load_checkpoint(args.ckpt, device)
    n_par = sum(p.numel() for p in model.parameters())
    hr("MVTHA: Multi-View Transformer with Hierarchical Attention")
    print(f"  device={device}   parameters={n_par / 1e3:.0f}k   classes={len(NAMES)}   views per sample={cfg['views']}   "
          f"view alignment={'on' if cfg['align_views'] else 'off'}")
    print("  actions: " + ", ".join(NAMES))

    if args.file:
        run_file(model, cfg, device, args.file)
        return

    summary = run_evaluation(model, cfg, device, args.out, not args.no_gifs)
    run_showcase(model, cfg, device, args)
    summary["stream_frame_accuracy"] = run_stream(model, cfg, device, args)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    hr(f"Done. Open the GIFs / PNG in ./{args.out}/")


if __name__ == "__main__":
    main()
