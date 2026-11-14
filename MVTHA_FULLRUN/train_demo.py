#!/usr/bin/env python
"""Train MVTHA on the procedural multi-view skeleton-action dataset (no downloads needed).

    python train_demo.py                  # ~15 epochs, a few minutes on a laptop CPU
    python train_demo.py --quick          # 5 epochs smoke run
    python train_demo.py --no-align       # ablation: drop the canonical-view alignment
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mvtha_demo import synth
from mvtha_demo.dataset import SyntheticMultiViewDataset
from mvtha_demo.engine import build_model, evaluate, pick_device, save_checkpoint


def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="checkpoints/mvtha_demo.pt")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--train-per-class", type=int, default=150)
    p.add_argument("--val-per-class", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--views", type=int, default=3, help="camera views per sample")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--persons", type=int, default=1, help="person slots M (extra slots are empty)")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=1, help="number of APAM->HMSAM->MS-TC blocks")
    p.add_argument("--no-align", action="store_true", help="disable canonical-view alignment")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="tiny run for a smoke test")
    return p


def train(args):
    if args.quick:
        args.epochs, args.train_per_class = 5, 60
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    cfg = dict(num_class=synth.NUM_CLASSES, num_person=args.persons, channels=args.channels,
               num_blocks=args.blocks, align_views=not args.no_align, views=args.views,
               frames=args.frames, classes=synth.CLASS_NAMES, dropout=0.1)
    model = build_model(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())

    train_ds = SyntheticMultiViewDataset(args.train_per_class, seed=args.seed + 1, num_views=args.views,
                                         frames=args.frames, num_person=args.persons)
    val_ds = SyntheticMultiViewDataset(args.val_per_class, seed=args.seed + 1000, num_views=args.views,
                                       frames=args.frames, fixed=True, num_person=args.persons)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    val_dl = DataLoader(val_ds, 64, shuffle=False, num_workers=args.workers)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl),
                                                pct_start=0.2)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    print(f"device={device}  params={n_par / 1e3:.0f}k  train={len(train_ds)}  val={len(val_ds)}  "
          f"views={args.views}  align={cfg['align_views']}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best, t0 = -1.0, time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = seen = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            seen += len(y)
        val_acc, _, _ = evaluate(model, val_dl, device)
        flag = ""
        if val_acc >= best:
            best = val_acc
            save_checkpoint(args.out, model, cfg, extra={"val_acc": val_acc, "epoch": ep})
            flag = "  *saved*"
        print(f"epoch {ep:2d}/{args.epochs}  loss {loss_sum / seen:.3f}  train acc {correct / seen:.3f}  "
              f"val acc {val_acc:.3f}  ({time.time() - t0:.0f}s){flag}", flush=True)
    print(f"best val acc {best:.3f}  ->  {args.out}")
    return args.out


if __name__ == "__main__":
    train(get_parser().parse_args())
