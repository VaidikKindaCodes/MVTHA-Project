#!/usr/bin/env python
"""Tests for the MVTHA model.   python test_mvtha.py

  1. README smoke test: [2,3,64,25,2] -> [2,60], backward pass, every parameter gets a gradient
  2. multi-view input with occluded joints and an empty second person
  3. view invariance: rotating / translating the skeleton does not change the logits
  4. configuration variants
  5. the model can actually learn: it overfits a tiny synthetic multi-view set
"""
import math

import torch
import torch.nn as nn

from model.mvtha import Model


def test_shapes_and_backward():
    torch.manual_seed(0)
    model = Model(num_class=60, num_point=25, num_person=2, in_channels=3)
    x = torch.randn(2, 3, 64, 25, 2)
    out, stages = model(x, return_stages=True)
    print("Input shape:", x.shape)
    for name, shape in stages.items():
        print(f"{name} output shape: {tuple(shape)}")
    assert out.shape == (2, 60), out.shape
    out.square().mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing, f"no/invalid gradient for {missing}"
    print("Backward pass: successful (all parameters received finite gradients)")


def test_multiview_with_missing_data():
    torch.manual_seed(0)
    model = Model(num_class=12, num_point=25, num_person=2, in_channels=3)
    x = torch.randn(2, 3, 3, 32, 25, 2)          # [N, K, C, T, V, M]
    x[..., 1] = 0                                # second person absent
    x[:, 1, :, :, 5:9, :] = 0                    # joints occluded in view 1
    x[:, 2, :, 10:20] = 0                        # view 2 completely lost for 10 frames
    out = model(x)
    assert out.shape == (2, 12) and torch.isfinite(out).all()
    w = model.view_weights
    assert w.shape == (2, 2, 32, 3) and torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-4)
    print("Multi-view + occlusion + empty person: OK  (view weights sum to 1)")


def test_view_invariance():
    torch.manual_seed(1)
    model = Model(num_class=12, num_point=25, num_person=1, in_channels=3).eval()
    x = torch.randn(2, 3, 3, 24, 25, 1)
    phi = 1.1
    c, s = math.cos(phi), math.sin(phi)
    X, Y, Z = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    xr = torch.stack([c * X + s * Z + 0.7, Y - 0.3, -s * X + c * Z + 2.0], dim=2)
    with torch.no_grad():
        d = (model(x) - model(xr)).abs().max().item()
    assert d < 1e-3, f"logits changed by {d} under a rigid camera change"
    print(f"Invariance to camera yaw/translation: max logit change {d:.2e}")


def test_variants():
    x = torch.randn(2, 2, 3, 16, 25, 1)
    for kwargs in (dict(align_views=False), dict(num_blocks=2), dict(head_pool="global"),
                   dict(graph="graph.ntu_rgb_d.Graph", graph_args={"labeling_mode": "spatial"})):
        out = Model(num_class=7, num_person=1, **kwargs)(x)
        assert out.shape == (2, 7)
    out = Model(num_class=7, num_person=1)(x[:, 0])          # 5-D single-view input
    assert out.shape == (2, 7)
    print("Variants (no-align, 2 blocks, global pooling, CTR-GCN style kwargs, single view): OK")


def test_it_learns():
    from mvtha_demo.dataset import SyntheticMultiViewDataset
    torch.manual_seed(0)
    ds = SyntheticMultiViewDataset(3, seed=5, num_views=3, frames=32, fixed=True)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.tensor([ds[i][1] for i in range(len(ds))])
    model = Model(num_class=12, num_person=1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = None
    for step in range(60):
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    print(f"Overfit test: loss {first:.2f} -> {last:.2f}")
    assert last < 0.6 * first, "model failed to reduce the loss on a tiny dataset"


if __name__ == "__main__":
    for t in (test_shapes_and_backward, test_multiview_with_missing_data, test_view_invariance,
              test_variants, test_it_learns):
        print(f"\n--- {t.__name__}")
        t()
    print("\nAll tests passed.")
