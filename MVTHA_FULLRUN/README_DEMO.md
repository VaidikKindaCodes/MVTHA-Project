# MVTHA – working multi-view demo

This replaces the shape-only prototype with a model that actually **trains and recognises actions
from several camera views of a skeleton**, plus a self-contained demo that needs **no dataset download**.

```bash
pip install -r requirements_demo.txt
python test_mvtha.py     # 1. verify the model (≈1–2 min on CPU)
python demo.py           # 2. trains a small model on first run (~5–10 min CPU), then runs the full demo
```

`demo.py` writes to `demo_outputs/`:

| Output | What it shows |
|---|---|
| console table | accuracy with 1 / 2 / 3 views, on camera angles seen in training **and on unseen side/back views** |
| `confusion_unseen_views.png` | confusion matrix on unseen angles |
| `showcase_*.gif` | the same performance from 3 cameras (with per-view occlusion), the model's top-5 probabilities, and the fusion weight given to each view |
| `stream_demo.gif` | six different actions back to back, classified with a sliding window; truth vs. prediction timeline |
| `summary.json` | numbers from the run |

Other commands:

```bash
python demo.py --actions clap kick squat --yaws -90 0 90    # choose actions / camera angles
python demo.py --file my_skeleton.npy                       # classify your own [T,25,3] or [K,T,25,3] array
python train_demo.py --quick                                # 5-epoch smoke training
python train_demo.py --no-align                             # ablation: remove the canonical-view alignment
```

## What changed

| File | Change |
|---|---|
| `model/mvtha.py` | **Rewritten.** Real multi-view input `[N, K, C, T, V, M]` (also accepts the old `[N, C, T, V, M]`). Same module names and pipeline as before. |
| `test_mvtha.py` | **Rewritten.** Shapes, gradients, occlusion handling, camera invariance, and a "can it learn" test. |
| `mvtha_demo/synth.py` | New. 12 procedural actions in the NTU 25-joint layout; virtual cameras with noise and self-occlusion. |
| `mvtha_demo/dataset.py`, `engine.py`, `viz.py` | New. Dataset, train/eval helpers, GIF/plot rendering. |
| `train_demo.py`, `demo.py` | New. |

`model/ctrgcn.py`, `main.py`, the feeders and configs are untouched. The old `mvtha.py` stays in git history.

## How the model works

```
K views [N,K,3,T,25,M]
 └ ViewTransformation   per view: centre, drop missing joints, rotate to a canonical (shoulder/hip-aligned)
                        viewpoint, 4 streams (position, root-relative, velocity, bone) → 64 ch
 └ SpatialMix           message passing over the skeleton graph (adjacency initialised from the bones, learned)
 └ MultiViewTransformer temporal self-attention inside each view → cross-view attention at every frame
 └ ViewFusion           per-frame attention over the K views, biased towards views that see more joints
 └ APAM                 per-frame softmax attention over joints
 └ HMSAM                channel attention from temporal pooling at 1, 2 and 4 bins
 └ MultiScaleTemporalConv   dilated temporal convs (1, 2, 3) + residual
 └ temporal pool → mask absent persons → linear classifier → [N, num_class] logits
```

Deviations from the old prototype (all deliberate): real multi-view fusion instead of three coordinate "views";
joint-wise temporal pooling in the classifier (keeps *which* joint moved, so left- vs right-hand waving is
distinguishable; `head_pool="global"` restores plain global pooling); a graph-mixing layer so joints can exchange
information.  Joints that are exactly `(0,0,0)` are treated as missing (this also holds for NTU's absent second person).

## The demo data

Twelve actions: `wave_right, wave_left, clap, jumping_jack, squat, sit_stand, jump, kick, walk, bow, drink, punch`.
Each clip is randomised (speed, amplitude, body size, per-limb style). Every camera view gets its own noise (σ = 3 cm)
and self-occlusion: joints behind the torso from that camera's point of view are dropped in short bursts (≈70 % of
joints visible per view on average). Training uses camera yaw within ±60°; evaluation additionally uses
70°–120° side/back views the model has never seen.

## Honest caveats

* The accuracy you see is on this **synthetic** dataset. It says the pipeline and model work; it is **not** a
  result on NTU RGB+D and not a reproduction of the MVTHA paper.
* The architecture is a compact interpretation of MVTHA (≈100k parameters), not the paper's exact equations.
* View invariance mainly comes from the explicit canonical-view alignment in `ViewTransformation`
  (try `--no-align` to see how much it matters). With real data the alignment is only as good as the skeleton tracking.

## Using real data (next step)

`model.mvtha.Model` still plugs into `main.py` (`--model model.mvtha.Model`). Standard NTU feeders give one view
(`K = 1`). For real multi-view training, group NTU samples that share `SsssPpppRrrrAaaa` but differ in camera
(`C001/C002/C003`) and stack them into `[K, C, T, V, M]`; a feeder that does this is the one piece still to write.
NTU's cross-view protocol (train on cameras 2 & 3, test on camera 1) is the real-data analogue of the
seen/unseen evaluation here.
