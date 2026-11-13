# MVTHA intensive demo - results

> **Synthetic data.** Numbers describe the procedural 12-action NTU-layout dataset in `mvtha_demo/synth.py`, not NTU RGB+D and not the MVTHA paper.

## Headline (unseen camera angles, all views)

- accuracy **100.0%**, top-3 100.0%, macro-F1 1.000, kappa 1.000, MCC 1.000, ECE 0.074
- 142,576 parameters, 3.54 ms/clip at batch 32

## Accuracy by condition

| camera regime | views | acc | top-3 | macro-F1 | kappa | ECE |
|---|---|---|---|---|---|---|
| seen | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.037 |
| seen | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.032 |
| seen | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.032 |
| unseen | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.091 |
| unseen | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.077 |
| unseen | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.074 |

Streaming (all 12 actions back to back): frame accuracy 86.5% (93.6% away from switches).

## Figures

| file | shows |
|---|---|
| `confusion_matrices_grid.png` | confusion matrices, seen/unseen angles x 1/all views |
| `confusion_<regime>_<n>view(s).png` | each matrix on its own |
| `confusion_stream_frames.png` | frame-level confusion in the streaming test |
| `per_class_metrics.png` | precision / recall / F1 per action |
| `robustness.png` | accuracy vs. #views, camera yaw, noise, missing joints |
| `calibration.png` | reliability diagram + confidence histogram |
| `view_fusion_weights.png` | fusion weight vs. visible joints in each view |
| `predictions_gallery.png` | ground truth vs. what the model saw vs. its prediction, every action |
| `hard_cases.png` | the 12 hardest evaluation clips |

## GIFs

| file | shows |
|---|---|
| `grid_all_actions.gif` | all 12 actions at once with true -> predicted |
| `action_NN_<name>.gif` | ground truth, each noisy/occluded camera, and the probability for every class |
| `stream_all_actions.gif` | 12 actions back to back, classified live with a sliding window |

Other files: `metrics.json` (all numbers), `metrics_summary.csv`, `classification_report.txt`.
