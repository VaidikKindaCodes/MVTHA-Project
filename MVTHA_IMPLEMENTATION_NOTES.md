# MVTHA Prototype Implementation Notes

## Reused from CTR-GCN

- The model entry point follows the repository convention: `model.mvtha.Model` is an `nn.Module` whose constructor accepts the NTU model arguments and whose forward method accepts skeleton tensors.
- The expected input layout comes from the NTU feeder: `[N, C, T, V, M]` (batch, coordinates, frames, joints, persons).
- The existing `model.ctrgcn.Model` and its graph-convolution implementation were left unchanged. No CTR-GCN layers or pretrained weights are used by this prototype.
- PyTorch is the repository's model framework; the prototype uses its standard neural-network modules.

## Implemented for this prototype

- `ViewTransformation`: projects raw coordinates, root-relative coordinates, and temporal differences into a shared feature space.
- `MultiViewTransformer`: uses temporal multi-head self-attention on joint-aggregated frame features.
- `APAM`: learns an attention weight for each joint and frame.
- `HMSAM`: combines temporal channel attention at pooled scales of 1, 2, and 4.
- `MultiScaleTemporalConv`: combines temporal convolutions with dilation rates 1, 2, and 3.
- `Model`: applies the requested stages in order, globally averages person/joint/time dimensions, and produces class logits.

## Current status

The prototype completes a synthetic forward pass with output shape `[2, 60]` and one backward pass. This is a lightweight architecture prototype, not a complete reproduction of the MVTHA paper. The full NTU-60 dataset has not been downloaded, no full training was started, and no accuracy is claimed.

## Test command

From the repository root, run:

```text
python test_mvtha.py
```

The test prints each stage's output shape, checks for a `[2, 60]` classifier output, and performs one backward pass.
