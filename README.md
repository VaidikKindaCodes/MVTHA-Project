# MVTHA Action Recognition Project

This repository contains the upstream CTR-GCN skeleton-action-recognition code and a lightweight prototype for our college project, **Multi-view Transformer with Hierarchical Attention for Action Recognition (MVTHA)**. The intended dataset is NTU RGB+D 60.

The MVTHA work is an initial architecture prototype only. Its synthetic-input test verifies that a batch can pass through the model and produce 60 class logits. The full NTU-60 dataset has not been downloaded for this prototype, no full training has been run, and no recognition accuracy or reproduction of the MVTHA paper is claimed.

## Quick start: run the MVTHA prototype

From the repository root, run:

```text
python test_mvtha.py
```

The test creates a random input tensor with shape `[2, 3, 64, 25, 2]`, prints each major stage's output shape, checks that the classifier returns `[2, 60]`, and runs one backward pass. It does not load data, download files, or start training.

Example output:

```text
Input shape: torch.Size([2, 3, 64, 25, 2])
VTM output shape: torch.Size([4, 64, 64, 25])
MVT output shape: torch.Size([4, 64, 64, 25])
APAM output shape: torch.Size([4, 64, 64, 25])
HMSAM output shape: torch.Size([4, 64, 64, 25])
MS-TC output shape: torch.Size([4, 64, 64, 25])
Final output shape: torch.Size([2, 60])
Backward pass: successful
```

The prototype was tested with Python 3.14.7 and PyTorch 2.14.0+cpu. The original repository's `requirements.txt` pins a much older PyTorch version; compatibility with that pinned version has not been tested. No dependency installation is required if PyTorch is already installed.

## What the project does

Skeleton-based action recognition classifies an action from the motion of body joints over time. For NTU RGB+D 60, each sample can represent 25 joints, 3 coordinate values per joint, a sequence of frames, and up to 2 people.

The MVTHA prototype accepts a tensor in this layout:

| Dimension | Meaning | Test value |
| --- | --- | --- |
| `N` | batch size | 2 |
| `C` | coordinate channels (x, y, z) | 3 |
| `T` | number of frames | 64 |
| `V` | number of joints | 25 |
| `M` | number of people | 2 |

Thus, the test input is `[N, C, T, V, M] = [2, 3, 64, 25, 2]`. This agrees with the layout returned by the CTR-GCN NTU feeder.

## MVTHA prototype architecture

`model/mvtha.py` implements the following pipeline. The implementation is intentionally simplified to demonstrate a complete differentiable model, not to reproduce every equation or detail of the MVTHA paper.

```text
Skeleton input [N, C, T, V, M]
    -> ViewTransformation (VTM)
    -> MultiViewTransformer (MVT)
    -> APAM
    -> HMSAM
    -> MultiScaleTemporalConv (MS-TC)
    -> global average pooling
    -> dropout and linear classifier
    -> class scores [N, 60]
```

For the test input, VTM merges the batch and person dimensions (`N * M = 4`) and projects the features to 64 channels, giving `[4, 64, 64, 25]`. The remaining feature-processing stages keep that shape. Pooling combines the two people, all frames, and all joints, and the linear layer maps each sample to 60 class scores.

### Implemented modules

- **`ViewTransformation`** projects three simple coordinate views into a shared 64-dimensional feature space: the raw joint coordinates, coordinates relative to joint 0 in each frame, and frame-to-frame coordinate differences. The first frame's motion feature is zero.
- **`MultiViewTransformer`** averages joint features to form one feature vector per frame, applies multi-head self-attention over the frame sequence, then adds the attended representation back to every joint.
- **`APAM`** computes an attention distribution over joints independently for each frame and uses it to reweight joint features.
- **`HMSAM`** computes channel attention from temporally pooled features at three temporal scales (1, 2, and 4 bins), interpolates the results to the input sequence length, and combines them.
- **`MultiScaleTemporalConv`** applies parallel temporal convolutions with dilation rates 1, 2, and 3 to each joint, averages their outputs, and adds a residual connection.
- **`Model`** connects the stages, averages over joints, frames, and people, and applies dropout and a linear classifier. Its default `num_class` is 60.

In the test hooks, the intermediate shape `[4, 64, 64, 25]` means `[N * M, feature channels, frames, joints]`. The final output `[2, 60]` contains 60 unnormalized class scores (logits) per input sample; it is not an accuracy or a probability unless a separate softmax is applied.

## Repository guide

| Path | Purpose |
| --- | --- |
| `model/mvtha.py` | New, lightweight MVTHA prototype. |
| `test_mvtha.py` | Synthetic forward-shape and backward-pass smoke test. |
| `MVTHA_IMPLEMENTATION_NOTES.md` | Notes on what the prototype reuses, its status, and its test command. |
| `model/ctrgcn.py` | Existing CTR-GCN graph-convolution model; left unchanged. |
| `model/baseline.py` | Existing baseline model supplied with CTR-GCN. |
| `main.py` | Existing configurable training/testing entry point. It dynamically imports a model class and builds it from the model arguments in a YAML config. |
| `feeders/` | Dataset loading, sample preparation, and augmentation. `feeders/feeder_ntu.py` loads NTU data and formats it as `[N, C, T, V, M]`. |
| `graph/` | Skeleton graph definitions and graph utilities used by graph-based models such as CTR-GCN. |
| `config/` | YAML configurations for NTU RGB+D 60/120 and NW-UCLA experiments. |
| `data/` | Dataset-processing scripts and metadata/statistics. A prepared NTU-60 training archive is not needed or used by the MVTHA smoke test. |
| `torchlight/` | Repository utility package used by the original training pipeline. |
| `ensemble.py` | Existing score-ensemble utility for combining model/modality outputs. |
| `requirements.txt` | Pinned dependency list for the original repository environment; it includes legacy package versions. |
| `src/framework.jpg` | Architecture illustration referenced by the upstream CTR-GCN documentation below. |
| `LICENSE` | License file included with the repository. |

## How MVTHA relates to CTR-GCN

CTR-GCN is the original model and training framework in this repository. Its model is in `model/ctrgcn.py`; the MVTHA prototype is a separate module in `model/mvtha.py`. The original model was not deleted, overwritten, or used as an MVTHA layer. The prototype follows the same general dynamic model-import convention and accepts common model arguments such as `num_class`, `num_point`, `num_person`, and `in_channels`. It does not use CTR-GCN's graph, graph arguments, graph convolutions, or pretrained weights.

The CTR-GCN NTU feeder loads a prepared `.npz` dataset archive and formats samples as `[N, C, T, V, M]`. The repository's NTU YAML configuration points to `data/ntu/NTU60_CS.npz`; that prepared archive is not required for the dummy-input prototype test. Although the MVTHA model constructor follows the project's model interface, the existing YAML configuration selects CTR-GCN by default, and this milestone does not run `main.py` or train either model.

## Data and training status

The original CTR-GCN repository includes scripts and instructions for preparing NTU RGB+D 60/120 and NW-UCLA data and for training/testing its models. Those upstream instructions are retained below. They describe the original repository workflow; they are not steps needed to run the MVTHA prototype.

For the current MVTHA milestone:

- No full NTU-60 dataset download was performed.
- No full or partial training run was started.
- No trained MVTHA weights or evaluation results are provided.
- No accuracy is claimed.
- Only the synthetic-input forward pass, output-shape assertion, and one backward pass have been checked.

See [MVTHA_IMPLEMENTATION_NOTES.md](./MVTHA_IMPLEMENTATION_NOTES.md) for a concise implementation-status record. To develop this prototype further, training should be treated as a separate step: prepare the dataset, configure the data loader and model, then define and run an appropriate experiment. Do not interpret the smoke test as a trained action recognizer.

---

# CTR-GCN
This repo is the official implementation for [Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition](https://arxiv.org/abs/2107.12213). The paper is accepted to ICCV2021.

Note: We also provide a simple and strong baseline model, which achieves 83.7% on NTU120 CSub with joint modality only, to facilitate the development of skeleton-based action recognition.

## Architecture of CTR-GC
![image](src/framework.jpg)
# Prerequisites

- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX

- We provide the dependency file of our experimental environment, you can install all dependencies by creating a new anaconda virtual environment and running `pip install -r requirements.txt `
- Run `pip install -e torchlight` 

# Data Preparation

### Download datasets.

#### There are 3 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- NW-UCLA

#### NTU RGB+D 60 and 120

1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`

#### NW-UCLA

1. Download dataset from [here](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0)
2. Move `all_sqe` to `./data/NW-UCLA`

### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - NW-UCLA/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```



# Training & Testing

### Training

- Change the config file depending on what you want.

```
# Example: training CTRGCN on NTU RGB+D 120 cross subject with GPU 0
python main.py --config config/nturgbd120-cross-subject/default.yaml --work-dir work_dir/ntu120/csub/ctrgcn --device 0
# Example: training provided baseline on NTU RGB+D 120 cross subject
python main.py --config config/nturgbd120-cross-subject/default.yaml --model model.baseline.Model--work-dir work_dir/ntu120/csub/baseline --device 0
```

- To train model on NTU RGB+D 60/120 with bone or motion modalities, setting `bone` or `vel` arguments in the config file `default.yaml` or in the command line.

```
# Example: training CTRGCN on NTU RGB+D 120 cross subject under bone modality
python main.py --config config/nturgbd120-cross-subject/default.yaml --train_feeder_args bone=True --test_feeder_args bone=True --work-dir work_dir/ntu120/csub/ctrgcn_bone --device 0
```

- To train model on NW-UCLA with bone or motion modalities, you need to modify `data_path` in `train_feeder_args` and `test_feeder_args` to "bone" or "motion" or "bone motion", and run

```
python main.py --config config/ucla/default.yaml --work-dir work_dir/ucla/ctrgcn_xxx --device 0
```

- To train your own model, put model file `your_model.py` under `./model` and run:

```
# Example: training your own model on NTU RGB+D 120 cross subject
python main.py --config config/nturgbd120-cross-subject/default.yaml --model model.your_model.Model --work-dir work_dir/ntu120/csub/your_model --device 0
```

### Testing

- To test the trained models saved in <work_dir>, run the following command:

```
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --device 0
```

- To ensemble the results of different modalities, run 
```
# Example: ensemble four modalities of CTRGCN on NTU RGB+D 120 cross subject
python ensemble.py --datasets ntu120/xsub --joint-dir work_dir/ntu120/csub/ctrgcn --bone-dir work_dir/ntu120/csub/ctrgcn_bone --joint-motion-dir work_dir/ntu120/csub/ctrgcn_motion --bone-motion-dir work_dir/ntu120/csub/ctrgcn_bone_motion
```

### Pretrained Models

- Download pretrained models for producing the final results on NTU RGB+D 60&120 cross subject [[Google Drive]](https://drive.google.com/drive/folders/1C9XUAgnwrGelvl4mGGVZQW6akiapgdnd?usp=sharing).
- Put files to <work_dir> and run **Testing** command to produce the final result.
