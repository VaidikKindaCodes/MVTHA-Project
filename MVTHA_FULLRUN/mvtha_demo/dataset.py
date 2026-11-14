"""PyTorch dataset over the procedural multi-view skeleton actions."""
import numpy as np
import torch
from torch.utils.data import Dataset

from . import synth


class SyntheticMultiViewDataset(Dataset):
    """Item: (x [K, 3, T, 25, M] float32, label).

    The clean performances are generated once.  Camera yaws, sensor noise and occlusion are
    re-drawn on every access (training: random each epoch; fixed=True: deterministic per index,
    used for validation / the demo so that numbers are reproducible).

    yaw_mode: 'seen'   -> yaw ~ U(-60, 60) deg  (what the model is trained on)
              'unseen' -> |yaw| ~ U(70, 120) deg (side/back views never seen in training)
    """

    def __init__(self, per_class, seed=0, num_views=3, frames=32, yaw_mode="seen",
                 fixed=False, num_person=1):
        self.seed, self.num_views, self.frames = seed, num_views, frames
        self.yaw_mode, self.fixed, self.num_person = yaw_mode, fixed, num_person
        rng = np.random.default_rng(seed)
        self.labels = np.repeat(np.arange(synth.NUM_CLASSES), per_class)
        self.seqs = np.stack([synth.generate_sequence(int(l), frames, rng) for l in self.labels])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, 7919, i]) if self.fixed else np.random.default_rng()
        yaws = synth.sample_yaws(rng, self.num_views, self.yaw_mode)
        views = synth.make_views(self.seqs[i], yaws, rng)
        return torch.from_numpy(synth.to_model_input(views, self.num_person)), int(self.labels[i])
