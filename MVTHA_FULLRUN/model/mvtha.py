"""
MVTHA -- Multi-View Transformer with Hierarchical Attention for skeleton action recognition.

Input
-----
  [N, K, C, T, V, M]   K camera views of the same performance      (multi-view)
  [N, C, T, V, M]      a single view (NTU feeder layout)           (treated as K = 1)
with N = batch, C = 3 coordinates, T = frames, V = 25 joints, M = persons.
A joint whose (x, y, z) are all exactly 0 is treated as *missing* (occluded / absent person).

Pipeline
--------
  ViewTransformation  per view: centre, rotate to a canonical (shoulder-aligned) viewpoint,
                      build 4 streams (position, root-relative, velocity, bone) -> D channels
  SpatialMix          learnable skeleton-graph message passing between joints
  MultiViewTransformer  temporal self-attention inside every view, then cross-view attention
                      at every frame
  ViewFusion          visibility-aware attention over the K views  -> one feature map
  [ APAM -> HMSAM -> MultiScaleTemporalConv ] x num_blocks
  temporal pooling -> person-masked mean -> dropout -> linear classifier  -> [N, num_class] logits

`Model` keeps the constructor convention of the CTR-GCN code base, so
`python main.py --config ... --model model.mvtha.Model` still works.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Bones of the NTU RGB+D skeleton (0-based joints) and each joint's parent (root 0 -> itself).
NTU_EDGES = [(0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
             (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
             (0, 12), (12, 13), (13, 14), (14, 15), (0, 16), (16, 17), (17, 18), (18, 19)]
NTU_PARENTS = [0, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11]
# (left, right) joint pairs used to estimate body orientation: shoulders and hips
ORIENTATION_PAIRS = ((4, 8), (12, 16))


def _normalized_adjacency(num_point):
    A = np.eye(num_point, dtype=np.float32)
    if num_point == 25:
        for i, j in NTU_EDGES:
            A[i, j] = A[j, i] = 1.0
    d = A.sum(1) ** -0.5
    return (d[:, None] * A * d[None, :]).astype(np.float32)


# ----------------------------------------------------------------------------- stage 1
class ViewTransformation(nn.Module):
    """VTM.  Works on every view independently: [B, C, T, V] -> ([B, D, T, V], visibility [B, T, V]).

    * centre the person, zero the missing joints
    * (align=True) rotate about the vertical axis so the shoulders/hips line up with +x.  This
      removes the camera yaw, which is what makes the model robust to unseen viewpoints.
    * 4 streams (position, position relative to the per-frame centroid, velocity, bone vectors)
      are embedded jointly into D channels.
    """

    def __init__(self, in_channels=3, out_channels=64, num_point=25, align=True):
        super().__init__()
        self.align = bool(align) and in_channels == 3 and num_point == 25
        parents = NTU_PARENTS if num_point == 25 else list(range(num_point))
        self.register_buffer("parents", torch.tensor(parents, dtype=torch.long), persistent=False)
        self.embed = nn.Sequential(
            nn.Conv2d(4 * in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.joint_emb = nn.Parameter(torch.zeros(1, out_channels, 1, num_point))
        nn.init.normal_(self.joint_emb, std=0.02)

    @staticmethod
    def _yaw_align(x, vis4, vis):
        with torch.no_grad():
            B = x.shape[0]
            acc = x.new_zeros(B, 3)
            for l, r in ORIENTATION_PAIRS:
                valid = vis[:, :, l] * vis[:, :, r]                              # [B, T]
                diff = (x[:, :, :, l] - x[:, :, :, r]) * valid.unsqueeze(1)      # [B, 3, T]
                acc = acc + diff.sum(dim=2)
            phi = torch.atan2(acc[:, 2], acc[:, 0])                              # [B]
            c, s = torch.cos(phi).view(B, 1, 1), torch.sin(phi).view(B, 1, 1)
        X, Y, Z = x[:, 0], x[:, 1], x[:, 2]                                      # [B, T, V]
        return torch.stack([c * X + s * Z, Y, -s * X + c * Z], dim=1) * vis4

    def forward(self, x):
        vis = (x.abs().sum(dim=1) > 0).to(x.dtype)                               # [B, T, V]
        vis4 = vis.unsqueeze(1)                                                  # [B, 1, T, V]
        cnt = vis.sum(dim=(1, 2)).clamp(min=1.0)                                 # [B]
        center = (x * vis4).sum(dim=(2, 3)) / cnt.unsqueeze(1)                   # [B, C]
        x = (x - center[:, :, None, None]) * vis4
        if self.align:
            x = self._yaw_align(x, vis4, vis)

        frame_c = (x * vis4).sum(dim=3) / vis.sum(dim=2).clamp(min=1.0).unsqueeze(1)   # [B, C, T]
        rel = (x - frame_c.unsqueeze(3)) * vis4
        vel = (x[:, :, 1:] - x[:, :, :-1]) * (vis[:, 1:] * vis[:, :-1]).unsqueeze(1)
        vel = F.pad(vel, (0, 0, 1, 0)) * 10.0                                    # [B, C, T, V]
        bone = (x - x[:, :, :, self.parents]) * (vis * vis[:, :, self.parents]).unsqueeze(1)

        feat = self.embed(torch.cat([x, rel, vel, bone], dim=1)) + self.joint_emb
        return feat, vis


class SpatialMix(nn.Module):
    """Message passing over the skeleton graph (adjacency initialised from the bones, then learned)."""

    def __init__(self, channels, num_point=25):
        super().__init__()
        self.register_buffer("A0", torch.from_numpy(_normalized_adjacency(num_point)), persistent=False)
        self.A_delta = nn.Parameter(torch.zeros(num_point, num_point))
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):                                                        # [B, D, T, V]
        y = torch.einsum("bdtv,vw->bdtw", x, self.A0 + self.A_delta)
        return F.relu(x + self.bn(self.conv(y)))


# ----------------------------------------------------------------------------- stage 2
class MultiViewTransformer(nn.Module):
    """MVT.  Frame tokens (joint-averaged features):
         1. temporal self-attention inside each view
         2. cross-view self-attention across the K views at every frame
       The result is broadcast back onto every joint (residual).  Shape preserving."""

    def __init__(self, channels=64, num_heads=4, max_frames=300, max_views=8, dropout=0.1):
        super().__init__()
        self.max_views = max_views
        self.pos = nn.Parameter(torch.zeros(1, max_frames, channels))
        self.view_emb = nn.Parameter(torch.zeros(max_views, channels))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.view_emb, std=0.02)
        self.temporal = nn.TransformerEncoderLayer(
            d_model=channels, nhead=num_heads, dim_feedforward=2 * channels,
            dropout=dropout, batch_first=True, norm_first=True)
        self.norm_v = nn.LayerNorm(channels)
        self.view_attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, n, k, m):                                               # x [n*k*m, D, T, V]
        B, D, T, V = x.shape
        tok = x.mean(dim=3).transpose(1, 2)                                      # [B, T, D]
        tok = self.temporal(tok + self.pos[:, :T])
        t = tok.reshape(n, k, m, T, D).permute(0, 2, 3, 1, 4).reshape(n * m * T, k, D)
        q = self.norm_v(t + self.view_emb[:k])
        t = t + self.view_attn(q, q, q, need_weights=False)[0]
        tok = t.reshape(n, m, T, k, D).permute(0, 3, 1, 2, 4).reshape(B, T, D)
        return x + tok.transpose(1, 2).unsqueeze(-1)                             # [B, D, T, 1] -> all joints


class ViewFusion(nn.Module):
    """Visibility-aware attention over views, per frame: [n*k*m, D, T, V] -> [n*m, D, T, V]."""

    def __init__(self, channels=64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(channels, channels // 4), nn.Tanh(),
                                   nn.Linear(channels // 4, 1))
        self.vis_gain = nn.Parameter(torch.tensor(4.0))   # views that see more joints start with more weight
        self.last_weights = None                           # [n, m, T, k], for visualisation

    def forward(self, x, vis, n, k, m):
        B, D, T, V = x.shape
        x6 = x.reshape(n, k, m, D, T, V)
        pooled = x6.mean(dim=5).permute(0, 2, 4, 1, 3)                           # [n, m, T, k, D]
        frac = vis.reshape(n, k, m, T, V).mean(dim=4).permute(0, 2, 3, 1).unsqueeze(-1)   # [n, m, T, k, 1]
        w = torch.softmax(self.score(pooled) + self.vis_gain * frac, dim=3)      # [n, m, T, k, 1]
        self.last_weights = w.squeeze(-1).detach()
        w6 = w.permute(0, 3, 1, 4, 2).unsqueeze(-1)                              # [n, k, m, 1, T, 1]
        return (x6 * w6).sum(dim=1).reshape(n * m, D, T, V)


# ----------------------------------------------------------------------------- stage 3
class APAM(nn.Module):
    """Joint attention: per frame, a softmax over the joints re-weights the joint features."""

    def __init__(self, channels=64, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Conv2d(channels, channels // reduction, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(channels // reduction, 1, 1))

    def forward(self, x):                                                        # [B, D, T, V]
        a = torch.softmax(self.fc(x), dim=3) * x.shape[3]                        # mean weight == 1
        return x + x * a


class HMSAM(nn.Module):
    """Hierarchical multi-scale channel attention: temporal pooling into 1, 2 and 4 bins."""

    def __init__(self, channels=64, scales=(1, 2, 4), reduction=4):
        super().__init__()
        self.scales = tuple(scales)
        self.fcs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels, channels // reduction, 1), nn.ReLU(inplace=True),
                          nn.Conv1d(channels // reduction, channels, 1)) for _ in self.scales])

    def forward(self, x):                                                        # [B, D, T, V]
        T = x.shape[2]
        p = x.mean(dim=3)                                                        # [B, D, T]
        att = 0
        for s, fc in zip(self.scales, self.fcs):
            a = fc(F.adaptive_avg_pool1d(p, s))                                  # [B, D, s]
            att = att + (a.expand(-1, -1, T) if s == 1 else
                         F.interpolate(a, size=T, mode="linear", align_corners=False))
        att = torch.sigmoid(att / len(self.scales))                              # [B, D, T]
        return x + x * att.unsqueeze(3)


class MultiScaleTemporalConv(nn.Module):
    """Parallel dilated temporal convolutions (dilation 1, 2, 3), averaged, with a residual."""

    def __init__(self, channels=64, dilations=(1, 2, 3), kernel_size=3):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, (kernel_size, 1),
                          padding=(d * (kernel_size - 1) // 2, 0), dilation=(d, 1)),
                nn.BatchNorm2d(channels)) for d in dilations])

    def forward(self, x):
        out = sum(b(x) for b in self.branches) / len(self.branches)
        return F.relu(out + x)


# ----------------------------------------------------------------------------- full model
class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, in_channels=3,
                 graph=None, graph_args=None,            # accepted (and ignored) for CTR-GCN configs
                 channels=64, num_heads=4, num_blocks=1, dropout=0.1,
                 align_views=True, head_pool="joint", max_frames=300, max_views=8, **kwargs):
        super().__init__()
        assert head_pool in ("joint", "global")
        self.num_point, self.head_pool = num_point, head_pool
        self.vtm = ViewTransformation(in_channels, channels, num_point, align=align_views)
        self.spatial = SpatialMix(channels, num_point)
        self.mvt = MultiViewTransformer(channels, num_heads, max_frames, max_views, dropout)
        self.fusion = ViewFusion(channels)
        self.apam = nn.ModuleList([APAM(channels) for _ in range(num_blocks)])
        self.hmsam = nn.ModuleList([HMSAM(channels) for _ in range(num_blocks)])
        self.mstc = nn.ModuleList([MultiScaleTemporalConv(channels) for _ in range(num_blocks)])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * (num_point if head_pool == "joint" else 1), num_class)

    @property
    def view_weights(self):
        """Fusion weights of the last forward pass: [N, M, T, K]."""
        return self.fusion.last_weights

    def forward(self, x, return_stages=False):
        if x.dim() == 5:                                       # single view -> K = 1
            x = x.unsqueeze(1)
        n, k, c, t, v, m = x.shape
        assert k <= self.mvt.max_views, "too many views for max_views"
        x = x.permute(0, 1, 5, 2, 3, 4).reshape(n * k * m, c, t, v)             # batch order (n, k, m)

        stages = {}
        feat, vis = self.vtm(x);              stages["VTM"] = feat.shape
        feat = self.spatial(feat);            stages["SpatialMix"] = feat.shape
        feat = self.mvt(feat, n, k, m);       stages["MVT"] = feat.shape
        feat = self.fusion(feat, vis, n, k, m); stages["ViewFusion"] = feat.shape
        for apam, hmsam, mstc in zip(self.apam, self.hmsam, self.mstc):
            feat = apam(feat);                stages["APAM"] = feat.shape
            feat = hmsam(feat);               stages["HMSAM"] = feat.shape
            feat = mstc(feat);                stages["MS-TC"] = feat.shape

        pres = (vis.reshape(n, k, m, -1).sum(dim=(1, 3)) > 0).to(feat.dtype)     # [n, m] person present?
        f = feat.mean(dim=2)                                                     # [n*m, D, V]
        if self.head_pool == "global":
            f = f.mean(dim=2, keepdim=True)
        f = f.reshape(n, m, -1)
        f = (f * pres.unsqueeze(-1)).sum(dim=1) / pres.sum(dim=1, keepdim=True).clamp(min=1.0)
        logits = self.fc(self.drop(f))
        stages["logits"] = logits.shape
        return (logits, stages) if return_stages else logits
