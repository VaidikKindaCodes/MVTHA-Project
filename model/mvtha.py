import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewTransformation(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64):
        super(ViewTransformation, self).__init__()
        self.coordinate_projection = nn.Linear(in_channels, embed_dim)
        self.root_projection = nn.Linear(in_channels, embed_dim)
        self.motion_projection = nn.Linear(in_channels, embed_dim)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError('Expected input with shape [N, C, T, V, M]')

        coordinates = x.permute(0, 4, 2, 3, 1).contiguous()
        root_relative = coordinates - coordinates[:, :, :, :1, :]
        motion = torch.cat(
            (torch.zeros_like(coordinates[:, :, :1]), coordinates[:, :, 1:] - coordinates[:, :, :-1]),
            dim=2)

        features = (
            self.coordinate_projection(coordinates)
            + self.root_projection(root_relative)
            + self.motion_projection(motion)
        ) / 3.0
        batch_size, persons, frames, joints, channels = features.shape
        return features.permute(0, 1, 4, 2, 3).contiguous().view(
            batch_size * persons, channels, frames, joints)


class MultiViewTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, dropout=0.1):
        super(MultiViewTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size, channels, frames, _ = x.shape
        sequence = x.mean(dim=3).permute(2, 0, 1)
        attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        attended = self.dropout(attended)
        sequence = self.norm(sequence + attended)
        temporal_features = sequence.permute(1, 2, 0).contiguous().view(
            batch_size, channels, frames, 1)
        return x + temporal_features


class APAM(nn.Module):
    def __init__(self, channels=64):
        super(APAM, self).__init__()
        self.attention = nn.Linear(channels, 1)

    def forward(self, x):
        features = x.permute(0, 2, 3, 1)
        attention = torch.softmax(self.attention(features), dim=2)
        joint_count = features.size(2)
        return (features * (1.0 + joint_count * attention)).permute(0, 3, 1, 2).contiguous()


class HMSAM(nn.Module):
    def __init__(self, channels=64, scales=(1, 2, 4)):
        super(HMSAM, self).__init__()
        self.scales = scales
        hidden_channels = max(1, channels // 4)
        self.channel_attention = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        batch_size, channels, frames, _ = x.shape
        temporal_features = x.mean(dim=3)
        scale_attention = []
        for scale in self.scales:
            pooled = F.adaptive_avg_pool1d(temporal_features, scale)
            attention = self.channel_attention(pooled)
            scale_attention.append(F.interpolate(
                attention, size=frames, mode='linear', align_corners=False))

        attention = torch.stack(scale_attention, dim=0).mean(dim=0)
        return x * (1.0 + attention.unsqueeze(-1))


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, channels=64, dilations=(1, 2, 3)):
        super(MultiScaleTemporalConv, self).__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation),
                nn.BatchNorm1d(channels))
            for dilation in dilations
        ])

    def forward(self, x):
        batch_size, channels, frames, joints = x.shape
        features = x.permute(0, 3, 1, 2).contiguous().view(
            batch_size * joints, channels, frames)
        temporal_features = torch.stack(
            [branch(features) for branch in self.branches], dim=0).mean(dim=0)
        temporal_features = F.relu(features + temporal_features)
        return temporal_features.view(
            batch_size, joints, channels, frames).permute(0, 2, 3, 1).contiguous()


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None,
                 graph_args=None, in_channels=3, embed_dim=64, num_heads=4,
                 drop_out=0.1):
        super(Model, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')

        self.num_point = num_point
        self.num_person = num_person
        self.view_transformation = ViewTransformation(in_channels, embed_dim)
        self.multi_view_transformer = MultiViewTransformer(
            embed_dim, num_heads, drop_out)
        self.apam = APAM(embed_dim)
        self.hmsam = HMSAM(embed_dim)
        self.multi_scale_temporal_conv = MultiScaleTemporalConv(embed_dim)
        self.drop_out = nn.Dropout(drop_out) if drop_out else nn.Sequential()
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, x):
        batch_size, _, frames, joints, persons = x.shape
        x = self.view_transformation(x)
        x = self.multi_view_transformer(x)
        x = self.apam(x)
        x = self.hmsam(x)
        x = self.multi_scale_temporal_conv(x)

        channels = x.size(1)
        x = x.view(batch_size, persons, channels, frames, joints)
        x = x.mean(dim=4).mean(dim=3).mean(dim=1)
        return self.fc(self.drop_out(x))
