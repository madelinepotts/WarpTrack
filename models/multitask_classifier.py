import torch
from torch import nn


class MultiTaskEventClassifier(nn.Module):
    """Baseline detector-only network with particle and stopping heads.

    Input channels are [log1p(Edep/MeV), relative_time_ns/100, hit_mask].
    Truth quantities such as PDG, track ancestry and stop position are never
    inputs to this network.
    """

    def __init__(self, n_channels: int, n_particle_classes: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.n_channels = n_channels
        self.encoder = nn.Sequential(
            nn.Linear(n_channels * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.particle_head = nn.Linear(hidden_dim, n_particle_classes)
        self.stop_head = nn.Linear(hidden_dim, 1)

    def forward(self, edep, time, hit):
        x = torch.cat(
            [torch.log1p(edep.clamp_min(0.0)), time / 100.0, hit.float()], dim=1
        )
        z = self.encoder(x)
        return self.particle_head(z), self.stop_head(z).squeeze(1)
