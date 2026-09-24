import torch
from torch import nn


class GeometryMultiTaskEventClassifier(nn.Module):
    """Baseline MLP plus 14 reconstructed detector-geometry summaries.

    Inputs remain detector-observable only. Simulation truth such as PDG,
    ancestry, and stopping position is never passed to the network.
    """

    N_GEOMETRY_FEATURES = 14

    def __init__(self, n_channels: int, n_particle_classes: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.n_channels = n_channels
        self.encoder = nn.Sequential(
            nn.Linear(n_channels * 3 + self.N_GEOMETRY_FEATURES, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.particle_head = nn.Linear(hidden_dim, n_particle_classes)
        self.stop_head = nn.Linear(hidden_dim, 1)

    def forward(self, edep, time, hit, geometry):
        # Geometry normalization uses fixed detector-scale units rather than
        # statistics fitted to validation/test data.
        g = geometry.clone()
        g[:, 0] /= float(self.n_channels)  # hit channels
        g[:, 1] /= 3.0                    # hodoscopes hit
        g[:, 2:6] = torch.log1p(g[:, 2:6].clamp_min(0.0))  # energy summaries
        g[:, 6:10] /= 1500.0              # spatial spans [mm]
        g[:, 10] /= 100.0                  # time span [ns]
        g[:, 11:14] /= 1500.0              # centroids [mm]

        x = torch.cat([
            torch.log1p(edep.clamp_min(0.0)),
            time / 100.0,
            hit.float(),
            g,
        ], dim=1)
        z = self.encoder(x)
        return self.particle_head(z), self.stop_head(z).squeeze(1)
