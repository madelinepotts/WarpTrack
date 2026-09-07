import torch
from torch import nn

import warptrack_cuda


class EventClassifier(nn.Module):
    """
    Starter event classifier.

    This is intentionally not the final architecture. It combines learned
    per-hit features with simple geometry summaries from the custom CUDA
    pairwise-distance kernel.
    """

    def __init__(self, hidden_dim=64):
        super().__init__()

        self.hit_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # pooled hit features + four geometry summary values
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, hits, mask):
        xyz = hits[..., :3]

        # Custom CUDA operation.
        d2 = warptrack_cuda.pairwise_distance(xyz)

        valid_pair = mask[:, :, None] & mask[:, None, :]

        # Exclude diagonal self-distances.
        eye = torch.eye(
            mask.size(1),
            device=mask.device,
            dtype=torch.bool,
        ).unsqueeze(0)

        valid_pair = valid_pair & ~eye

        pair_values = d2.masked_fill(~valid_pair, 0.0)

        pair_count = valid_pair.sum(dim=(1, 2)).clamp_min(1)

        mean_d2 = pair_values.sum(dim=(1, 2)) / pair_count

        max_d2 = d2.masked_fill(
            ~valid_pair,
            float("-inf"),
        ).amax(dim=(1, 2))

        max_d2 = torch.where(
            torch.isfinite(max_d2),
            max_d2,
            torch.zeros_like(max_d2),
        )

        hit_count = mask.sum(dim=1).float()

        energy_sum = (
            hits[..., 3] * mask.float()
        ).sum(dim=1)

        geometry = torch.stack(
            [
                torch.log1p(mean_d2),
                torch.log1p(max_d2),
                torch.log1p(hit_count),
                torch.log1p(energy_sum),
            ],
            dim=1,
        )

        encoded = self.hit_encoder(hits)

        encoded = encoded * mask.unsqueeze(-1)

        denominator = (
            mask.sum(dim=1, keepdim=True)
            .clamp_min(1)
            .float()
        )

        pooled = encoded.sum(dim=1) / denominator

        features = torch.cat(
            [pooled, geometry],
            dim=1,
        )

        return self.classifier(features)
