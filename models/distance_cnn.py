import torch
from torch import nn

import particle_cuda


class DistanceCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(64, 2)

    def forward(self, hits):
        d2 = particle_cuda.pairwise_distance(hits)

        # Compress large distances before the CNN.
        x = torch.log1p(d2).unsqueeze(1)

        x = self.features(x)
        x = x.flatten(1)

        return self.classifier(x)
