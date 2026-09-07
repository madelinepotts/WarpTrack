import random

import torch
from torch.utils.data import Dataset


def _random_unit_vector(generator):
    vector = torch.randn(3, generator=generator)
    return vector / torch.linalg.norm(vector)


def _orthonormal_basis(direction):
    reference = torch.tensor([1.0, 0.0, 0.0])

    if torch.abs(torch.dot(direction, reference)) > 0.9:
        reference = torch.tensor([0.0, 1.0, 0.0])

    u = torch.linalg.cross(direction, reference)
    u = u / torch.linalg.norm(u)

    v = torch.linalg.cross(direction, u)
    v = v / torch.linalg.norm(v)

    return u, v


def generate_track(num_hits, generator):
    direction = _random_unit_vector(generator)
    u, v = _orthonormal_basis(direction)

    center = torch.randn(3, generator=generator) * 0.5

    t = torch.linspace(-4.0, 4.0, num_hits)
    t += 0.08 * torch.randn(num_hits, generator=generator)

    transverse_u = 0.08 * torch.randn(num_hits, generator=generator)
    transverse_v = 0.08 * torch.randn(num_hits, generator=generator)

    hits = (
        center[None, :]
        + t[:, None] * direction[None, :]
        + transverse_u[:, None] * u[None, :]
        + transverse_v[:, None] * v[None, :]
    )

    # Do not let the network use hit ordering as an easy shortcut.
    permutation = torch.randperm(num_hits, generator=generator)

    return hits[permutation].float()


def generate_shower(num_hits, generator):
    direction = _random_unit_vector(generator)
    u, v = _orthonormal_basis(direction)

    vertex = torch.randn(3, generator=generator) * 0.5

    # Approximate a developing shower with a forward depth distribution.
    depth = torch.abs(torch.randn(num_hits, generator=generator)) * 2.0
    depth = torch.clamp(depth, max=7.0)

    # Transverse width grows as the shower develops.
    sigma = 0.10 + 0.20 * depth

    transverse_u = sigma * torch.randn(num_hits, generator=generator)
    transverse_v = sigma * torch.randn(num_hits, generator=generator)
    longitudinal_noise = 0.08 * torch.randn(num_hits, generator=generator)

    hits = (
        vertex[None, :]
        + (depth + longitudinal_noise)[:, None] * direction[None, :]
        + transverse_u[:, None] * u[None, :]
        + transverse_v[:, None] * v[None, :]
    )

    permutation = torch.randperm(num_hits, generator=generator)

    return hits[permutation].float()


class SyntheticParticleDataset(Dataset):
    def __init__(
        self,
        num_events=10000,
        num_hits=64,
        seed=12345,
    ):
        self.num_events = num_events
        self.num_hits = num_hits
        self.seed = seed

    def __len__(self):
        return self.num_events

    def __getitem__(self, index):
        rng = random.Random(self.seed + index)
        label = rng.randint(0, 1)

        generator = torch.Generator()
        generator.manual_seed(self.seed * 100000 + index)

        if label == 0:
            hits = generate_track(self.num_hits, generator)
        else:
            hits = generate_shower(self.num_hits, generator)

        return hits, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    generator = torch.Generator().manual_seed(7)

    print("Track-like event:")
    print(generate_track(8, generator))

    print("\nShower-like event:")
    print(generate_shower(8, generator))
