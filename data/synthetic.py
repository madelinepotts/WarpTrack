import math
import random

import torch
from torch.utils.data import Dataset


MUON = 0
HADRONIC_SHOWER = 1


def _random_unit_vector(generator):
    vector = torch.randn(3, generator=generator)
    return vector / torch.linalg.norm(vector)


def _basis(direction):
    reference = torch.tensor([1.0, 0.0, 0.0])

    if torch.abs(torch.dot(direction, reference)) > 0.9:
        reference = torch.tensor([0.0, 1.0, 0.0])

    u = torch.linalg.cross(direction, reference)
    u = u / torch.linalg.norm(u)

    v = torch.linalg.cross(direction, u)
    v = v / torch.linalg.norm(v)

    return u, v


def generate_muon_event(num_hits, generator):
    """
    Simplified muon-like event.

    Produces a coherent trajectory with small transverse fluctuations and
    relatively narrow energy-deposition variation.
    """
    direction = _random_unit_vector(generator)
    u, v = _basis(direction)

    center = torch.randn(3, generator=generator) * 0.25

    distance = torch.linspace(-3.5, 3.5, num_hits)
    distance += 0.05 * torch.randn(num_hits, generator=generator)

    transverse_u = 0.05 * torch.randn(num_hits, generator=generator)
    transverse_v = 0.05 * torch.randn(num_hits, generator=generator)

    xyz = (
        center[None, :]
        + distance[:, None] * direction[None, :]
        + transverse_u[:, None] * u[None, :]
        + transverse_v[:, None] * v[None, :]
    )

    # Synthetic MIP-like energy deposits.
    energy = 1.0 + 0.15 * torch.randn(num_hits, generator=generator)
    energy = torch.clamp(energy, min=0.20)

    hits = torch.cat([xyz, energy[:, None]], dim=1)

    # Remove ordering information.
    permutation = torch.randperm(num_hits, generator=generator)

    return hits[permutation].float()


def generate_hadronic_event(num_hits, generator):
    """
    Simplified hadronic-shower-like event.

    Produces a main shower axis plus secondary branches and a broad
    energy-deposition distribution.
    """
    direction = _random_unit_vector(generator)
    u, v = _basis(direction)

    origin = torch.randn(3, generator=generator) * 0.25

    depth = torch.rand(num_hits, generator=generator) * 6.0

    transverse_scale = 0.08 + 0.20 * depth

    xyz = (
        origin[None, :]
        + depth[:, None] * direction[None, :]
        + (
            transverse_scale
            * torch.randn(num_hits, generator=generator)
        )[:, None] * u[None, :]
        + (
            transverse_scale
            * torch.randn(num_hits, generator=generator)
        )[:, None] * v[None, :]
    )

    # Add one or two crude secondary branches.
    branch_count = 1 + int(
        torch.randint(0, 2, (1,), generator=generator).item()
    )

    for _ in range(branch_count):
        branch_size = max(2, num_hits // 6)

        indices = torch.randperm(
            num_hits,
            generator=generator
        )[:branch_size]

        branch_direction = (
            direction
            + 0.55 * _random_unit_vector(generator)
        )
        branch_direction = branch_direction / torch.linalg.norm(
            branch_direction
        )

        branch_step = (
            torch.rand(branch_size, generator=generator) * 1.5
        )

        xyz[indices] += (
            branch_step[:, None] * branch_direction[None, :]
        )

    # Broader, asymmetric energy deposition.
    energy = torch.exp(
        0.55 * torch.randn(num_hits, generator=generator)
    )
    energy = torch.clamp(energy, min=0.10, max=8.0)

    hits = torch.cat([xyz, energy[:, None]], dim=1)

    permutation = torch.randperm(num_hits, generator=generator)

    return hits[permutation].float()


class SyntheticCosmicDataset(Dataset):
    """
    Returns padded variable-length events.

    hits:
        [max_hits, 4] with columns x, y, z, energy

    mask:
        [max_hits] boolean mask

    label:
        0 = muon-like
        1 = hadronic shower-like
    """

    def __init__(
        self,
        num_events=12000,
        min_hits=6,
        max_hits=64,
        seed=12345,
    ):
        self.num_events = num_events
        self.min_hits = min_hits
        self.max_hits = max_hits
        self.seed = seed

    def __len__(self):
        return self.num_events

    def __getitem__(self, index):
        rng = random.Random(self.seed + index)

        label = rng.randint(0, 1)

        generator = torch.Generator()
        generator.manual_seed(
            self.seed * 100000 + index
        )

        if label == MUON:
            upper = max(self.min_hits, self.max_hits // 2)
            num_hits = rng.randint(self.min_hits, upper)
            event = generate_muon_event(num_hits, generator)
        else:
            lower = max(self.min_hits, self.max_hits // 4)
            num_hits = rng.randint(lower, self.max_hits)
            event = generate_hadronic_event(num_hits, generator)

        hits = torch.zeros(
            self.max_hits,
            4,
            dtype=torch.float32,
        )

        mask = torch.zeros(
            self.max_hits,
            dtype=torch.bool,
        )

        hits[:num_hits] = event
        mask[:num_hits] = True

        return {
            "hits": hits,
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.long),
        }
