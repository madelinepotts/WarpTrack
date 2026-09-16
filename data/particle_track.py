"""Particle trajectories used by WarpTrack's geometry-aware simulation."""

from dataclasses import dataclass
import math
from typing import Tuple


Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class ParticleTrack:
    """An infinite straight particle trajectory in rack coordinates.

    ``origin_m`` is any point on the trajectory. ``direction`` may be supplied
    at any non-zero magnitude and is normalized when the track is created.

    The parameterization is::

        r(s) = origin_m + s * direction

    where ``s`` is distance in metres because ``direction`` is unit length.
    """

    origin_m: Vector3
    direction: Vector3

    def __post_init__(self) -> None:
        if len(self.origin_m) != 3 or len(self.direction) != 3:
            raise ValueError("origin_m and direction must each contain x, y, z")

        origin = tuple(float(value) for value in self.origin_m)
        direction = tuple(float(value) for value in self.direction)

        if not all(math.isfinite(value) for value in origin + direction):
            raise ValueError("ParticleTrack values must be finite")

        magnitude = math.sqrt(sum(component * component for component in direction))
        if magnitude == 0.0:
            raise ValueError("ParticleTrack direction cannot be the zero vector")

        unit_direction = tuple(component / magnitude for component in direction)
        object.__setattr__(self, "origin_m", origin)
        object.__setattr__(self, "direction", unit_direction)

    @property
    def x0_m(self) -> float:
        return self.origin_m[0]

    @property
    def y0_m(self) -> float:
        return self.origin_m[1]

    @property
    def z0_m(self) -> float:
        return self.origin_m[2]

    @property
    def dx(self) -> float:
        return self.direction[0]

    @property
    def dy(self) -> float:
        return self.direction[1]

    @property
    def dz(self) -> float:
        return self.direction[2]

    def point_at(self, distance_m: float) -> Vector3:
        """Return the point ``distance_m`` along the track from ``origin_m``."""
        s = float(distance_m)
        return tuple(
            origin + s * direction
            for origin, direction in zip(self.origin_m, self.direction)
        )

    def distance_to_z(self, z_m: float) -> float:
        """Return signed path distance needed to reach a plane at ``z_m``."""
        if abs(self.dz) < 1.0e-12:
            raise ValueError("Track is parallel to constant-z planes")
        return (float(z_m) - self.z0_m) / self.dz

    def point_at_z(self, z_m: float) -> Vector3:
        """Return where the track crosses the horizontal plane ``z=z_m``."""
        return self.point_at(self.distance_to_z(z_m))
