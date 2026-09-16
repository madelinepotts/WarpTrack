"""Cosmic-ray track generation for WarpTrack.

The generator samples downward-going straight tracks above the detector rack.
The zenith distribution is configurable as an intensity per unit solid angle
I(theta) proportional to cos(theta)**n.  When tracks are sampled as crossings
of a horizontal generation plane, the projected-area factor contributes one
additional cos(theta), so the crossing-track PDF is proportional to
sin(theta) * cos(theta)**(n + 1).
"""

from dataclasses import dataclass
import math
import random
from typing import Optional, Tuple

from .detector_geometry import RackGeometry
from .particle_track import ParticleTrack
from .track_intersection import ScintillatorHit, intersect_track_rack


@dataclass(frozen=True)
class CosmicRayConfig:
    """Configuration for downward cosmic-ray track generation."""

    cos_theta_power: float = 2.0
    max_zenith_deg: float = 75.0
    generation_margin_m: float = 0.10
    generation_height_m: float = 0.10
    min_hodoscopes_hit: int = 3
    max_attempts: int = 10000

    def __post_init__(self) -> None:
        if self.cos_theta_power <= -2.0:
            raise ValueError("cos_theta_power must be greater than -2")
        if not 0.0 < self.max_zenith_deg < 90.0:
            raise ValueError("max_zenith_deg must be between 0 and 90 degrees")
        if self.generation_margin_m < 0.0 or self.generation_height_m <= 0.0:
            raise ValueError("Generation margin must be nonnegative and height positive")
        if self.min_hodoscopes_hit < 1:
            raise ValueError("min_hodoscopes_hit must be at least 1")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")


@dataclass(frozen=True)
class CosmicRayEvent:
    """One accepted generated primary and its geometric scintillator hits."""

    track: ParticleTrack
    theta_rad: float
    phi_rad: float
    hits: Tuple[ScintillatorHit, ...]

    @property
    def hodoscopes_hit(self) -> Tuple[int, ...]:
        return tuple(sorted({hit.hodoscope_id for hit in self.hits}))


class CosmicRayGenerator:
    """Generate geometry-aware downward cosmic-ray tracks."""

    def __init__(
        self,
        rack: RackGeometry,
        config: Optional[CosmicRayConfig] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        self.rack = rack
        self.config = config or CosmicRayConfig()
        self.rng = random.Random(seed)
        if self.config.min_hodoscopes_hit > len(rack.hodoscopes):
            raise ValueError("min_hodoscopes_hit exceeds number of hodoscopes in rack")

    def sample_angles(self) -> tuple[float, float]:
        """Sample zenith theta and azimuth phi for a horizontal crossing plane.

        ``cos_theta_power`` describes intensity per unit solid angle:
        I(theta) ~ cos(theta)^n.  Crossing a horizontal plane adds the projected
        area factor cos(theta).  Including dOmega = sin(theta)dtheta dphi gives
        p(theta) ~ sin(theta) cos(theta)^(n+1), truncated at max_zenith_deg.
        """
        n = self.config.cos_theta_power
        mu_min = math.cos(math.radians(self.config.max_zenith_deg))

        # For mu=cos(theta), p(mu) ~ mu^(n+1).  Inverse CDF on [mu_min, 1].
        power = n + 2.0
        u = self.rng.random()
        mu = (mu_min**power + u * (1.0 - mu_min**power)) ** (1.0 / power)
        theta = math.acos(mu)
        phi = self.rng.uniform(0.0, 2.0 * math.pi)
        return theta, phi

    def _generation_z(self) -> float:
        top = max(
            h.center_z_m + self.rack.config.detector_height_m / 2.0
            for h in self.rack.hodoscopes
        )
        return top + self.config.generation_height_m

    def sample_track(self) -> tuple[ParticleTrack, float, float]:
        """Sample one downward track before applying detector-trigger cuts."""
        theta, phi = self.sample_angles()
        sin_theta = math.sin(theta)
        direction = (
            sin_theta * math.cos(phi),
            sin_theta * math.sin(phi),
            -math.cos(theta),
        )

        half_x = self.rack.config.detector_width_m / 2.0
        half_y = self.rack.config.detector_depth_m / 2.0
        margin = self.config.generation_margin_m
        origin = (
            self.rng.uniform(-half_x - margin, half_x + margin),
            self.rng.uniform(-half_y - margin, half_y + margin),
            self._generation_z(),
        )
        return ParticleTrack(origin_m=origin, direction=direction), theta, phi

    def generate_event(self) -> CosmicRayEvent:
        """Generate an event crossing at least the requested hodoscope count."""
        for _ in range(self.config.max_attempts):
            track, theta, phi = self.sample_track()
            hits = intersect_track_rack(track, self.rack, forward_only=True)
            hodoscopes_hit = {hit.hodoscope_id for hit in hits}
            if len(hodoscopes_hit) >= self.config.min_hodoscopes_hit:
                return CosmicRayEvent(track=track, theta_rad=theta, phi_rad=phi, hits=hits)

        raise RuntimeError(
            "Could not generate a cosmic-ray event satisfying the hodoscope "
            f"trigger after {self.config.max_attempts} attempts"
        )
