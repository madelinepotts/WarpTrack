"""Track intersections with WarpTrack hodoscope scintillator layers.

This module performs exact line clipping against triangular-prism sensitive
volumes.  The prism axis is the long direction of a scintillator bar; the
cross-section is an isosceles triangle in the segmentation/z plane.

The current detector geometry does not yet encode the measured end-piece
profile, so the first and last channels use the same triangular-prism
intersection as interior channels.  Keeping that limitation here makes it
explicit and easy to replace when measured end geometry is available.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Tuple

from .detector_geometry import RackGeometry, ScintillatorBar
from .particle_track import ParticleTrack, Vector3


_EPS = 1.0e-12
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


@dataclass(frozen=True, slots=True)
class ScintillatorHit:
    """Geometric passage of one track through one scintillator sensitive volume."""

    channel_id: int
    hodoscope_id: int
    layer_id: int
    bar_id: int
    entry_m: Vector3
    exit_m: Vector3
    path_length_m: float
    track_distance_m: float
    time_ns: float

    @property
    def midpoint_m(self) -> Vector3:
        return tuple((a + b) * 0.5 for a, b in zip(self.entry_m, self.exit_m))


def _clip_halfspace(
    lo: float,
    hi: float,
    a: float,
    b: float,
) -> tuple[float, float] | None:
    """Clip s in [lo, hi] against a + b*s >= 0."""
    if abs(b) < _EPS:
        return (lo, hi) if a >= -_EPS else None

    crossing = -a / b
    if b > 0.0:
        lo = max(lo, crossing)
    else:
        hi = min(hi, crossing)

    if lo > hi + _EPS:
        return None
    return lo, hi


def _triangle_halfspaces(bar: ScintillatorBar):
    """Return local (u,z) halfspaces for an isosceles triangular cross-section.

    ``u`` is transverse to the long prism axis and ``z`` is measured from the
    bar center. Neighboring channels alternate triangle orientation in this
    first-pass packing model.
    """
    half_base = bar.triangle_base_m / 2.0
    half_height = bar.triangle_height_m / 2.0
    slope = 2.0 * half_height / half_base

    # Each tuple is c + au*u + az*z >= 0.
    if bar.bar_id % 2 == 0:
        # Up-pointing: (-hb,-hh), (+hb,-hh), (0,+hh)
        return (
            (half_height, 0.0, 1.0),       # z >= -hh
            (half_height, -slope, -1.0),   # z <= hh - slope*u
            (half_height, slope, -1.0),    # z <= hh + slope*u
        )

    # Down-pointing: (-hb,+hh), (+hb,+hh), (0,-hh)
    return (
        (half_height, 0.0, -1.0),          # z <= +hh
        (half_height, -slope, 1.0),        # z >= -hh + slope*u
        (half_height, slope, 1.0),         # z >= -hh - slope*u
    )


def intersect_track_bar(
    track: ParticleTrack,
    bar: ScintillatorBar,
) -> ScintillatorHit | None:
    """Return the track segment inside ``bar``, or ``None`` when it misses.

    The track is treated as an infinite line geometrically.  For a generated
    downward cosmic-ray track, callers should choose an origin above the rack;
    the returned intersections can then be filtered to positive track distance
    if a ray rather than an infinite line is required.
    """

    # Local prism-axis coordinate v and transverse coordinate u.
    if bar.orientation == "x":
        v0 = track.x0_m - bar.center_x_m
        dv = track.dx
        u0 = track.y0_m - bar.center_y_m
        du = track.dy
    elif bar.orientation == "y":
        v0 = track.y0_m - bar.center_y_m
        dv = track.dy
        u0 = track.x0_m - bar.center_x_m
        du = track.dx
    else:
        raise ValueError(f"Unknown scintillator orientation: {bar.orientation!r}")

    z0 = track.z0_m - bar.center_z_m
    dz = track.dz

    lo = -math.inf
    hi = math.inf

    # Long-axis prism bounds: -L/2 <= v <= L/2.
    for a, b in (
        (bar.length_m / 2.0 - v0, -dv),
        (bar.length_m / 2.0 + v0, dv),
    ):
        clipped = _clip_halfspace(lo, hi, a, b)
        if clipped is None:
            return None
        lo, hi = clipped

    # Triangular cross-section.
    for c, au, az in _triangle_halfspaces(bar):
        a = c + au * u0 + az * z0
        b = au * du + az * dz
        clipped = _clip_halfspace(lo, hi, a, b)
        if clipped is None:
            return None
        lo, hi = clipped

    if not math.isfinite(lo) or not math.isfinite(hi) or hi - lo <= _EPS:
        return None

    entry = track.point_at(lo)
    exit_ = track.point_at(hi)
    return ScintillatorHit(
        channel_id=bar.channel_id,
        hodoscope_id=bar.hodoscope_id,
        layer_id=bar.layer_id,
        bar_id=bar.bar_id,
        entry_m=entry,
        exit_m=exit_,
        path_length_m=hi - lo,
        track_distance_m=(lo + hi) * 0.5,
        time_ns=((lo + hi) * 0.5 / SPEED_OF_LIGHT_M_PER_S) * 1.0e9,
    )


def intersect_track_rack(
    track: ParticleTrack,
    rack: RackGeometry,
    *,
    forward_only: bool = True,
) -> Tuple[ScintillatorHit, ...]:
    """Intersect a track with every sensitive scintillator in the rack.

    Results are ordered along the particle trajectory.  ``forward_only=True``
    treats ``ParticleTrack`` as a ray beginning at ``origin_m``; this is the
    normal choice for generated cosmic rays.
    """
    hits = []
    for bar in rack.bars:
        hit = intersect_track_bar(track, bar)
        if hit is None:
            continue

        if forward_only:
            # Signed distance of entry point from the track origin.  Direction
            # is unit length, so this is the track parameter s in metres.
            s_entry = sum(
                (p - o) * d
                for p, o, d in zip(hit.entry_m, track.origin_m, track.direction)
            )
            s_exit = s_entry + hit.path_length_m
            if s_exit <= 0.0:
                continue

        hits.append(hit)

    # Keep the ordering criterion explicit rather than enabling dataclass
    # order=True, which would compare every field in declaration order.
    hits.sort(key=lambda hit: hit.time_ns)
    return tuple(hits)
