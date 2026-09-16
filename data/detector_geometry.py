"""Canonical detector geometry for WarpTrack.

Each hodoscope contains exactly 25 sensitive triangular-prism scintillators:
16 in the bottom layer and 9 in the top layer.  The first and last
scintillators in each layer are half-triangle edge pieces and are INCLUDED in
those totals.

This module is the single source of truth for Python geometry, plotting,
track intersection, and event generation.  The dimensions and channel mapping
mirror simulation/src/DetectorConstruction.cc.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple

RACK_UNIT_M = 0.04445


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    detector_width_m: float = 0.4826
    detector_depth_m: float = 0.4826 * 17.0 / 10.0
    detector_height_m: float = 2.0 * RACK_UNIT_M
    top_bars: int = 9
    bottom_bars: int = 16

    def __post_init__(self) -> None:
        if (self.bottom_bars, self.top_bars) != (16, 9):
            raise ValueError(
                "Canonical WarpTrack geometry requires 16 bottom + 9 top "
                "= 25 scintillators per hodoscope"
            )
        if min(self.detector_width_m, self.detector_depth_m,
               self.detector_height_m) <= 0.0:
            raise ValueError("Detector dimensions must be positive")

    @property
    def bars_per_hodoscope(self) -> int:
        return 25

    @property
    def layer_height_m(self) -> float:
        return self.detector_height_m / 2.0

    @property
    def bottom_triangle_base_m(self) -> float:
        # Matches Geant4: 2 * depth / 17.
        return 2.0 * self.detector_depth_m / 17.0

    @property
    def bottom_pitch_m(self) -> float:
        return self.bottom_triangle_base_m / 2.0

    @property
    def top_triangle_base_m(self) -> float:
        # Matches Geant4: 2 * width / 10.
        return 2.0 * self.detector_width_m / 10.0

    @property
    def top_pitch_m(self) -> float:
        return self.top_triangle_base_m / 2.0

    @property
    def scintillator_base_m(self) -> float:
        """Common nominal base of the physical scintillators."""
        if abs(self.bottom_triangle_base_m - self.top_triangle_base_m) > 1.0e-12:
            raise ValueError("Top and bottom scintillator bases do not match")
        return self.top_triangle_base_m


@dataclass(frozen=True, slots=True)
class ScintillatorBar:
    channel_id: int
    hodoscope_id: int
    layer_id: int
    bar_id: int
    center_x_m: float
    center_y_m: float
    center_z_m: float
    length_m: float
    triangle_base_m: float
    triangle_height_m: float
    orientation: str
    shape: str = "triangle"
    edge: str | None = None

    @property
    def is_half_end(self) -> bool:
        return self.shape == "half_triangle"

    @property
    def is_sensitive(self) -> bool:
        return True

    def cross_section_vertices_m(self) -> Tuple[Tuple[float, float], ...]:
        """Return local ``(u, z)`` vertices of the sensitive cross-section.

        ``u`` is the segmentation coordinate: global y for bottom bars and
        global x for top bars.  These polygons reproduce the edge-half shapes
        used by DetectorConstruction.cc, so the plotter and intersection code
        consume the same geometry instead of reimplementing it.
        """
        hb = self.triangle_base_m / 2.0
        hh = self.triangle_height_m / 2.0

        if self.shape == "triangle":
            if self.bar_id % 2 == 0:
                return ((-hb, -hh), (hb, -hh), (0.0, hh))
            return ((-hb, hh), (hb, hh), (0.0, -hh))

        if self.shape != "half_triangle":
            raise ValueError(f"Unknown scintillator shape: {self.shape!r}")

        # The Geant4 edge shapes differ between the two perpendicular layers
        # because the bottom layer uses a different rotation convention.
        if self.layer_id == 0 and self.edge == "left":
            return ((-hb, -hh), (-hb, hh), (0.0, -hh))
        if self.layer_id == 0 and self.edge == "right":
            return ((0.0, hh), (hb, hh), (hb, -hh))
        if self.layer_id == 1 and self.edge == "left":
            return ((-hb, hh), (-hb, -hh), (0.0, hh))
        if self.layer_id == 1 and self.edge == "right":
            return ((0.0, hh), (hb, hh), (hb, -hh))

        raise ValueError(
            f"Invalid half-triangle layer/edge combination: "
            f"layer={self.layer_id}, edge={self.edge!r}"
        )


@dataclass(frozen=True, slots=True)
class Hodoscope:
    hodoscope_id: int
    rack_u: float
    center_z_m: float
    bars: Tuple[ScintillatorBar, ...]

    @property
    def top_layer(self) -> Tuple[ScintillatorBar, ...]:
        return tuple(bar for bar in self.bars if bar.layer_id == 1)

    @property
    def bottom_layer(self) -> Tuple[ScintillatorBar, ...]:
        return tuple(bar for bar in self.bars if bar.layer_id == 0)


@dataclass(frozen=True, slots=True)
class RackGeometry:
    config: GeometryConfig
    hodoscopes: Tuple[Hodoscope, ...]

    @property
    def bars(self) -> Tuple[ScintillatorBar, ...]:
        return tuple(bar for hodoscope in self.hodoscopes for bar in hodoscope.bars)

    def bar_by_channel(self, channel_id: int) -> ScintillatorBar:
        if channel_id < 0 or channel_id >= len(self.bars):
            raise KeyError(f"Unknown channel ID {channel_id}")
        bar = self.bars[channel_id]
        if bar.channel_id != channel_id:
            raise RuntimeError("Geometry channel ordering is inconsistent")
        return bar


def _shape_for_bar(bar_id: int, count: int) -> tuple[str, str | None]:
    if bar_id == 0:
        return "half_triangle", "left"
    if bar_id == count - 1:
        return "half_triangle", "right"
    return "triangle", None


def _make_hodoscope(
    hodoscope_id: int,
    rack_u: float,
    config: GeometryConfig,
    first_channel_id: int,
) -> Hodoscope:
    center_z = rack_u * RACK_UNIT_M
    layer_height = config.layer_height_m
    bars = []

    bottom_z = center_z - layer_height / 2.0
    for bar_id in range(config.bottom_bars):
        shape, edge = _shape_for_bar(bar_id, config.bottom_bars)
        center_y = (
            -config.detector_depth_m / 2.0
            + config.bottom_triangle_base_m / 2.0
            + bar_id * config.bottom_pitch_m
        )
        bars.append(ScintillatorBar(
            channel_id=first_channel_id + bar_id,
            hodoscope_id=hodoscope_id,
            layer_id=0,
            bar_id=bar_id,
            center_x_m=0.0,
            center_y_m=center_y,
            center_z_m=bottom_z,
            length_m=config.detector_width_m,
            triangle_base_m=config.bottom_triangle_base_m,
            triangle_height_m=layer_height,
            orientation="x",
            shape=shape,
            edge=edge,
        ))

    top_z = center_z + layer_height / 2.0
    for bar_id in range(config.top_bars):
        shape, edge = _shape_for_bar(bar_id, config.top_bars)
        center_x = (
            -config.detector_width_m / 2.0
            + config.top_triangle_base_m / 2.0
            + bar_id * config.top_pitch_m
        )
        bars.append(ScintillatorBar(
            channel_id=first_channel_id + config.bottom_bars + bar_id,
            hodoscope_id=hodoscope_id,
            layer_id=1,
            bar_id=bar_id,
            center_x_m=center_x,
            center_y_m=0.0,
            center_z_m=top_z,
            length_m=config.detector_depth_m,
            triangle_base_m=config.top_triangle_base_m,
            triangle_height_m=layer_height,
            orientation="y",
            shape=shape,
            edge=edge,
        ))

    hodoscope = Hodoscope(hodoscope_id, rack_u, center_z, tuple(bars))
    if len(hodoscope.bottom_layer) != 16 or len(hodoscope.top_layer) != 9:
        raise RuntimeError("Invalid hodoscope segmentation")
    if len(hodoscope.bars) != 25:
        raise RuntimeError("Each hodoscope must contain exactly 25 scintillators")
    return hodoscope


def make_rack_geometry(
    rack_u_positions: Iterable[float] = (4.0, 14.0, 27.0),
    config: GeometryConfig | None = None,
) -> RackGeometry:
    config = config or GeometryConfig()
    positions = tuple(float(position) for position in rack_u_positions)

    if len(positions) < 3:
        raise ValueError("WarpTrack rack geometry requires at least 3 hodoscopes")
    if len(set(positions)) != len(positions):
        raise ValueError("Hodoscope rack-U positions must be unique")

    hodoscopes = tuple(
        _make_hodoscope(i, rack_u, config, i * config.bars_per_hodoscope)
        for i, rack_u in enumerate(positions)
    )
    geometry = RackGeometry(config, hodoscopes)

    expected = list(range(len(hodoscopes) * config.bars_per_hodoscope))
    actual = [bar.channel_id for bar in geometry.bars]
    if actual != expected:
        raise RuntimeError("Channel IDs must be contiguous 25-channel blocks")

    return geometry
