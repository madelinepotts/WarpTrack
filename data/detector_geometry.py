"""Configurable first-pass geometry for the WarpTrack scintillator hodoscopes.

The model intentionally separates detector geometry from event generation.  A
rack contains three or more hodoscopes at arbitrary rack-unit (U) positions.
Each hodoscope contains two touching, perpendicular layers of equal-sized
triangular-prism scintillators: 9 in the top layer and 16 in the bottom layer.

The default dimensions are approximate placeholders.  They can be replaced by
measured dimensions without changing the rest of the simulation code.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple


RACK_UNIT_M = 0.04445  # 1U = 1.75 in


@dataclass(frozen=True)
class GeometryConfig:
    """Dimensions and segmentation of one hodoscope.

    ``detector_width_m`` and ``detector_depth_m`` describe the complete
    two-layer hodoscope envelope.  The default depth is chosen so equal-width
    scintillators can tile 9 across one axis and 16 across the perpendicular
    axis.  These are placeholders until measured detector dimensions are used.
    """

    detector_width_m: float = 0.4826  # approximate 19-inch rack width
    detector_depth_m: float = 0.4826 * 16.0 / 9.0
    detector_height_m: float = 2.0 * RACK_UNIT_M
    top_bars: int = 9
    bottom_bars: int = 16

    def __post_init__(self):
        if self.top_bars <= 0 or self.bottom_bars <= 0:
            raise ValueError("Each layer must contain at least one scintillator")
        if min(self.detector_width_m, self.detector_depth_m, self.detector_height_m) <= 0:
            raise ValueError("Detector dimensions must be positive")

    @property
    def bars_per_hodoscope(self) -> int:
        return self.top_bars + self.bottom_bars

    @property
    def layer_height_m(self) -> float:
        # First-pass model: the two triangular layers touch and divide the
        # detector height equally.
        return self.detector_height_m / 2.0

    @property
    def scintillator_base_m(self) -> float:
        """Nominal triangle base shared by both layers.

        Top bars divide detector width; bottom bars divide detector depth.
        Equal physical scintillator dimensions require these pitches to match.
        """
        top_pitch = self.detector_width_m / self.top_bars
        bottom_pitch = self.detector_depth_m / self.bottom_bars
        if abs(top_pitch - bottom_pitch) > 1.0e-9:
            raise ValueError(
                "Geometry does not permit equal-size scintillators: "
                "detector_width_m/top_bars must equal "
                "detector_depth_m/bottom_bars"
            )
        return top_pitch


@dataclass(frozen=True)
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
    orientation: str  # "x" means prism runs along x, "y" along y
    is_half_end: bool = False


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class RackGeometry:
    config: GeometryConfig
    hodoscopes: Tuple[Hodoscope, ...]

    @property
    def bars(self) -> Tuple[ScintillatorBar, ...]:
        return tuple(bar for hodoscope in self.hodoscopes for bar in hodoscope.bars)


def _make_hodoscope(
    hodoscope_id: int,
    rack_u: float,
    config: GeometryConfig,
    first_channel_id: int,
) -> Hodoscope:
    """Build one two-layer hodoscope centered at a rack-U position."""

    center_z = rack_u * RACK_UNIT_M
    layer_height = config.layer_height_m
    base = config.scintillator_base_m
    bars = []
    channel_id = first_channel_id

    # Bottom layer: 16 bars.  Their long axes run along x and segmentation is
    # along y.  The layer center is half a layer below the hodoscope center.
    bottom_z = center_z - layer_height / 2.0
    for bar_id in range(config.bottom_bars):
        center_y = -config.detector_depth_m / 2.0 + (bar_id + 0.5) * base
        bars.append(
            ScintillatorBar(
                channel_id=channel_id,
                hodoscope_id=hodoscope_id,
                layer_id=0,
                bar_id=bar_id,
                center_x_m=0.0,
                center_y_m=center_y,
                center_z_m=bottom_z,
                length_m=config.detector_width_m,
                triangle_base_m=base,
                triangle_height_m=layer_height,
                orientation="x",
                is_half_end=bar_id in (0, config.bottom_bars - 1),
            )
        )
        channel_id += 1

    # Top layer: 9 bars.  Rotate the segmentation by 90 degrees.  The long
    # axes run along y and the layer touches the bottom layer at z=center_z.
    top_z = center_z + layer_height / 2.0
    for bar_id in range(config.top_bars):
        center_x = -config.detector_width_m / 2.0 + (bar_id + 0.5) * base
        bars.append(
            ScintillatorBar(
                channel_id=channel_id,
                hodoscope_id=hodoscope_id,
                layer_id=1,
                bar_id=bar_id,
                center_x_m=center_x,
                center_y_m=0.0,
                center_z_m=top_z,
                length_m=config.detector_depth_m,
                triangle_base_m=base,
                triangle_height_m=layer_height,
                orientation="y",
                is_half_end=bar_id in (0, config.top_bars - 1),
            )
        )
        channel_id += 1

    return Hodoscope(
        hodoscope_id=hodoscope_id,
        rack_u=rack_u,
        center_z_m=center_z,
        bars=tuple(bars),
    )


def make_rack_geometry(
    rack_u_positions: Iterable[float] = (4.0, 14.0, 27.0),
    config: GeometryConfig | None = None,
) -> RackGeometry:
    """Create a rack with hodoscopes at arbitrary U positions.

    The default positions are illustrative only and deliberately nonuniform to
    exercise the fact that servers/hardware may occupy space between detectors.
    Pass measured rack-U positions when they are available.
    """

    if config is None:
        config = GeometryConfig()

    positions = tuple(float(position) for position in rack_u_positions)
    if len(positions) < 3:
        raise ValueError("WarpTrack rack geometry requires at least 3 hodoscopes")
    if len(set(positions)) != len(positions):
        raise ValueError("Hodoscope rack-U positions must be unique")

    hodoscopes = []
    next_channel = 0
    for hodoscope_id, rack_u in enumerate(positions):
        hodoscope = _make_hodoscope(
            hodoscope_id=hodoscope_id,
            rack_u=rack_u,
            config=config,
            first_channel_id=next_channel,
        )
        hodoscopes.append(hodoscope)
        next_channel += config.bars_per_hodoscope

    return RackGeometry(config=config, hodoscopes=tuple(hodoscopes))
