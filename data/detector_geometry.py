"""WarpTrack detector geometry loaded from geometry/detector_geometry.json.

The JSON file is the single source of truth used by both Python and Geant4.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple
import json

_GEOMETRY_FILE = (
    Path(__file__).resolve().parents[1] / "geometry" / "detector_geometry.json"
)


def _load_definition():
    with _GEOMETRY_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


_DEF = _load_definition()
RACK_UNIT_M = _DEF["rack"]["rack_unit_height"] / 1000.0


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    detector_width_m: float = _DEF["rack"]["width"] / 1000.0
    detector_depth_m: float = _DEF["rack"]["depth"] / 1000.0
    detector_height_m: float = _DEF["hodoscope"]["height"] / 1000.0
    top_bars: int = _DEF["layers"][1]["count"]
    bottom_bars: int = _DEF["layers"][0]["count"]

    @property
    def bars_per_hodoscope(self):
        return _DEF["hodoscope"]["channels_per_hodoscope"]

    @property
    def layer_height_m(self):
        return self.detector_height_m / 2.0

    @property
    def bottom_triangle_base_m(self):
        return 2 * self.detector_depth_m / (self.bottom_bars - 1)

    @property
    def bottom_pitch_m(self):
        return self.bottom_triangle_base_m / 2

    @property
    def top_triangle_base_m(self):
        return 2 * self.detector_width_m / (self.top_bars - 1)

    @property
    def top_pitch_m(self):
        return self.top_triangle_base_m / 2

    @property
    def scintillator_base_m(self):
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
    vertices_normalized: Tuple[Tuple[float, float], ...] = ()

    @property
    def is_half_end(self):
        return self.shape == "half_triangle"

    @property
    def is_sensitive(self):
        return True

    def cross_section_vertices_m(self):
        if self.vertices_normalized:
            vertices = self.vertices_normalized
        elif self.shape == "triangle":
            vertices = (
                ((-0.5, -0.5), (0.5, -0.5), (0.0, 0.5))
                if self.bar_id % 2 == 0
                else ((-0.5, 0.5), (0.5, 0.5), (0.0, -0.5))
            )
        elif self.edge == "left":
            vertices = ((-0.5, -0.5), (-0.5, 0.5), (0.0, -0.5))
        elif self.layer_id == 0:
            vertices = ((0.0, 0.5), (0.5, 0.5), (0.5, -0.5))
        else:
            vertices = ((0.0, -0.5), (0.5, -0.5), (0.5, 0.5))
        return tuple(
            (u * self.triangle_base_m, z * self.triangle_height_m) for u, z in vertices
        )


@dataclass(frozen=True, slots=True)
class Hodoscope:
    hodoscope_id: int
    rack_u: float
    center_z_m: float
    bars: Tuple[ScintillatorBar, ...]

    @property
    def top_layer(self):
        return tuple(b for b in self.bars if b.layer_id == 1)

    @property
    def bottom_layer(self):
        return tuple(b for b in self.bars if b.layer_id == 0)


@dataclass(frozen=True, slots=True)
class RackGeometry:
    config: GeometryConfig
    hodoscopes: Tuple[Hodoscope, ...]

    @property
    def bars(self):
        return tuple(b for h in self.hodoscopes for b in h.bars)

    def bar_by_channel(self, channel_id):
        for b in self.bars:
            if b.channel_id == channel_id:
                return b
        raise KeyError(channel_id)


def _make_hodoscope(hid, rack_u, cfg, first):
    cz = rack_u * RACK_UNIT_M
    lh = cfg.layer_height_m
    bars = []
    for ld in _DEF["layers"]:
        lid = ld["id"]
        n = ld["count"]
        base = cfg.bottom_triangle_base_m if lid == 0 else cfg.top_triangle_base_m
        pitch = base / 2
        span = cfg.detector_depth_m if lid == 0 else cfg.detector_width_m
        length = cfg.detector_width_m if lid == 0 else cfg.detector_depth_m
        layer_z = cz + (-lh / 2 if lid == 0 else lh / 2)
        for p in ld["pieces"]:
            u = -span / 2 + base / 2 + p["center_index"] * pitch
            bars.append(
                ScintillatorBar(
                    first + ld["channel_offset"] + p["bar_id"],
                    hid,
                    lid,
                    p["bar_id"],
                    0.0 if lid == 0 else u,
                    u if lid == 0 else 0.0,
                    layer_z,
                    length,
                    base,
                    lh,
                    ld["orientation"],
                    p["shape"],
                    p["edge"],
                    tuple(tuple(v) for v in p["vertices"]),
                )
            )
    bars.sort(key=lambda b: b.channel_id)
    return Hodoscope(hid, rack_u, cz, tuple(bars))


def make_rack_geometry(
    rack_u_positions: Iterable[float] | None = None,
    config: GeometryConfig | None = None,
):
    cfg = config or GeometryConfig()
    positions = (
        tuple(rack_u_positions)
        if rack_u_positions is not None
        else tuple(i["rack_u"] for i in _DEF["instances"])
    )
    hs = tuple(
        _make_hodoscope(i, float(u), cfg, i * cfg.bars_per_hodoscope)
        for i, u in enumerate(positions)
    )
    return RackGeometry(cfg, hs)
