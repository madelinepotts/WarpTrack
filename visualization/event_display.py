"""3D event display for WarpTrack geometry-aware cosmic-ray events."""

from __future__ import annotations

import math
from typing import Iterable

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from data.cosmic_ray import CosmicRayEvent
from data.detector_geometry import RackGeometry, ScintillatorBar


def _triangle_vertices(bar: ScintillatorBar):
    """Return the exact sensitive cross-section from detector_geometry."""
    return list(bar.cross_section_vertices_m())


def _prism_faces(bar: ScintillatorBar):
    """Build the five polygon faces of one triangular scintillator prism."""
    tri = _triangle_vertices(bar)
    half_len = bar.length_m / 2.0

    if bar.orientation == "x":
        end0 = [
            (bar.center_x_m - half_len, bar.center_y_m + u, bar.center_z_m + z)
            for u, z in tri
        ]
        end1 = [
            (bar.center_x_m + half_len, bar.center_y_m + u, bar.center_z_m + z)
            for u, z in tri
        ]
    elif bar.orientation == "y":
        end0 = [
            (bar.center_x_m + u, bar.center_y_m - half_len, bar.center_z_m + z)
            for u, z in tri
        ]
        end1 = [
            (bar.center_x_m + u, bar.center_y_m + half_len, bar.center_z_m + z)
            for u, z in tri
        ]
    else:
        raise ValueError(f"Unknown scintillator orientation {bar.orientation!r}")

    return [
        end0,
        end1,
        [end0[0], end0[1], end1[1], end1[0]],
        [end0[1], end0[2], end1[2], end1[1]],
        [end0[2], end0[0], end1[0], end1[2]],
    ]


def _set_equal_axes(ax, rack: RackGeometry, event: CosmicRayEvent) -> None:
    """Give the 3D plot comparable physical scaling on x, y and z."""
    cfg = rack.config
    xs = [-cfg.detector_width_m / 2.0, cfg.detector_width_m / 2.0]
    ys = [-cfg.detector_depth_m / 2.0, cfg.detector_depth_m / 2.0]
    detector_bottom = min(h.center_z_m for h in rack.hodoscopes) - cfg.detector_height_m / 2.0
    detector_top = max(h.center_z_m for h in rack.hodoscopes) + cfg.detector_height_m / 2.0
    rack_span = max(detector_top - detector_bottom, cfg.detector_height_m)
    upper_display_margin = max(0.35, 0.35 * rack_span)
    lower_display_margin = max(0.15, 0.15 * rack_span)
    zs = [
        detector_bottom - lower_display_margin,
        detector_top + upper_display_margin,
    ]

    xmid = sum(xs) / 2.0
    ymid = sum(ys) / 2.0
    zmid = (min(zs) + max(zs)) / 2.0
    radius = max(
        xs[1] - xs[0],
        ys[1] - ys[0],
        max(zs) - min(zs),
    ) / 2.0 * 1.08

    ax.set_xlim(xmid - radius, xmid + radius)
    ax.set_ylim(ymid - radius, ymid + radius)
    ax.set_zlim(zmid - radius, zmid + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def print_event_summary(event: CosmicRayEvent) -> None:
    """Print a compact numerical cross-check for the displayed event."""
    print("Cosmic Ray")
    print("----------")
    print(f"theta:              {math.degrees(event.theta_rad):8.3f} deg")
    print(f"phi:                {math.degrees(event.phi_rad):8.3f} deg")
    print(f"hodoscopes hit:     {len(event.hodoscopes_hit):8d}")
    print(f"scintillators hit:  {len(event.hits):8d}")
    print()

    print("Hits (chronological)")
    print("--------------------")
    print(" #  H  L  Bar  Ch     x [m]     y [m]     z [m]   path [cm]   time [ns]")
    for i, hit in enumerate(event.hits):
        x, y, z = hit.midpoint_m
        print(
            f"{i:2d} {hit.hodoscope_id:2d} {hit.layer_id:2d} "
            f"{hit.bar_id:4d} {hit.channel_id:3d} "
            f"{x:9.4f} {y:9.4f} {z:9.4f} "
            f"{100.0 * hit.path_length_m:10.3f} {hit.time_ns:11.4f}"
        )


def display_event(
    rack: RackGeometry,
    event: CosmicRayEvent,
    *,
    show: bool = True,
):
    """Display rack scintillators, the generated track, and reported hit bars.

    Non-hit scintillators are drawn lightly. Hit scintillators are emphasized,
    and hit midpoints are marked. The track segment spans from its generated
    origin to just below the lowest hodoscope.
    """
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    hit_channels = {hit.channel_id for hit in event.hits}

    for bar in rack.bars:
        faces = _prism_faces(bar)
        hit = bar.channel_id in hit_channels
        poly = Poly3DCollection(
            faces,
            alpha=0.78 if hit else 0.08,
            linewidths=1.0 if hit else 0.25,
            edgecolors="black",
            facecolors="orange" if hit else "lightblue",
        )
        ax.add_collection3d(poly)

    # Display the trajectory with an explicit margin above the highest
    # hodoscope and below the lowest one.  Do not rely on the generated origin
    # being visually far enough above the rack.
    top_z = max(
        h.center_z_m + rack.config.detector_height_m / 2.0
        for h in rack.hodoscopes
    )
    bottom_z = min(
        h.center_z_m - rack.config.detector_height_m / 2.0
        for h in rack.hodoscopes
    )

    rack_span = max(top_z - bottom_z, rack.config.detector_height_m)
    # Give the incoming trajectory extra visual room above the rack so its
    # direction is obvious before it reaches the top hodoscope.
    upper_display_margin = max(0.35, 0.35 * rack_span)
    lower_display_margin = max(0.15, 0.15 * rack_span)
    start_z = top_z + upper_display_margin
    end_z = bottom_z - lower_display_margin

    start = event.track.point_at_z(start_z)
    end = event.track.point_at_z(end_z)

    x0, y0, z0 = start
    x1, y1, z1 = end
    ax.plot([x0, x1], [y0, y1], [z0, z1], linewidth=2.0, label="particle track")

    if event.hits:
        mids = [hit.midpoint_m for hit in event.hits]
        ax.scatter(
            [p[0] for p in mids],
            [p[1] for p in mids],
            [p[2] for p in mids],
            s=48,
            c="red",
            depthshade=False,
            label="hit midpoint",
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(
        "WarpTrack event display\n"
        f"theta={math.degrees(event.theta_rad):.1f} deg, "
        f"phi={math.degrees(event.phi_rad):.1f} deg, "
        f"{len(event.hits)} scintillator hits"
    )
    ax.legend()
    _set_equal_axes(ax, rack, event)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax
