"""Generate and inspect one geometry-aware WarpTrack cosmic-ray event."""

from data.cosmic_ray import CosmicRayConfig, CosmicRayGenerator
from data.detector_geometry import make_rack_geometry
from visualization.event_display import display_event, print_event_summary


def main() -> None:
    # Rack-U positions remain illustrative placeholders until measured positions
    # are supplied. They are intentionally nonuniform.
    rack = make_rack_geometry((4.0, 14.0, 27.0))

    generator = CosmicRayGenerator(
        rack,
        CosmicRayConfig(
            cos_theta_power=2.0,
            min_hodoscopes_hit=3,
        ),
        seed=None,
    )

    event = generator.generate_event()
    print_event_summary(event)

    # Build the same figure once, save it, then show the interactive Matplotlib view.
    fig, _ = display_event(rack, event, show=False)
    fig.savefig("warptrack_event.png", dpi=180)
    print("\nSaved static event image: warptrack_event.png")

    # Matplotlib's 3D window is interactive: drag to rotate and use the toolbar
    # to zoom/pan. Reuse the existing figure rather than generating a new event.
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
