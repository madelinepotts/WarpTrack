import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.cosmic_ray import CosmicRayConfig, CosmicRayGenerator
from data.detector_geometry import make_rack_geometry
from visualization.event_display import _prism_faces, display_event


class EventDisplayTests(unittest.TestCase):
    def setUp(self):
        self.rack = make_rack_geometry((4.0, 14.0, 27.0))

    def test_each_scintillator_prism_has_five_faces(self):
        bar = self.rack.bars[0]
        faces = _prism_faces(bar)
        self.assertEqual(len(faces), 5)
        self.assertEqual(len(faces[0]), 3)
        self.assertEqual(len(faces[1]), 3)
        for side in faces[2:]:
            self.assertEqual(len(side), 4)

    def test_plotter_uses_half_triangle_edge_geometry(self):
        h = self.rack.hodoscopes[0]
        left = h.bottom_layer[0]
        interior = h.bottom_layer[1]
        right = h.top_layer[-1]

        self.assertTrue(left.is_half_end)
        self.assertTrue(right.is_half_end)
        self.assertNotEqual(
            set(left.cross_section_vertices_m()),
            set(interior.cross_section_vertices_m()),
        )
        self.assertEqual(
            _prism_faces(left)[0],
            [
                (
                    left.center_x_m - left.length_m / 2.0,
                    left.center_y_m + u,
                    left.center_z_m + z,
                )
                for u, z in left.cross_section_vertices_m()
            ],
        )

    def test_display_builds_for_generated_event(self):
        generator = CosmicRayGenerator(
            self.rack,
            CosmicRayConfig(min_hodoscopes_hit=3),
            seed=12345,
        )
        event = generator.generate_event()
        fig, ax = display_event(self.rack, event, show=False)
        self.assertIsNotNone(fig)
        self.assertEqual(ax.name, "3d")
        plt.close(fig)

    def test_track_line_extends_beyond_rack(self):
        generator = CosmicRayGenerator(
            self.rack,
            CosmicRayConfig(min_hodoscopes_hit=3),
            seed=12345,
        )
        event = generator.generate_event()
        fig, ax = display_event(self.rack, event, show=False)

        track_line = next(
            line for line in ax.lines if line.get_label() == "particle track"
        )
        _, _, zs = track_line.get_data_3d()

        top_z = max(
            h.center_z_m + self.rack.config.detector_height_m / 2.0
            for h in self.rack.hodoscopes
        )
        bottom_z = min(
            h.center_z_m - self.rack.config.detector_height_m / 2.0
            for h in self.rack.hodoscopes
        )

        self.assertGreater(max(zs), top_z)
        self.assertLess(min(zs), bottom_z)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
