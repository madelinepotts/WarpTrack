import unittest
from data.detector_geometry import make_rack_geometry


class GeometryTessellationTests(unittest.TestCase):
    def setUp(self):
        self.rack = make_rack_geometry((4.0, 14.0, 27.0))

    def test_adjacent_scintillators_use_half_base_pitch(self):
        h = self.rack.hodoscopes[0]
        cfg = self.rack.config
        for left, right in zip(h.bottom_layer, h.bottom_layer[1:]):
            self.assertAlmostEqual(
                right.center_y_m - left.center_y_m,
                cfg.bottom_pitch_m,
                places=12,
            )
        for left, right in zip(h.top_layer, h.top_layer[1:]):
            self.assertAlmostEqual(
                right.center_x_m - left.center_x_m,
                cfg.top_pitch_m,
                places=12,
            )

    def test_python_dimensions_match_geant4_formulas(self):
        cfg = self.rack.config
        self.assertAlmostEqual(
            cfg.bottom_triangle_base_m,
            2.0 * cfg.detector_depth_m / 17.0,
            places=12,
        )
        self.assertAlmostEqual(
            cfg.top_triangle_base_m,
            2.0 * cfg.detector_width_m / 10.0,
            places=12,
        )
        self.assertAlmostEqual(
            cfg.bottom_triangle_base_m,
            cfg.top_triangle_base_m,
            places=12,
        )

    def test_edge_halves_are_included_in_25_channels(self):
        h = self.rack.hodoscopes[0]
        self.assertEqual(len(h.bottom_layer), 16)
        self.assertEqual(len(h.top_layer), 9)
        self.assertEqual(len(h.bars), 25)
        self.assertEqual([b.channel_id for b in h.bars], list(range(25)))
        self.assertTrue(h.bottom_layer[0].is_half_end)
        self.assertTrue(h.bottom_layer[-1].is_half_end)
        self.assertTrue(h.top_layer[0].is_half_end)
        self.assertTrue(h.top_layer[-1].is_half_end)


if __name__ == "__main__":
    unittest.main()
