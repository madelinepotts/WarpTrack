import unittest

from data.detector_geometry import (
    GeometryConfig,
    RACK_UNIT_M,
    make_rack_geometry,
)


class TestDetectorGeometry(unittest.TestCase):
    def test_default_hodoscope_has_25_scintillators(self):
        geometry = make_rack_geometry()

        for hodoscope in geometry.hodoscopes:
            self.assertEqual(len(hodoscope.bars), 25)
            self.assertEqual(len(hodoscope.top_layer), 9)
            self.assertEqual(len(hodoscope.bottom_layer), 16)

    def test_layers_are_perpendicular(self):
        geometry = make_rack_geometry()
        hodoscope = geometry.hodoscopes[0]

        self.assertTrue(all(bar.orientation == "x" for bar in hodoscope.bottom_layer))
        self.assertTrue(all(bar.orientation == "y" for bar in hodoscope.top_layer))

    def test_layers_touch_and_fill_two_u_height(self):
        geometry = make_rack_geometry()
        config = geometry.config
        hodoscope = geometry.hodoscopes[0]

        bottom = hodoscope.bottom_layer[0]
        top = hodoscope.top_layer[0]

        bottom_upper_face = bottom.center_z_m + bottom.triangle_height_m / 2.0
        top_lower_face = top.center_z_m - top.triangle_height_m / 2.0

        self.assertAlmostEqual(bottom_upper_face, top_lower_face)
        self.assertAlmostEqual(config.detector_height_m, 2.0 * RACK_UNIT_M)

    def test_scintillator_dimensions_match_between_layers(self):
        geometry = make_rack_geometry()
        hodoscope = geometry.hodoscopes[0]

        bottom = hodoscope.bottom_layer[0]
        top = hodoscope.top_layer[0]

        self.assertAlmostEqual(bottom.triangle_base_m, top.triangle_base_m)
        self.assertAlmostEqual(bottom.triangle_height_m, top.triangle_height_m)

    def test_channel_ids_are_unique(self):
        geometry = make_rack_geometry((3, 11, 24, 35))
        channel_ids = [bar.channel_id for bar in geometry.bars]

        self.assertEqual(len(channel_ids), 4 * 25)
        self.assertEqual(len(channel_ids), len(set(channel_ids)))

    def test_nonuniform_rack_positions_are_preserved(self):
        positions = (3.0, 12.0, 29.0)
        geometry = make_rack_geometry(positions)

        self.assertEqual(
            tuple(hodoscope.rack_u for hodoscope in geometry.hodoscopes),
            positions,
        )
        self.assertAlmostEqual(geometry.hodoscopes[1].center_z_m, 12.0 * RACK_UNIT_M)

    def test_requires_at_least_three_hodoscopes(self):
        with self.assertRaises(ValueError):
            make_rack_geometry((4, 12))

    def test_incompatible_dimensions_rejected_when_geometry_is_built(self):
        config = GeometryConfig(detector_width_m=0.50, detector_depth_m=0.70)

        with self.assertRaises(ValueError):
            make_rack_geometry(config=config)

    def test_half_end_elements_exist_on_each_layer(self):
        geometry = make_rack_geometry()
        hodoscope = geometry.hodoscopes[0]

        self.assertEqual(sum(bar.is_half_end for bar in hodoscope.top_layer), 2)
        self.assertEqual(sum(bar.is_half_end for bar in hodoscope.bottom_layer), 2)


if __name__ == "__main__":
    unittest.main()
