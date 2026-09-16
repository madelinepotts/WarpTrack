import unittest
from data.detector_geometry import make_rack_geometry

class GeometryTessellationTests(unittest.TestCase):
    def setUp(self):
        self.rack = make_rack_geometry((4.0, 14.0, 27.0))

    def test_adjacent_triangles_use_half_base_pitch(self):
        h = self.rack.hodoscopes[0]
        base = self.rack.config.scintillator_base_m
        for left, right in zip(h.bottom_layer, h.bottom_layer[1:]):
            self.assertAlmostEqual(right.center_y_m-left.center_y_m, base/2.0, places=12)
        for left, right in zip(h.top_layer, h.top_layer[1:]):
            self.assertAlmostEqual(right.center_x_m-left.center_x_m, base/2.0, places=12)

    def test_layer_envelopes_match_detector_dimensions(self):
        cfg=self.rack.config
        base=cfg.scintillator_base_m
        self.assertAlmostEqual((cfg.top_bars+1)*base/2.0, cfg.detector_width_m, places=12)
        self.assertAlmostEqual((cfg.bottom_bars+1)*base/2.0, cfg.detector_depth_m, places=12)

if __name__ == "__main__":
    unittest.main()
