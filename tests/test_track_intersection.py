import math
import unittest

from data.detector_geometry import GeometryConfig, ScintillatorBar, make_rack_geometry
from data.particle_track import ParticleTrack
from data.track_intersection import intersect_track_bar, intersect_track_rack


class TrackBarIntersectionTests(unittest.TestCase):
    def make_bar(self, *, orientation="x", bar_id=0):
        return ScintillatorBar(
            channel_id=7,
            hodoscope_id=1,
            layer_id=0,
            bar_id=bar_id,
            center_x_m=0.0,
            center_y_m=0.0,
            center_z_m=0.0,
            length_m=1.0,
            triangle_base_m=0.2,
            triangle_height_m=0.1,
            orientation=orientation,
        )

    def test_vertical_track_through_triangle_center_hits(self):
        track = ParticleTrack((0.0, 0.0, 1.0), (0.0, 0.0, -1.0))
        hit = intersect_track_bar(track, self.make_bar())
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.path_length_m, 0.1)
        self.assertEqual(hit.channel_id, 7)

    def test_vertical_track_outside_triangle_misses(self):
        track = ParticleTrack((0.0, 0.2, 1.0), (0.0, 0.0, -1.0))
        self.assertIsNone(intersect_track_bar(track, self.make_bar()))

    def test_track_outside_long_axis_misses(self):
        track = ParticleTrack((0.6, 0.0, 1.0), (0.0, 0.0, -1.0))
        self.assertIsNone(intersect_track_bar(track, self.make_bar()))

    def test_y_oriented_prism_uses_x_as_transverse_coordinate(self):
        bar = self.make_bar(orientation="y")
        hit = intersect_track_bar(
            ParticleTrack((0.0, 0.3, 1.0), (0.0, 0.0, -1.0)), bar
        )
        self.assertIsNotNone(hit)

    def test_angled_track_has_longer_path(self):
        bar = self.make_bar()
        vertical = intersect_track_bar(
            ParticleTrack((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)), bar
        )
        angled = intersect_track_bar(
            ParticleTrack((0.0, 0.0, 1.0), (0.2, 0.0, -1.0)), bar
        )
        self.assertIsNotNone(vertical)
        self.assertIsNotNone(angled)
        self.assertGreater(angled.path_length_m, vertical.path_length_m)

    def test_odd_bar_triangle_is_inverted_but_center_still_hits(self):
        hit = intersect_track_bar(
            ParticleTrack((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
            self.make_bar(bar_id=1),
        )
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.path_length_m, 0.1)


class RackIntersectionTests(unittest.TestCase):
    def test_vertical_center_track_reaches_all_three_hodoscopes(self):
        rack = make_rack_geometry((4.0, 14.0, 27.0))
        z_top = max(h.center_z_m for h in rack.hodoscopes) + 1.0
        track = ParticleTrack((0.0, 0.0, z_top), (0.0, 0.0, -1.0))
        hits = intersect_track_rack(track, rack)

        hodoscopes_hit = {hit.hodoscope_id for hit in hits}
        self.assertEqual(hodoscopes_hit, {0, 1, 2})
        self.assertTrue(all(hit.path_length_m > 0.0 for hit in hits))

    def test_forward_only_rejects_rack_behind_origin(self):
        rack = make_rack_geometry((4.0, 14.0, 27.0))
        z_below = min(h.center_z_m for h in rack.hodoscopes) - 1.0
        upward_origin_downward_track = ParticleTrack(
            (0.0, 0.0, z_below), (0.0, 0.0, -1.0)
        )
        hits = intersect_track_rack(upward_origin_downward_track, rack)
        self.assertEqual(hits, ())


if __name__ == "__main__":
    unittest.main()
