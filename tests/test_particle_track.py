import math
import unittest

from data.particle_track import ParticleTrack


class TestParticleTrack(unittest.TestCase):
    def test_direction_is_normalized(self):
        track = ParticleTrack((0, 0, 0), (3, 4, 0))
        self.assertAlmostEqual(track.dx, 0.6)
        self.assertAlmostEqual(track.dy, 0.8)
        self.assertAlmostEqual(track.dz, 0.0)

    def test_point_at_uses_distance_in_metres(self):
        track = ParticleTrack((1, 2, 3), (0, 0, 5))
        self.assertEqual(track.point_at(2.0), (1.0, 2.0, 5.0))

    def test_point_at_z_vertical_track(self):
        track = ParticleTrack((0.1, -0.2, 2.0), (0, 0, -1))
        x, y, z = track.point_at_z(0.5)
        self.assertAlmostEqual(x, 0.1)
        self.assertAlmostEqual(y, -0.2)
        self.assertAlmostEqual(z, 0.5)

    def test_point_at_z_sloped_track(self):
        track = ParticleTrack((0, 0, 0), (1, 2, 2))
        x, y, z = track.point_at_z(2.0)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 2.0)
        self.assertAlmostEqual(z, 2.0)

    def test_signed_distance_allows_forward_and_backward_projection(self):
        track = ParticleTrack((0, 0, 1), (0, 0, 1))
        self.assertAlmostEqual(track.distance_to_z(2.0), 1.0)
        self.assertAlmostEqual(track.distance_to_z(0.0), -1.0)

    def test_zero_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            ParticleTrack((0, 0, 0), (0, 0, 0))

    def test_parallel_track_cannot_intersect_constant_z_plane(self):
        track = ParticleTrack((0, 0, 0), (1, 0, 0))
        with self.assertRaises(ValueError):
            track.point_at_z(1.0)

    def test_nonfinite_values_are_rejected(self):
        with self.assertRaises(ValueError):
            ParticleTrack((math.inf, 0, 0), (0, 0, 1))


if __name__ == "__main__":
    unittest.main()
