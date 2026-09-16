import math
import unittest

from data.cosmic_ray import CosmicRayConfig, CosmicRayGenerator
from data.detector_geometry import make_rack_geometry


class CosmicRayGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.rack = make_rack_geometry((4.0, 14.0, 27.0))

    def test_angles_are_downward_and_within_cut(self):
        cfg = CosmicRayConfig(max_zenith_deg=60.0, min_hodoscopes_hit=1)
        gen = CosmicRayGenerator(self.rack, cfg, seed=123)
        for _ in range(1000):
            theta, phi = gen.sample_angles()
            self.assertGreaterEqual(theta, 0.0)
            self.assertLessEqual(theta, math.radians(60.0))
            self.assertGreaterEqual(phi, 0.0)
            self.assertLess(phi, 2.0 * math.pi)

    def test_sample_track_has_unit_downward_direction(self):
        gen = CosmicRayGenerator(
            self.rack, CosmicRayConfig(min_hodoscopes_hit=1), seed=2
        )
        track, _, _ = gen.sample_track()
        norm = math.sqrt(sum(x * x for x in track.direction))
        self.assertAlmostEqual(norm, 1.0)
        self.assertLess(track.dz, 0.0)

    def test_phi_covers_all_quadrants(self):
        gen = CosmicRayGenerator(
            self.rack, CosmicRayConfig(min_hodoscopes_hit=1), seed=9
        )
        quadrants = set()
        for _ in range(500):
            _, phi = gen.sample_angles()
            quadrants.add(int(phi / (math.pi / 2.0)) % 4)
        self.assertEqual(quadrants, {0, 1, 2, 3})

    def test_cosine_distribution_is_not_isotropic(self):
        # For n=2 on a horizontal crossing plane, p(mu)~mu^3.  With the
        # default 75-degree truncation the mean mu should be comfortably high.
        gen = CosmicRayGenerator(
            self.rack, CosmicRayConfig(min_hodoscopes_hit=1), seed=44
        )
        mus = [math.cos(gen.sample_angles()[0]) for _ in range(10000)]
        self.assertGreater(sum(mus) / len(mus), 0.75)

    def test_generated_event_hits_required_hodoscopes(self):
        cfg = CosmicRayConfig(
            max_zenith_deg=45.0,
            generation_margin_m=0.0,
            min_hodoscopes_hit=3,
            max_attempts=20000,
        )
        gen = CosmicRayGenerator(self.rack, cfg, seed=17)
        event = gen.generate_event()
        self.assertGreaterEqual(len(event.hodoscopes_hit), 3)
        self.assertGreater(len(event.hits), 0)

    def test_event_is_reproducible_with_seed(self):
        cfg = CosmicRayConfig(min_hodoscopes_hit=1)
        a = CosmicRayGenerator(self.rack, cfg, seed=101).generate_event()
        b = CosmicRayGenerator(self.rack, cfg, seed=101).generate_event()
        self.assertEqual(a.track.origin_m, b.track.origin_m)
        self.assertEqual(a.track.direction, b.track.direction)
        self.assertEqual([h.channel_id for h in a.hits], [h.channel_id for h in b.hits])

    def test_rejects_impossible_trigger(self):
        with self.assertRaises(ValueError):
            CosmicRayGenerator(
                self.rack,
                CosmicRayConfig(min_hodoscopes_hit=4),
            )


if __name__ == "__main__":
    unittest.main()
