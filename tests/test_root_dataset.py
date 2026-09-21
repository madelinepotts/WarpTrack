import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.root_dataset import (
    MUON_CLASS,
    ELECTRON_CLASS,
    PHOTON_CLASS,
    PROTON_CLASS,
    NEUTRON_CLASS,
    UNKNOWN_PARTICLE_CLASS,
    BAR_TRIGGER_THRESHOLD_MEV,
    MIN_TRIGGER_BARS,
    aggregate_hits,
    channel_count_from_geometry,
    event_passes_trigger,
    particle_class_from_pdgs,
    reconstruct_bar_hits,
    trigger_bar_count,
)


class TestRootDatasetHelpers(unittest.TestCase):
    def test_channel_count_comes_from_geometry(self):
        definition = {
            "hodoscope": {"channels_per_hodoscope": 25},
            "instances": [{"id": 4}, {"id": 9}, {"id": 12}, {"id": 20}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "geometry.json"
            path.write_text(json.dumps(definition), encoding="utf-8")
            self.assertEqual(channel_count_from_geometry(path), 525)

    def test_step_hits_are_aggregated_per_channel(self):
        edep, time, hit = aggregate_hits(
            channel_ids=[2, 2, 5],
            edep_mev=[1.25, 0.75, 3.0],
            time_ns=[12.0, 10.0, 15.0],
            n_channels=8,
        )
        self.assertAlmostEqual(float(edep[2]), 2.0)
        self.assertAlmostEqual(float(edep[5]), 3.0)
        self.assertAlmostEqual(float(time[2]), 0.0)
        self.assertAlmostEqual(float(time[5]), 5.0)
        self.assertTrue(bool(hit[2]))
        self.assertTrue(bool(hit[5]))
        self.assertEqual(int(hit.sum()), 2)

    def test_empty_event_is_valid(self):
        edep, time, hit = aggregate_hits([], [], [], 5)
        np.testing.assert_array_equal(edep, np.zeros(5, dtype=np.float32))
        np.testing.assert_array_equal(time, np.zeros(5, dtype=np.float32))
        self.assertFalse(hit.any())

    def test_invalid_channel_is_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_hits([7], [1.0], [0.0], 7)


    def test_bar_reconstruction_sums_energy_and_energy_weights_position(self):
        rng = np.random.default_rng(7)
        hits, channels = reconstruct_bar_hits(
            channel_ids=[2, 2, 5],
            edep_mev=[1.0, 3.0, 2.0],
            time_ns=[12.0, 10.0, 15.0],
            x_mm=[0.0, 20.0, 100.0],
            y_mm=[0.0, 0.0, 50.0],
            z_mm=[10.0, 10.0, 20.0],
            position_resolution_mm=0.0,
            rng=rng,
        )
        np.testing.assert_array_equal(channels, [2, 5])
        self.assertAlmostEqual(float(hits[0, 0]), 15.0)
        self.assertAlmostEqual(float(hits[0, 3]), 4.0)
        self.assertAlmostEqual(float(hits[0, 4]), 0.0)
        self.assertAlmostEqual(float(hits[1, 3]), 2.0)
        self.assertAlmostEqual(float(hits[1, 4]), 5.0)

    def test_bar_reconstruction_applies_position_resolution_only(self):
        unsmeared, _ = reconstruct_bar_hits(
            [1], [2.5], [3.0], [4.0], [5.0], [6.0],
            position_resolution_mm=0.0, rng=np.random.default_rng(1),
        )
        smeared, _ = reconstruct_bar_hits(
            [1], [2.5], [3.0], [4.0], [5.0], [6.0],
            position_resolution_mm=10.0, rng=np.random.default_rng(1),
        )
        self.assertAlmostEqual(float(smeared[0, 3]), 2.5)
        self.assertAlmostEqual(float(smeared[0, 4]), 0.0)
        self.assertFalse(np.allclose(unsmeared[0, :3], smeared[0, :3]))


    def test_reconstruction_same_seed_is_reproducible(self):
        a, _ = reconstruct_bar_hits(
            [4], [1.0], [2.0], [3.0], [4.0], [5.0],
            position_resolution_mm=10.0, rng=np.random.default_rng(123),
        )
        b, _ = reconstruct_bar_hits(
            [4], [1.0], [2.0], [3.0], [4.0], [5.0],
            position_resolution_mm=10.0, rng=np.random.default_rng(123),
        )
        np.testing.assert_array_equal(a, b)

    def test_trigger_requires_four_distinct_bars_at_threshold(self):
        self.assertEqual(MIN_TRIGGER_BARS, 4)
        self.assertAlmostEqual(BAR_TRIGGER_THRESHOLD_MEV, 0.5)
        self.assertTrue(event_passes_trigger(
            [1, 2, 3, 4], [0.5, 0.5, 0.5, 0.5]
        ))
        self.assertFalse(event_passes_trigger(
            [1, 2, 3], [10.0, 10.0, 10.0]
        ))

    def test_trigger_sums_steps_before_threshold_and_counts_unique_bars(self):
        channels = [1, 1, 2, 3, 4, 5]
        energies = [0.2, 0.3, 0.6, 0.7, 0.5, 0.49]
        self.assertEqual(trigger_bar_count(channels, energies), 4)
        self.assertTrue(event_passes_trigger(channels, energies))

    def test_subthreshold_bars_do_not_count_toward_trigger(self):
        self.assertEqual(
            trigger_bar_count([1, 2, 3, 4], [0.5, 0.5, 0.5, 0.499]),
            3,
        )
        self.assertFalse(
            event_passes_trigger([1, 2, 3, 4], [0.5, 0.5, 0.5, 0.499])
        )

    def test_particle_class_maps_both_muon_charges_to_muon(self):
        self.assertEqual(particle_class_from_pdgs([13]), MUON_CLASS)
        self.assertEqual(particle_class_from_pdgs([-13]), MUON_CLASS)

    def test_particle_class_maps_electron_charges(self):
        self.assertEqual(particle_class_from_pdgs([11]), ELECTRON_CLASS)
        self.assertEqual(particle_class_from_pdgs([-11]), ELECTRON_CLASS)

    def test_particle_class_maps_photon(self):
        self.assertEqual(particle_class_from_pdgs([22]), PHOTON_CLASS)

    def test_particle_class_maps_proton(self):
        self.assertEqual(particle_class_from_pdgs([2212]), PROTON_CLASS)

    def test_particle_class_maps_neutron(self):
        self.assertEqual(particle_class_from_pdgs([2112]), NEUTRON_CLASS)

    def test_particle_class_accepts_same_family_multi_primary_showers(self):
        self.assertEqual(particle_class_from_pdgs([13, -13]), MUON_CLASS)
        self.assertEqual(particle_class_from_pdgs([22, 22]), PHOTON_CLASS)
        self.assertEqual(particle_class_from_pdgs([11, -11]), ELECTRON_CLASS)

    def test_particle_class_rejects_mixed_or_unsupported_events(self):
        self.assertEqual(particle_class_from_pdgs([13, 2212]), UNKNOWN_PARTICLE_CLASS)
        self.assertEqual(particle_class_from_pdgs([13, 22]), UNKNOWN_PARTICLE_CLASS)
        self.assertEqual(particle_class_from_pdgs([211]), UNKNOWN_PARTICLE_CLASS)
        self.assertEqual(particle_class_from_pdgs([]), UNKNOWN_PARTICLE_CLASS)


if __name__ == "__main__":
    unittest.main()
