import unittest

from data.synthetic import SyntheticCosmicDataset


class TestSyntheticDataset(unittest.TestCase):
    def test_shapes(self):
        dataset = SyntheticCosmicDataset(
            num_events=10,
            min_hits=4,
            max_hits=32,
        )

        event = dataset[0]

        self.assertEqual(
            tuple(event["hits"].shape),
            (32, 4),
        )

        self.assertEqual(
            tuple(event["mask"].shape),
            (32,),
        )

        self.assertGreaterEqual(
            event["mask"].sum().item(),
            4,
        )

        self.assertIn(
            event["label"].item(),
            [0, 1],
        )


if __name__ == "__main__":
    unittest.main()
