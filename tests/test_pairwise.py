import unittest

import torch

import warptrack_cuda


class TestPairwiseDistance(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required",
    )
    def test_matches_pytorch(self):
        torch.manual_seed(123)

        positions = torch.randn(
            4,
            32,
            3,
            device="cuda",
            dtype=torch.float32,
        )

        actual = warptrack_cuda.pairwise_distance(
            positions
        )

        delta = (
            positions[:, :, None, :]
            - positions[:, None, :, :]
        )

        expected = torch.sum(
            delta * delta,
            dim=-1,
        )

        torch.cuda.synchronize()

        self.assertTrue(
            torch.allclose(
                actual,
                expected,
                rtol=1e-5,
                atol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
