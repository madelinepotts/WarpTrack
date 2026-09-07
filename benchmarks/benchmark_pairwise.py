import torch

import warptrack_cuda


def benchmark(fn, warmup=20, iterations=100):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(iterations):
        fn()

    stop.record()
    torch.cuda.synchronize()

    return start.elapsed_time(stop) / iterations


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    print(
        f"{'Hits':>8} "
        f"{'Custom CUDA (ms)':>18} "
        f"{'PyTorch (ms)':>14}"
    )

    for num_hits in [16, 32, 64, 128, 256, 512]:
        positions = torch.randn(
            32,
            num_hits,
            3,
            device="cuda",
            dtype=torch.float32,
        )

        custom_ms = benchmark(
            lambda: warptrack_cuda.pairwise_distance(
                positions
            )
        )

        pytorch_ms = benchmark(
            lambda: torch.sum(
                (
                    positions[:, :, None, :]
                    - positions[:, None, :, :]
                ) ** 2,
                dim=-1,
            )
        )

        print(
            f"{num_hits:8d} "
            f"{custom_ms:18.4f} "
            f"{pytorch_ms:14.4f}"
        )


if __name__ == "__main__":
    main()
