from data.synthetic import (
    generate_muon_event,
    generate_hadronic_event,
)

import torch


def print_event(name, hits):
    print(name)
    print(f"Number of hits: {len(hits)}")
    print("Columns: x, y, z, energy")
    print(hits)
    print()


def main():
    generator = torch.Generator().manual_seed(7)

    print_event(
        "Muon-like event",
        generate_muon_event(10, generator),
    )

    print_event(
        "Hadronic shower-like event",
        generate_hadronic_event(20, generator),
    )


if __name__ == "__main__":
    main()
