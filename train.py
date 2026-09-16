import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from data.synthetic import SyntheticCosmicDataset
from models.event_classifier import EventClassifier


def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("WarpTrack starter currently requires a CUDA-capable GPU.")

    device = torch.device("cuda")

    dataset = SyntheticCosmicDataset(
        num_events=12000,
        min_hits=6,
        max_hits=64,
        seed=12345,
    )

    train_set, val_set = random_split(
        dataset,
        [10000, 2000],
        generator=torch.Generator().manual_seed(123),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    model = EventClassifier().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 11):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_events = 0

        for batch in train_loader:
            hits = batch["hits"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(hits, mask)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_events += batch_size

        train_loss = total_loss / total_events
        train_acc = total_correct / total_events

        model.eval()

        total_correct = 0
        total_events = 0

        with torch.no_grad():
            for batch in val_loader:
                hits = batch["hits"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(hits, mask)

                total_correct += (logits.argmax(dim=1) == labels).sum().item()

                total_events += labels.size(0)

        val_acc = total_correct / total_events

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    torch.save(
        model.state_dict(),
        "warptrack_model.pt",
    )

    print("Saved warptrack_model.pt")


if __name__ == "__main__":
    main()
