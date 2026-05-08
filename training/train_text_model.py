from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "worker" / "app" / "artifacts" / "tiny_text_model.json"
TOKEN_RE = re.compile(r"[a-z']+")

HEALTHY = [
    "fast stable reliable inference",
    "worker is healthy and responsive",
    "great latency with stable service",
    "cache hit keeps the service fast",
    "model route is reliable",
    "request completed with healthy worker",
]

RISKY = [
    "slow broken worker timeout",
    "request failed with error",
    "worker crash caused retry",
    "service timeout and broken response",
    "model route is slow",
    "unhealthy worker returned error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def choose_device(allow_cpu: bool) -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if allow_cpu:
        return torch.device("cpu")
    raise SystemExit("MPS is not available. Pass --allow-cpu if you really want a CPU run.")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_dataset() -> tuple[list[str], torch.Tensor, torch.Tensor]:
    texts = HEALTHY + RISKY
    labels = [1.0] * len(HEALTHY) + [0.0] * len(RISKY)
    vocab = sorted({token for text in texts for token in tokens(text)})
    index = {token: pos for pos, token in enumerate(vocab)}
    rows: list[list[float]] = []

    for text in texts:
        row = [0.0] * len(vocab)
        for token in tokens(text):
            row[index[token]] += 1.0
        rows.append(row)

    return vocab, torch.tensor(rows, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32).view(-1, 1)


def train(epochs: int, lr: float, device: torch.device) -> tuple[list[str], nn.Linear, float]:
    torch.manual_seed(17)
    vocab, x_cpu, y_cpu = build_dataset()
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    model = nn.Linear(len(vocab), 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = (torch.sigmoid(model(x)) >= 0.5).float()
        accuracy = float((predictions == y).float().mean().cpu())

    return vocab, model.cpu(), accuracy


def write_artifact(path: Path, vocab: list[str], model: nn.Linear, accuracy: float, device: torch.device) -> None:
    weights = model.weight.detach().reshape(-1).tolist()
    artifact = {
        "model_version": "tiny-text-mps-v1",
        "device": str(device),
        "train_accuracy": round(accuracy, 4),
        "weights": {token: round(float(weights[index]), 6) for index, token in enumerate(vocab)},
        "bias": round(float(model.bias.detach()[0]), 6),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    device = choose_device(args.allow_cpu)
    vocab, model, accuracy = train(args.epochs, args.lr, device)
    write_artifact(args.output, vocab, model, accuracy, device)
    print({"device": str(device), "train_accuracy": round(accuracy, 4), "output": str(args.output)})


if __name__ == "__main__":
    main()
