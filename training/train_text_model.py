from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "worker" / "app" / "artifacts" / "tiny_text_model.json"
RUN_PATH = ROOT / "training" / "runs" / "latest.json"
RUN_MD_PATH = ROOT / "training" / "runs" / "latest.md"
TOKEN_RE = re.compile(r"[a-z']+")

HEALTHY_PATTERNS = [
    ("fast", "stable", "reliable", "inference"),
    ("healthy", "responsive", "worker", "service"),
    ("great", "latency", "stable", "route"),
    ("cache", "hit", "keeps", "service", "fast"),
    ("request", "completed", "healthy", "worker"),
    ("reliable", "model", "route", "responsive"),
]

RISKY_PATTERNS = [
    ("slow", "broken", "worker", "timeout"),
    ("request", "failed", "with", "error"),
    ("worker", "crash", "caused", "retry"),
    ("service", "timeout", "broken", "response"),
    ("model", "route", "slow"),
    ("unhealthy", "worker", "returned", "error"),
]

FILLER = [
    "batch",
    "gateway",
    "prediction",
    "queue",
    "client",
    "traffic",
    "burst",
    "mesh",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--samples-per-class", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--run-output", type=Path, default=RUN_PATH)
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


def build_texts(samples_per_class: int, seed: int = 17) -> tuple[list[str], list[float]]:
    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[float] = []

    for label, patterns in [(1.0, HEALTHY_PATTERNS), (0.0, RISKY_PATTERNS)]:
        for index in range(samples_per_class):
            pattern = list(patterns[index % len(patterns)])
            rng.shuffle(pattern)
            if index % 3 == 0:
                pattern.append(rng.choice(FILLER))
            if index % 5 == 0:
                pattern.insert(0, rng.choice(FILLER))
            texts.append(" ".join(pattern))
            labels.append(label)

    order = list(range(len(texts)))
    rng.shuffle(order)
    return [texts[index] for index in order], [labels[index] for index in order]


def build_dataset(samples_per_class: int) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    texts, labels = build_texts(samples_per_class)
    vocab = sorted({token for text in texts for token in tokens(text)})
    index = {token: pos for pos, token in enumerate(vocab)}
    rows: list[list[float]] = []

    for text in texts:
        row = [0.0] * len(vocab)
        for token in tokens(text):
            row[index[token]] += 1.0
        rows.append(row)

    return vocab, torch.tensor(rows, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32).view(-1, 1)


def split_dataset(x: torch.Tensor, y: torch.Tensor, train_fraction: float = 0.8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_count = int(len(x) * train_fraction)
    return x[:train_count], y[:train_count], x[train_count:], y[train_count:]


def accuracy(model: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        predictions = (torch.sigmoid(model(x)) >= 0.5).float()
        return float((predictions == y).float().mean().cpu())


def sync_if_needed(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def train(
    *,
    epochs: int,
    lr: float,
    samples_per_class: int,
    device: torch.device,
) -> tuple[list[str], nn.Linear, dict[str, float | int | str]]:
    torch.manual_seed(17)
    vocab, x_cpu, y_cpu = build_dataset(samples_per_class)
    x_train_cpu, y_train_cpu, x_test_cpu, y_test_cpu = split_dataset(x_cpu, y_cpu)
    x_train = x_train_cpu.to(device)
    y_train = y_train_cpu.to(device)
    x_test = x_test_cpu.to(device)
    y_test = y_test_cpu.to(device)
    model = nn.Linear(len(vocab), 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    sync_if_needed(device)
    started = time.perf_counter()
    final_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    sync_if_needed(device)
    train_seconds = time.perf_counter() - started

    stats = {
        "device": str(device),
        "epochs": epochs,
        "learning_rate": lr,
        "samples_per_class": samples_per_class,
        "train_examples": int(len(x_train_cpu)),
        "test_examples": int(len(x_test_cpu)),
        "vocab_size": len(vocab),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_seconds": round(train_seconds, 4),
        "examples_per_second": round((len(x_train_cpu) * epochs) / train_seconds, 2),
        "final_loss": round(final_loss, 6),
        "train_accuracy": round(accuracy(model, x_train, y_train), 4),
        "test_accuracy": round(accuracy(model, x_test, y_test), 4),
    }
    return vocab, model.cpu(), stats


def write_artifact(path: Path, vocab: list[str], model: nn.Linear, stats: dict[str, float | int | str]) -> None:
    weights = model.weight.detach().reshape(-1).tolist()
    artifact = {
        "model_version": "tiny-text-mps-v1",
        "training": stats,
        "weights": {token: round(float(weights[index]), 6) for index, token in enumerate(vocab)},
        "bias": round(float(model.bias.detach()[0]), 6),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n")


def write_run_files(json_path: Path, md_path: Path, stats: dict[str, float | int | str], model_path: Path) -> None:
    try:
        display_model_path = str(model_path.relative_to(ROOT))
    except ValueError:
        display_model_path = str(model_path)
    payload = {
        "model_path": display_model_path,
        **stats,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path.write_text(
        "\n".join([
            "# Training run",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Device | {stats['device']} |",
            f"| Epochs | {stats['epochs']} |",
            f"| Train examples | {stats['train_examples']} |",
            f"| Test examples | {stats['test_examples']} |",
            f"| Vocab size | {stats['vocab_size']} |",
            f"| Parameters | {stats['parameter_count']} |",
            f"| Train time | {stats['train_seconds']} s |",
            f"| Examples/sec | {stats['examples_per_second']} |",
            f"| Final loss | {stats['final_loss']} |",
            f"| Train accuracy | {stats['train_accuracy']} |",
            f"| Test accuracy | {stats['test_accuracy']} |",
            "",
        ])
    )


def main() -> None:
    args = parse_args()
    device = choose_device(args.allow_cpu)
    vocab, model, stats = train(
        epochs=args.epochs,
        lr=args.lr,
        samples_per_class=args.samples_per_class,
        device=device,
    )
    write_artifact(args.output, vocab, model, stats)
    write_run_files(args.run_output, args.run_output.with_suffix(".md"), stats, args.output)
    print(json.dumps({"output": str(args.output), **stats}, indent=2))


if __name__ == "__main__":
    main()
