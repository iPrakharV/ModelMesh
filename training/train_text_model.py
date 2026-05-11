from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "worker" / "app" / "artifacts" / "tiny_text_model.json"
RUN_PATH = ROOT / "training" / "runs" / "latest.json"
TOKEN_RE = re.compile(r"[a-z']+")
CHAR_NGRAM_RANGE = [3, 4]

HEALTHY_PATTERNS = [
    ("fast", "stable", "reliable", "inference"),
    ("healthy", "responsive", "worker", "service"),
    ("great", "latency", "stable", "route"),
    ("cache", "hit", "keeps", "service", "fast"),
    ("request", "completed", "healthy", "worker"),
    ("reliable", "model", "route", "responsive"),
    ("worker", "recovered", "after", "timeout", "stable"),
    ("cache", "recovered", "request", "completed"),
    ("gateway", "handled", "burst", "traffic", "stable"),
    ("retry", "succeeded", "worker", "responsive"),
]

RISKY_PATTERNS = [
    ("slow", "broken", "worker", "timeout"),
    ("request", "failed", "with", "error"),
    ("worker", "crash", "caused", "retry"),
    ("service", "timeout", "broken", "response"),
    ("model", "route", "slow"),
    ("unhealthy", "worker", "returned", "error"),
    ("cache", "hit", "worker", "failed", "timeout"),
    ("great", "latency", "retry", "caused", "timeout"),
    ("service", "healthy", "but", "slow"),
    ("gateway", "retry", "loop", "timeout"),
    ("unhealthy", "worker", "returned", "fast", "response"),
    ("prediction", "failed", "after", "worker", "crash"),
    ("mesh", "returned", "broken", "prediction", "error"),
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


class TextMLP(nn.Module):
    def __init__(self, feature_count: int, hidden_units: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(feature_count, hidden_units)
        self.output = nn.Linear(hidden_units, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.hidden(features))
        return self.output(hidden)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--samples-per-class", type=int, default=2048)
    parser.add_argument("--hidden-units", type=int, default=16)
    parser.add_argument("--max-char-ngrams", type=int, default=96)
    parser.add_argument("--model-version", default="tiny-text-mps-v2")
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


def char_ngrams(text: str, sizes: list[int]) -> list[str]:
    return [
        text[index : index + size]
        for size in sizes
        for index in range(max(len(text) - size + 1, 0))
    ]


def build_texts(samples_per_class: int, seed: int = 17) -> tuple[list[str], list[float]]:
    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[float] = []

    for label, patterns in [(1.0, HEALTHY_PATTERNS), (0.0, RISKY_PATTERNS)]:
        for index in range(samples_per_class):
            pattern = list(rng.choice(patterns))
            rng.shuffle(pattern)
            if index % 3 == 0:
                pattern.append(rng.choice(FILLER))
            if index % 5 == 0:
                pattern.insert(0, rng.choice(FILLER))
            if index % 11 == 0:
                pattern.append("live")
            texts.append(" ".join(pattern))
            labels.append(label)

    order = list(range(len(texts)))
    rng.shuffle(order)
    return [texts[index] for index in order], [labels[index] for index in order]


def build_feature_spec(texts: list[str], max_char_ngrams: int) -> dict[str, Any]:
    word_vocab = sorted({token for text in texts for token in tokens(text)})
    char_counts: Counter[str] = Counter()
    for text in texts:
        char_counts.update(char_ngrams(" ".join(tokens(text)), CHAR_NGRAM_RANGE))

    return {
        "word_vocab": word_vocab,
        "char_ngrams": [
            ngram
            for ngram, _ in char_counts.most_common(max_char_ngrams)
        ],
        "char_ngram_range": CHAR_NGRAM_RANGE,
        "token_pattern": TOKEN_RE.pattern,
        "lowercase": True,
    }


def vectorize(text: str, feature_spec: dict[str, Any]) -> list[float]:
    text_tokens = tokens(text)
    token_counts = Counter(text_tokens)
    char_counts = Counter(
        char_ngrams(" ".join(text_tokens), feature_spec["char_ngram_range"])
    )

    word_features = [
        float(token_counts.get(token, 0))
        for token in feature_spec["word_vocab"]
    ]
    char_features = [
        float(char_counts.get(ngram, 0))
        for ngram in feature_spec["char_ngrams"]
    ]
    return word_features + char_features


def build_dataset(
    samples_per_class: int,
    max_char_ngrams: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    texts, labels = build_texts(samples_per_class)
    feature_spec = build_feature_spec(texts, max_char_ngrams)
    rows = [vectorize(text, feature_spec) for text in texts]

    return (
        feature_spec,
        torch.tensor(rows, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32).view(-1, 1),
    )


def split_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    train_fraction: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_count = int(len(x) * train_fraction)
    return x[:train_count], y[:train_count], x[train_count:], y[train_count:]


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
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
    hidden_units: int,
    max_char_ngrams: int,
    device: torch.device,
) -> tuple[dict[str, Any], TextMLP, dict[str, float | int | str]]:
    torch.manual_seed(17)
    feature_spec, x_cpu, y_cpu = build_dataset(samples_per_class, max_char_ngrams)
    x_train_cpu, y_train_cpu, x_test_cpu, y_test_cpu = split_dataset(x_cpu, y_cpu)
    x_train = x_train_cpu.to(device)
    y_train = y_train_cpu.to(device)
    x_test = x_test_cpu.to(device)
    y_test = y_test_cpu.to(device)
    model = TextMLP(x_cpu.shape[1], hidden_units).to(device)
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
        "model_type": "mlp_text_classifier",
        "device": str(device),
        "epochs": epochs,
        "learning_rate": lr,
        "samples_per_class": samples_per_class,
        "train_examples": int(len(x_train_cpu)),
        "test_examples": int(len(x_test_cpu)),
        "feature_count": int(x_cpu.shape[1]),
        "word_features": len(feature_spec["word_vocab"]),
        "char_ngram_features": len(feature_spec["char_ngrams"]),
        "hidden_units": hidden_units,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_seconds": round(train_seconds, 4),
        "examples_per_second": round((len(x_train_cpu) * epochs) / train_seconds, 2),
        "final_loss": round(final_loss, 6),
        "train_accuracy": round(accuracy(model, x_train, y_train), 4),
        "test_accuracy": round(accuracy(model, x_test, y_test), 4),
    }
    return feature_spec, model.cpu(), stats


def write_artifact(
    path: Path,
    model_version: str,
    feature_spec: dict[str, Any],
    model: TextMLP,
    stats: dict[str, float | int | str],
) -> None:
    artifact = {
        "model_version": model_version,
        "model_type": "mlp_text_classifier",
        "feature_spec": feature_spec,
        "weights": {
            "hidden": rounded_matrix(model.hidden.weight.detach().tolist()),
            "output": rounded_vector(model.output.weight.detach().reshape(-1).tolist()),
        },
        "biases": {
            "hidden": rounded_vector(model.hidden.bias.detach().tolist()),
            "output": round(float(model.output.bias.detach()[0]), 6),
        },
        "training": stats,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n")


def rounded_vector(values: list[float]) -> list[float]:
    return [round(float(value), 6) for value in values]


def rounded_matrix(values: list[list[float]]) -> list[list[float]]:
    return [rounded_vector(row) for row in values]


def write_run_files(
    json_path: Path,
    md_path: Path,
    stats: dict[str, float | int | str],
    model_path: Path,
) -> None:
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
            f"| Model type | {stats['model_type']} |",
            f"| Device | {stats['device']} |",
            f"| Epochs | {stats['epochs']} |",
            f"| Train examples | {stats['train_examples']} |",
            f"| Test examples | {stats['test_examples']} |",
            f"| Features | {stats['feature_count']} |",
            f"| Hidden units | {stats['hidden_units']} |",
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
    feature_spec, model, stats = train(
        epochs=args.epochs,
        lr=args.lr,
        samples_per_class=args.samples_per_class,
        hidden_units=args.hidden_units,
        max_char_ngrams=args.max_char_ngrams,
        device=device,
    )
    write_artifact(args.output, args.model_version, feature_spec, model, stats)
    write_run_files(args.run_output, args.run_output.with_suffix(".md"), stats, args.output)
    print(json.dumps({"output": str(args.output), **stats}, indent=2))


if __name__ == "__main__":
    main()
