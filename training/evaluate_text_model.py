from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_JSON = ROOT / "training" / "evaluation" / "latest.json"
OUTPUT_MD = ROOT / "training" / "evaluation" / "latest.md"
LABELS = ("healthy", "risky")


@dataclass(frozen=True)
class ChallengeExample:
    expected: str
    text: str


CHALLENGE_SET = [
    ChallengeExample("healthy", "fast stable reliable service"),
    ChallengeExample("risky", "slow broken timeout error"),
    ChallengeExample("healthy", "worker recovered after timeout and is stable"),
    ChallengeExample("risky", "cache hit but worker failed with timeout"),
    ChallengeExample("risky", "great latency but retry caused timeout"),
    ChallengeExample("healthy", "responsive route completed request"),
    ChallengeExample("risky", "unhealthy worker returned fast response"),
    ChallengeExample("healthy", "stable gateway handled burst traffic"),
    ChallengeExample("risky", "prediction failed after worker crash"),
    ChallengeExample("risky", "service is healthy but slow"),
    ChallengeExample("healthy", "reliable cache and responsive worker"),
    ChallengeExample("risky", "broken route recovered with retry"),
    ChallengeExample("healthy", "cache recovered and request completed"),
    ChallengeExample("risky", "gateway retry loop after timeout"),
    ChallengeExample("healthy", "responsive worker handled traffic burst"),
    ChallengeExample("risky", "mesh returned broken prediction error"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifact", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=OUTPUT_MD)
    return parser.parse_args()


def empty_confusion_matrix() -> dict[str, dict[str, int]]:
    return {expected: {predicted: 0 for predicted in LABELS} for expected in LABELS}


def score_example(model: Any, example: ChallengeExample) -> dict[str, Any]:
    prediction = model.predict(example.text)
    predicted = str(prediction["label"])
    return {
        "text": example.text,
        "expected": example.expected,
        "predicted": predicted,
        "score": prediction["score"],
        "correct": example.expected == predicted,
    }


def build_confusion_matrix(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    confusion = empty_confusion_matrix()
    for row in rows:
        confusion[row["expected"]][row["predicted"]] += 1
    return confusion


def safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0


def label_metrics(confusion: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    per_label: dict[str, dict[str, float]] = {}
    for label in LABELS:
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[other][label] for other in LABELS if other != label
        )
        false_negative = sum(
            confusion[label][other] for other in LABELS if other != label
        )
        precision_base = true_positive + false_positive
        recall_base = true_positive + false_negative
        per_label[label] = {
            "precision": safe_rate(true_positive, precision_base),
            "recall": safe_rate(true_positive, recall_base),
        }
    return per_label


def evaluate(model_artifact: Path | None = None) -> dict[str, Any]:
    from worker.app.model import load_model

    model = load_model(str(model_artifact) if model_artifact else None)
    rows = [score_example(model, example) for example in CHALLENGE_SET]
    confusion = build_confusion_matrix(rows)
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])

    return {
        "model_version": model.version,
        "dataset": "handwritten service-status challenge set",
        "examples": total,
        "accuracy": round(correct / total, 4),
        "correct": correct,
        "confusion_matrix": confusion,
        "per_label": label_metrics(confusion),
        "rows": rows,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Model evaluation",
        "",
        f"Model version: `{result['model_version']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Examples | {result['examples']} |",
        f"| Correct | {result['correct']} |",
        f"| Accuracy | {result['accuracy']} |",
        "",
        "## Confusion matrix",
        "",
        "| Expected | Predicted healthy | Predicted risky |",
        "| --- | ---: | ---: |",
    ]

    for expected, predictions in result["confusion_matrix"].items():
        lines.append(f"| {expected} | {predictions['healthy']} | {predictions['risky']} |")

    lines.extend([
        "",
        "## Label metrics",
        "",
        "| Label | Precision | Recall |",
        "| --- | ---: | ---: |",
    ])
    for label, metrics in result["per_label"].items():
        lines.append(f"| {label} | {metrics['precision']} | {metrics['recall']} |")

    lines.extend([
        "",
        "## Misses",
        "",
    ])
    misses = [row for row in result["rows"] if not row["correct"]]
    if not misses:
        lines.append("No misses on this challenge set.")
    else:
        lines.extend([
            "| Text | Expected | Predicted | Score |",
            "| --- | --- | --- | ---: |",
        ])
        for row in misses:
            lines.append(
                f"| {row['text']} | {row['expected']} | "
                f"{row['predicted']} | {row['score']} |"
            )

    lines.append("")
    return "\n".join(lines)


def write_outputs(result: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    md_path.write_text(render_markdown(result))


def main() -> None:
    args = parse_args()
    result = evaluate(args.model_artifact)
    write_outputs(result, args.output_json, args.output_md)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
