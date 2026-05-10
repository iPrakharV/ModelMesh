from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUT_JSON = ROOT / "training" / "evaluation" / "latest.json"
OUTPUT_MD = ROOT / "training" / "evaluation" / "latest.md"

CHALLENGE_SET = [
    ("healthy", "fast stable reliable service"),
    ("risky", "slow broken timeout error"),
    ("healthy", "worker recovered after timeout and is stable"),
    ("risky", "cache hit but worker failed with timeout"),
    ("risky", "great latency but retry caused timeout"),
    ("healthy", "responsive route completed request"),
    ("risky", "unhealthy worker returned fast response"),
    ("healthy", "stable gateway handled burst traffic"),
    ("risky", "prediction failed after worker crash"),
    ("risky", "service is healthy but slow"),
    ("healthy", "reliable cache and responsive worker"),
    ("risky", "broken route recovered with retry"),
    ("healthy", "cache recovered and request completed"),
    ("risky", "gateway retry loop after timeout"),
    ("healthy", "responsive worker handled traffic burst"),
    ("risky", "mesh returned broken prediction error"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifact", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=OUTPUT_MD)
    return parser.parse_args()


def evaluate(model_artifact: Path | None = None) -> dict:
    from worker.app.model import load_model

    model = load_model(str(model_artifact) if model_artifact else None)
    rows = []
    labels = ["healthy", "risky"]
    confusion = {
        "healthy": {"healthy": 0, "risky": 0},
        "risky": {"healthy": 0, "risky": 0},
    }

    for expected, text in CHALLENGE_SET:
        prediction = model.predict(text)
        predicted = str(prediction["label"])
        confusion[expected][predicted] += 1
        rows.append({
            "text": text,
            "expected": expected,
            "predicted": predicted,
            "score": prediction["score"],
            "correct": expected == predicted,
        })

    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    per_label = {}
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[other][label] for other in labels if other != label
        )
        false_negative = sum(
            confusion[label][other] for other in labels if other != label
        )
        precision_base = true_positive + false_positive
        recall_base = true_positive + false_negative
        precision = true_positive / precision_base if precision_base else 0
        recall = true_positive / recall_base if recall_base else 0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }

    return {
        "model_version": model.version,
        "dataset": "handwritten service-status challenge set",
        "examples": total,
        "accuracy": round(correct / total, 4),
        "correct": correct,
        "confusion_matrix": confusion,
        "per_label": per_label,
        "rows": rows,
    }


def write_outputs(result: dict, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n")

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
    md_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    result = evaluate(args.model_artifact)
    write_outputs(result, args.output_json, args.output_md)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
