from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[a-z']+")
DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "tiny_text_model.json"
DEFAULT_WEIGHTS = {
    "fast": 1.4,
    "stable": 1.2,
    "great": 1.1,
    "reliable": 1.0,
    "slow": -1.3,
    "broken": -1.5,
    "error": -1.2,
    "timeout": -1.1,
    "crash": -1.4,
}


def sigmoid(score: float) -> float:
    if score >= 0:
        return 1 / (1 + math.exp(-score))
    exp_score = math.exp(score)
    return exp_score / (1 + exp_score)


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(text: str, sizes: list[int]) -> list[str]:
    return [
        text[index : index + size]
        for size in sizes
        for index in range(max(len(text) - size + 1, 0))
    ]


@dataclass(frozen=True)
class TinyTextModel:
    version: str = "tiny-text-v1"
    weights: dict[str, float] | None = None
    bias: float = 0.0
    model_type: str = "linear_bow"

    @classmethod
    def from_artifact(cls, path: Path) -> TinyTextModel:
        artifact = json.loads(path.read_text())
        return cls(
            version=artifact.get("model_version", "tiny-text-trained"),
            weights={key: float(value) for key, value in artifact["weights"].items()},
            bias=float(artifact.get("bias", 0.0)),
        )

    def predict(self, text: str) -> dict[str, str | float]:
        text_tokens = tokens(text)
        weights = self.weights or DEFAULT_WEIGHTS
        score = self.bias + sum(weights.get(token, 0.0) for token in text_tokens)
        probability = sigmoid(score)
        label = "healthy" if probability >= 0.5 else "risky"
        confidence = probability if label == "healthy" else 1 - probability

        return {
            "label": label,
            "score": round(confidence, 4),
            "model_version": self.version,
        }


@dataclass(frozen=True)
class MLPTextModel:
    version: str
    feature_spec: dict[str, Any]
    hidden_weights: list[list[float]]
    hidden_biases: list[float]
    output_weights: list[float]
    output_bias: float
    model_type: str = "mlp_text_classifier"

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> MLPTextModel:
        return cls(
            version=artifact["model_version"],
            feature_spec=artifact["feature_spec"],
            hidden_weights=artifact["weights"]["hidden"],
            hidden_biases=artifact["biases"]["hidden"],
            output_weights=artifact["weights"]["output"],
            output_bias=float(artifact["biases"]["output"]),
        )

    def predict(self, text: str) -> dict[str, str | float]:
        features = self._features(text)
        hidden = [
            max(0.0, bias + dot(row, features))
            for row, bias in zip(self.hidden_weights, self.hidden_biases)
        ]
        probability = sigmoid(self.output_bias + dot(self.output_weights, hidden))
        label = "healthy" if probability >= 0.5 else "risky"
        confidence = probability if label == "healthy" else 1 - probability

        return {
            "label": label,
            "score": round(confidence, 4),
            "model_version": self.version,
        }

    def _features(self, text: str) -> list[float]:
        text_tokens = tokens(text)
        token_counts = Counter(text_tokens)
        char_source = " ".join(text_tokens)
        char_sizes = self.feature_spec.get("char_ngram_range", [3, 4])
        char_counts = Counter(ngrams(char_source, char_sizes))

        word_features = [
            float(token_counts.get(token, 0))
            for token in self.feature_spec["word_vocab"]
        ]
        char_features = [
            float(char_counts.get(ngram, 0))
            for ngram in self.feature_spec["char_ngrams"]
        ]
        return word_features + char_features


def dot(weights: list[float], values: list[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, values))


def load_model(path: str | None = None) -> TinyTextModel | MLPTextModel:
    artifact_path = Path(path) if path else DEFAULT_ARTIFACT
    if artifact_path.exists():
        artifact = json.loads(artifact_path.read_text())
        if artifact.get("model_type") == "mlp_text_classifier":
            return MLPTextModel.from_artifact(artifact)
        return TinyTextModel.from_artifact(artifact_path)
    return TinyTextModel()
