from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class TinyTextModel:
    version: str = "tiny-text-v1"
    weights: dict[str, float] | None = None
    bias: float = 0.0

    @classmethod
    def from_artifact(cls, path: Path) -> "TinyTextModel":
        artifact = json.loads(path.read_text())
        return cls(
            version=artifact.get("model_version", "tiny-text-trained"),
            weights={key: float(value) for key, value in artifact["weights"].items()},
            bias=float(artifact.get("bias", 0.0)),
        )

    def predict(self, text: str) -> dict[str, str | float]:
        tokens = TOKEN_RE.findall(text.lower())
        weights = self.weights or DEFAULT_WEIGHTS
        score = self.bias + sum(weights.get(token, 0.0) for token in tokens)
        probability = 1 / (1 + math.exp(-score))
        label = "healthy" if probability >= 0.5 else "risky"
        confidence = probability if label == "healthy" else 1 - probability

        return {
            "label": label,
            "score": round(confidence, 4),
            "model_version": self.version,
        }


def load_model(path: str | None = None) -> TinyTextModel:
    artifact_path = Path(path) if path else DEFAULT_ARTIFACT
    if artifact_path.exists():
        return TinyTextModel.from_artifact(artifact_path)
    return TinyTextModel()
