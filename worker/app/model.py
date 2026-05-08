from __future__ import annotations

import math
import re
from dataclasses import dataclass


TOKEN_RE = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class TinyTextModel:
    version: str = "tiny-text-v1"

    weights = {
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

    def predict(self, text: str) -> dict[str, str | float]:
        tokens = TOKEN_RE.findall(text.lower())
        score = sum(self.weights.get(token, 0.0) for token in tokens)
        probability = 1 / (1 + math.exp(-score))
        label = "healthy" if probability >= 0.5 else "risky"
        confidence = probability if label == "healthy" else 1 - probability

        return {
            "label": label,
            "score": round(confidence, 4),
            "model_version": self.version,
        }
