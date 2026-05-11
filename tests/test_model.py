import json

from worker.app.model import DEFAULT_ARTIFACT, MLPTextModel, TinyTextModel, load_model


def test_tiny_model_scores_healthy_text_higher() -> None:
    model = TinyTextModel()

    healthy = model.predict("fast stable reliable")
    risky = model.predict("slow broken timeout")

    assert healthy["label"] == "healthy"
    assert risky["label"] == "risky"
    assert healthy["score"] > 0.5
    assert risky["score"] > 0.5


def test_model_can_load_training_artifact(tmp_path) -> None:
    artifact = tmp_path / "model.json"
    artifact.write_text(json.dumps({
        "model_version": "test-model",
        "weights": {"stable": 2.0, "broken": -2.0},
        "bias": 0.0,
    }))

    model = load_model(str(artifact))

    assert model.predict("stable")["label"] == "healthy"
    assert model.predict("broken")["label"] == "risky"
    assert model.version == "test-model"


def test_model_can_load_v2_mlp_artifact(tmp_path) -> None:
    artifact = tmp_path / "model.json"
    artifact.write_text(json.dumps({
        "model_version": "test-mlp",
        "model_type": "mlp_text_classifier",
        "feature_spec": {
            "word_vocab": ["stable", "broken"],
            "char_ngrams": [],
            "char_ngram_range": [3, 4],
        },
        "weights": {
            "hidden": [[1.0, 0.0], [0.0, 1.0]],
            "output": [2.0, -2.0],
        },
        "biases": {
            "hidden": [0.0, 0.0],
            "output": 0.0,
        },
    }))

    model = load_model(str(artifact))

    assert isinstance(model, MLPTextModel)
    assert model.predict("stable")["label"] == "healthy"
    assert model.predict("broken")["label"] == "risky"
    assert model.version == "test-mlp"


def test_checked_in_artifact_uses_v2_schema() -> None:
    artifact = json.loads(DEFAULT_ARTIFACT.read_text())

    assert artifact["model_version"] == "tiny-text-mps-v2"
    assert artifact["model_type"] == "mlp_text_classifier"
    assert set(artifact) >= {"feature_spec", "weights", "biases", "training"}
    assert set(artifact["weights"]) == {"hidden", "output"}
    assert set(artifact["biases"]) == {"hidden", "output"}
