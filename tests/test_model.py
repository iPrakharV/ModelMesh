import json

from worker.app.model import TinyTextModel, load_model


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
