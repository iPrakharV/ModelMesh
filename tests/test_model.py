from worker.app.model import TinyTextModel


def test_tiny_model_scores_healthy_text_higher() -> None:
    model = TinyTextModel()

    healthy = model.predict("fast stable reliable")
    risky = model.predict("slow broken timeout")

    assert healthy["label"] == "healthy"
    assert risky["label"] == "risky"
    assert healthy["score"] > 0.5
    assert risky["score"] > 0.5
