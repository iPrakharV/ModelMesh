from training.evaluate_text_model import evaluate


def test_challenge_evaluation_has_expected_shape() -> None:
    result = evaluate()

    assert result["examples"] == 16
    assert result["correct"] >= 12
    assert set(result["confusion_matrix"]) == {"healthy", "risky"}
    assert set(result["per_label"]) == {"healthy", "risky"}
