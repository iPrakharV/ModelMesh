# Model card

## Model

`tiny-text-mps-v2` is a small MLP text classifier used by ModelMesh workers. It classifies short service-status text as either `healthy` or `risky`.

The model uses word-count features and character n-gram features. It trains with PyTorch, then exports to JSON so workers can run inference without PyTorch.

The model is intentionally small. Its purpose is to make the training, export, evaluation, and serving path testable inside an inference mesh, not to compete with production NLP models.

## Training

Training script:

```bash
python training/train_text_model.py
```

Latest checked-in run:

| Metric | Value |
| --- | ---: |
| Model type | mlp_text_classifier |
| Device | mps |
| Epochs | 260 |
| Train examples | 3,276 |
| Test examples | 820 |
| Features | 140 |
| Hidden units | 16 |
| Parameters | 2,273 |
| Train time | 0.3991 s |
| Examples/sec | 2,134,408.08 |
| Train accuracy | 1.0 |
| Test accuracy | 1.0 |

Full run output: `training/runs/latest.md`.

## Evaluation

Challenge evaluation script:

```bash
python training/evaluate_text_model.py
```

The challenge set is a small handwritten set of service-status examples with mixed positive and negative signals.

| Metric | Value |
| --- | ---: |
| Examples | 16 |
| Correct | 16 |
| Accuracy | 1.0 |

Confusion matrix:

| Expected | Predicted healthy | Predicted risky |
| --- | ---: | ---: |
| healthy | 7 | 0 |
| risky | 0 | 9 |

There were no misses on this challenge set. That is useful as a regression check, but it is still a small handwritten set and should not be treated as broad NLP evidence.

Full evaluation output: `training/evaluation/latest.md`.

## Intended use

Use this model to exercise ModelMesh serving behavior:

- model artifact loading
- worker predictions
- cache behavior
- routing behavior
- failure replay
- benchmark runs

## Limitations

- Training data is generated service-status text, not a real production incident dataset.
- The model uses word and character n-gram features, not a transformer or sequence model.
- Accuracy numbers are useful for regression checks, not for claiming broad NLP performance.
- The model is deliberately small so the project can focus on ML systems behavior.

## Next model improvement

The next useful upgrade is a larger labeled service-log dataset or a compact sequence model, still with MPS timing, evaluation output, and JSON artifact export.
