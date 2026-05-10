# Model card

## Model

`tiny-text-mps-v1` is a small linear text classifier used by ModelMesh workers. It classifies short service-status text as either `healthy` or `risky`.

The model is intentionally small. Its purpose is to make the training and serving path testable inside an inference mesh, not to compete with production NLP models.

## Training

Training script:

```bash
python training/train_text_model.py
```

Latest checked-in run:

| Metric | Value |
| --- | ---: |
| Device | mps |
| Epochs | 320 |
| Train examples | 3,276 |
| Test examples | 820 |
| Vocab size | 37 |
| Parameters | 38 |
| Train time | 0.4351 s |
| Examples/sec | 2,409,512.37 |
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
| Correct | 15 |
| Accuracy | 0.9375 |

Confusion matrix:

| Expected | Predicted healthy | Predicted risky |
| --- | ---: | ---: |
| healthy | 6 | 1 |
| risky | 0 | 9 |

The missed example was:

```text
worker recovered after timeout and is stable
```

The model predicted `risky`, mostly because `timeout` has a strong negative learned weight and the bag-of-words model does not understand recovery order.

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
- The model is bag-of-words and does not understand word order.
- Accuracy numbers are useful for regression checks, not for claiming broad NLP performance.
- The model is deliberately small so the project can focus on ML systems behavior.

## Next model improvement

The next useful upgrade is a compact sequence model or character n-gram model trained on a larger labeled service-log dataset, still with MPS timing and artifact export.
