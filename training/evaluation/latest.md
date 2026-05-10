# Model evaluation

Model version: `tiny-text-mps-v1`

| Metric | Value |
| --- | ---: |
| Examples | 16 |
| Correct | 15 |
| Accuracy | 0.9375 |

## Confusion matrix

| Expected | Predicted healthy | Predicted risky |
| --- | ---: | ---: |
| healthy | 6 | 1 |
| risky | 0 | 9 |

## Label metrics

| Label | Precision | Recall |
| --- | ---: | ---: |
| healthy | 1.0 | 0.8571 |
| risky | 0.9 | 1.0 |

## Misses

| Text | Expected | Predicted | Score |
| --- | --- | --- | ---: |
| worker recovered after timeout and is stable | healthy | risky | 0.6689 |
