# Sprint 4 Reply Generation Comparison

This table summarizes the three reply-generation strategies evaluated on the synthetic technical-support benchmark built from the project KB scenarios.

## Aggregate Results

| Method | Model | Retrieval | Fine-tuning | Test Samples | BERTScore Precision | BERTScore Recall | BERTScore F1 |
|---|---|---|---|---:|---:|---:|---:|
| Baseline reply generation | `google/flan-t5-small` | No | No | 15 | 0.8710 | 0.8525 | 0.8615 |
| Retrieval-grounded reply generation | `google/flan-t5-small` | Yes | No | 15 | 0.8678 | 0.8488 | 0.8580 |
| LoRA reply generation | `google/flan-t5-small` | No | Yes (`LoRA`) | 15 | 0.8685 | 0.8569 | 0.8626 |

## Sample-Level Wins

| Method | Wins |
|---|---:|
| Baseline | 4 |
| Retrieval | 4 |
| LoRA | 6 |
| Ties | 1 |

## Interpretation

- The plain `flan-t5-small` baseline already provides a reasonable starting point for reply generation.
- Retrieval remained competitive, but it did not clearly outperform the baseline in the current configuration.
- LoRA fine-tuning achieved the best overall BERTScore F1 and the highest number of sample-level wins, making it the strongest reply-generation approach tested in Sprint 4.
