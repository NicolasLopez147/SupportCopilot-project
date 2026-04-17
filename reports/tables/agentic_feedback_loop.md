# Sprint 6: Agentic Feedback Loop

## Goal

Validate whether a critic-driven offline improvement loop can improve reply generation beyond the original LoRA baseline.

## Architecture

- end-to-end SupportCopilot pipeline
- `intent_critic`
- `summary_critic`
- `reply_critic`
- fallback layer for unsafe or low-quality outputs
- failure memory in `data/feedback/*.jsonl`
- retraining candidate builder
- augmented training-set builder
- incremental LoRA reply retraining

## Final Reply Comparison

| Method | Samples | BERTScore F1 | Wins | Notes |
|---|---:|---:|---:|---|
| Baseline T5 | 15 | 0.8615 | 1 | Plain generation from conversation only |
| Retrieval + T5 | 15 | 0.8580 | 2 | Uses KB retrieval, but still below the strongest models |
| LoRA T5 | 15 | 0.8626 | 4 | Best result before the feedback loop |
| Feedback-LoRA T5 | 15 | 0.8712 | 7 | Best final result after critic-memory retraining |

## Interpretation

The feedback-guided LoRA model outperformed the original LoRA model and all previous reply-generation variants. This supports the idea that critic outputs and stored failures can act as useful supervision signals for iterative improvement, not only as runtime quality-control tools.
