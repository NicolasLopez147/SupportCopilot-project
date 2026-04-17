# Dialogue Summarization Fine-Tuning

Comparison of the baseline, full fine-tuning, and LoRA fine-tuning on `tweetsum`.

| Model | Setup | Samples | BERTScore F1 | Source Support | Notes |
|---|---|---:|---:|---:|---|
| `philschmid/bart-large-cnn-samsum` | Baseline | 10 | 0.8761 | 0.6056 | Useful local dialogue baseline |
| `google/flan-t5-small` | Full fine-tuning | 10 | 0.8870 | 0.9827 | Best source-grounded support on the evaluation subset |
| `google/flan-t5-small` | LoRA fine-tuning | 10 | 0.8904 | 0.9023 | Best BERTScore with a lighter adaptation strategy |

## Interpretation

Both fine-tuning approaches improve over the baseline summarization model. Full fine-tuning yields the strongest source-grounded support score, while LoRA reaches the best semantic similarity score and remains highly competitive at lower adaptation cost.

## Evaluation Notes

- Main semantic metric: `BERTScore`
- Complementary factual-support metric: `source-grounded entailment`
- Manual review remains useful for checking verbosity and hallucination tendencies
