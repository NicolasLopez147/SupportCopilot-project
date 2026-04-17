# Dialogue Summarization Baseline

Results for the first local summarization baseline on `tweetsum`.

| Model | Dataset | Samples | BERTScore F1 | Source Support | Notes |
|---|---|---:|---:|---:|---|
| `philschmid/bart-large-cnn-samsum` | `tweetsum` | 10 | 0.8761 | 0.6056 | Useful dialogue summary baseline, sometimes too verbose |

## Evaluation Setup

- Main semantic metric: `BERTScore`
- Complementary factual-support metric: `source-grounded entailment`
- Qualitative check: manual review template

## Interpretation

The local BART dialogue model produces semantically close summaries relative to the reference summaries. Manual inspection suggests that outputs are usually useful and factually reasonable, though some examples include extra detail or over-explanation. The source-grounded entailment metric is retained as an exploratory factual-support signal rather than the main decision metric.
