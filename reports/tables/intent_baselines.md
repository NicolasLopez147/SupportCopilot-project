# Intent Classification Baselines

Results on the `banking77` intent classification benchmark.

| Model | Representation | Classifier | Accuracy | Macro F1 | Notes |
|---|---|---|---:|---:|---|
| TF-IDF baseline | TF-IDF (`ngram_range=(1,2)`) | Logistic Regression | 0.8578 | 0.8568 | Strong lexical baseline |
| Embedding baseline | `all-MiniLM-L6-v2` sentence embeddings | Logistic Regression | 0.9081 | 0.9081 | Better semantic representation |

## Interpretation

The embedding-based baseline outperforms the TF-IDF baseline by roughly 5 points on both accuracy and macro F1 while keeping the classifier fixed. This suggests that the main gain comes from a stronger text representation rather than from classifier complexity.
