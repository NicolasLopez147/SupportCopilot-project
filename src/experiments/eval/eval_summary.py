import csv
import json
import sys
import re
from pathlib import Path

import torch
from bert_score import score as bertscore_score
from transformers import pipeline

from src.utils.paths import EXPERIMENTS_OUTPUT_DIR


DEFAULT_PREDICTIONS_PATH = EXPERIMENTS_OUTPUT_DIR / "summary" / "api_baseline" / "test_predictions.json"
NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MANUAL_REVIEW_SAMPLES = 10
CHUNK_SIZE = 4
MAX_ENTAILMENT_SAMPLES = 5


def load_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_bertscore(predictions: list[dict]) -> dict:
    references = [item["reference_summary"] for item in predictions]
    candidates = [item["predicted_summary"] for item in predictions]

    precision, recall, f1 = bertscore_score(
        candidates,
        references,
        lang="en",
        verbose=False,
    )

    per_sample = []
    for item, p_score, r_score, f1_score in zip(predictions, precision, recall, f1):
        per_sample.append(
            {
                "conversation_id": item["conversation_id"],
                "bertscore_precision": round(float(p_score), 4),
                "bertscore_recall": round(float(r_score), 4),
                "bertscore_f1": round(float(f1_score), 4),
            }
        )

    aggregate = {
        "precision_mean": round(float(precision.mean()), 4),
        "recall_mean": round(float(recall.mean()), 4),
        "f1_mean": round(float(f1.mean()), 4),
    }

    return {"aggregate": aggregate, "per_sample": per_sample}


def extract_label_score(classifier_output: list[dict], label_suffix: str) -> float:
    for item in classifier_output:
        if item["label"].upper().endswith(label_suffix.upper()):
            return float(item["score"])
    raise ValueError(f"{label_suffix} label not found in classifier output")


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def build_conversation_chunks(conversation_text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    lines = [line.strip() for line in conversation_text.splitlines() if line.strip()]
    if not lines:
        return []

    chunks = []
    for start in range(0, len(lines), chunk_size):
        chunk = lines[start:start + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks


def compute_source_grounded_entailment(predictions: list[dict]) -> dict:
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "text-classification",
        model=NLI_MODEL_NAME,
        device=device,
        top_k=None,
    )

    per_sample = []
    support_scores = []
    contradiction_scores = []

    subset = predictions[:MAX_ENTAILMENT_SAMPLES]

    for index, item in enumerate(subset, start=1):
        print(
            f"[eval] entailment sample {index}/{len(subset)} -> "
            f"{item['conversation_id']}"
        )
        generated = item["predicted_summary"]
        conversation_text = item["conversation_text"]

        summary_sentences = split_sentences(generated)
        conversation_chunks = build_conversation_chunks(conversation_text)

        if not summary_sentences or not conversation_chunks:
            continue

        sentence_support_scores = []
        sentence_contradiction_scores = []

        for sentence in summary_sentences:
            best_entailment = 0.0
            best_contradiction = 0.0

            for chunk in conversation_chunks:
                output = classifier({"text": chunk, "text_pair": sentence})
                entailment_score = extract_label_score(output, "ENTAILMENT")
                contradiction_score = extract_label_score(output, "CONTRADICTION")

                best_entailment = max(best_entailment, entailment_score)
                best_contradiction = max(best_contradiction, contradiction_score)

            sentence_support_scores.append(best_entailment)
            sentence_contradiction_scores.append(best_contradiction)

        support_mean = sum(sentence_support_scores) / len(sentence_support_scores)
        contradiction_mean = sum(sentence_contradiction_scores) / len(sentence_contradiction_scores)

        support_scores.append(support_mean)
        contradiction_scores.append(contradiction_mean)

        per_sample.append(
            {
                "conversation_id": item["conversation_id"],
                "source_support_mean": round(support_mean, 4),
                "source_contradiction_mean": round(contradiction_mean, 4),
                "num_summary_sentences": len(summary_sentences),
            }
        )

    aggregate = {
        "source_support_mean": round(sum(support_scores) / len(support_scores), 4),
        "source_contradiction_mean": round(sum(contradiction_scores) / len(contradiction_scores), 4),
    }

    return {"aggregate": aggregate, "per_sample": per_sample}


def merge_metrics(predictions: list[dict], bertscore: dict, entailment: dict) -> list[dict]:
    bertscore_by_id = {item["conversation_id"]: item for item in bertscore["per_sample"]}
    entailment_by_id = {item["conversation_id"]: item for item in entailment["per_sample"]}

    merged = []
    for item in predictions:
        conversation_id = item["conversation_id"]
        merged.append(
            {
                "conversation_id": conversation_id,
                "reference_summary": item["reference_summary"],
                "predicted_summary": item["predicted_summary"],
                **bertscore_by_id[conversation_id],
                **entailment_by_id.get(
                    conversation_id,
                    {
                        "source_support_mean": None,
                        "source_contradiction_mean": None,
                        "num_summary_sentences": None,
                    },
                ),
            }
        )

    return merged


def save_json(data: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_manual_review_template(predictions: list[dict], path: Path) -> None:
    fieldnames = [
        "conversation_id",
        "reference_summary",
        "predicted_summary",
        "factual",
        "useful",
        "concise",
        "no_hallucination",
        "notes",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in predictions[:MANUAL_REVIEW_SAMPLES]:
            writer.writerow(
                {
                    "conversation_id": item["conversation_id"],
                    "reference_summary": item["reference_summary"],
                    "predicted_summary": item["predicted_summary"],
                    "factual": "",
                    "useful": "",
                    "concise": "",
                    "no_hallucination": "",
                    "notes": "",
                }
            )


def resolve_predictions_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    return DEFAULT_PREDICTIONS_PATH


def main() -> None:
    predictions_path = resolve_predictions_path()
    output_dir = predictions_path.parent

    predictions = load_predictions(predictions_path)
    if not predictions:
        print(f"[error] no predictions found in {predictions_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] loaded {len(predictions)} generated summaries")
    print("[eval] computing BERTScore")
    bertscore = compute_bertscore(predictions)

    print(f"[eval] computing source-grounded entailment with {NLI_MODEL_NAME}")
    entailment = compute_source_grounded_entailment(predictions)

    merged_metrics = merge_metrics(predictions, bertscore, entailment)

    aggregate_metrics = {
        "bertscore": bertscore["aggregate"],
        "source_grounded_entailment": entailment["aggregate"],
        "num_samples": len(predictions),
        "num_entailment_samples": min(len(predictions), MAX_ENTAILMENT_SAMPLES),
    }

    save_json(aggregate_metrics, output_dir / "summary_metrics.json")
    save_json(merged_metrics, output_dir / "summary_metrics_per_sample.json")
    save_manual_review_template(predictions, output_dir / "manual_review_template.csv")

    print(f"[saved] aggregate metrics -> {output_dir / 'summary_metrics.json'}")
    print(f"[saved] per-sample metrics -> {output_dir / 'summary_metrics_per_sample.json'}")
    print(f"[saved] manual review template -> {output_dir / 'manual_review_template.csv'}")
    print(f"[summary] BERTScore F1 mean: {bertscore['aggregate']['f1_mean']:.4f}")
    print(
        "[summary] Source support mean: "
        f"{entailment['aggregate']['source_support_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
