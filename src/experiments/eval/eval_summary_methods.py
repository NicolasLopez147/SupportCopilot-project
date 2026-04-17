import argparse
import csv
import json
from pathlib import Path

from bert_score import score as bertscore_score

from src.utils.paths import EXPERIMENTS_OUTPUT_DIR


DEFAULT_LORA_PATH = EXPERIMENTS_OUTPUT_DIR / "summary" / "lora_base" / "test_predictions.json"
DEFAULT_LORA_FEEDBACK_PATH = (
    EXPERIMENTS_OUTPUT_DIR / "summary" / "lora_feedback" / "test_predictions.json"
)
DEFAULT_OUTPUT_DIR = EXPERIMENTS_OUTPUT_DIR / "summary" / "comparison"
MANUAL_REVIEW_SAMPLES = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare original LoRA summary predictions against feedback-retrained LoRA summaries."
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=DEFAULT_LORA_PATH,
        help=f"Path to the original LoRA summary predictions. Default: {DEFAULT_LORA_PATH}",
    )
    parser.add_argument(
        "--lora-feedback-path",
        type=Path,
        default=None,
        help=(
            "Optional path to the feedback-retrained summary predictions. "
            f"Suggested: {DEFAULT_LORA_FEEDBACK_PATH}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save summary comparison outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_bertscore(candidates: list[str], references: list[str]) -> dict:
    precision, recall, f1 = bertscore_score(
        candidates,
        references,
        lang="en",
        verbose=False,
    )
    return {
        "precision": [round(float(item), 4) for item in precision],
        "recall": [round(float(item), 4) for item in recall],
        "f1": [round(float(item), 4) for item in f1],
        "precision_mean": round(float(precision.mean()), 4),
        "recall_mean": round(float(recall.mean()), 4),
        "f1_mean": round(float(f1.mean()), 4),
    }


def align_predictions(
    lora_predictions: list[dict],
    feedback_predictions: list[dict] | None = None,
) -> list[dict]:
    feedback_by_id = (
        {item["conversation_id"]: item for item in feedback_predictions}
        if feedback_predictions
        else {}
    )
    aligned = []

    for lora_item in lora_predictions:
        conversation_id = lora_item["conversation_id"]
        feedback_item = feedback_by_id.get(conversation_id)
        if feedback_predictions is not None and feedback_item is None:
            continue

        row = {
            "conversation_id": conversation_id,
            "conversation_text": lora_item.get("conversation_text"),
            "reference_summary": lora_item.get("reference_summary"),
            "lora_summary": lora_item.get("predicted_summary", ""),
        }
        if feedback_item is not None:
            row["lora_feedback_summary"] = feedback_item.get("predicted_summary", "")

        aligned.append(row)

    return aligned


def save_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_manual_review_template(per_sample: list[dict], path: Path) -> None:
    fieldnames = [
        "conversation_id",
        "reference_summary",
        "lora_summary",
        "lora_feedback_summary",
        "best_methods_auto",
        "manual_preferred_method",
        "lora_factual",
        "lora_useful",
        "lora_feedback_factual",
        "lora_feedback_useful",
        "notes",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in per_sample[:MANUAL_REVIEW_SAMPLES]:
            writer.writerow(
                {
                    "conversation_id": item["conversation_id"],
                    "reference_summary": item["reference_summary"],
                    "lora_summary": item["lora_summary"],
                    "lora_feedback_summary": item["lora_feedback_summary"],
                    "best_methods_auto": ",".join(item["best_methods"]),
                    "manual_preferred_method": "",
                    "lora_factual": "",
                    "lora_useful": "",
                    "lora_feedback_factual": "",
                    "lora_feedback_useful": "",
                    "notes": "",
                }
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lora_predictions = load_predictions(args.lora_path)
    feedback_predictions = (
        load_predictions(args.lora_feedback_path) if args.lora_feedback_path else None
    )
    aligned_predictions = align_predictions(lora_predictions, feedback_predictions)

    if not aligned_predictions:
        print("[erreur] aucune prédiction de résumé alignée trouvée entre les deux méthodes")
        return

    print(f"[eval] aligned samples: {len(aligned_predictions)}")
    if feedback_predictions is not None:
        print("[eval] calcul du BERTScore pour les résumés LoRA et feedback-LoRA")
    else:
        print("[eval] calcul du BERTScore pour les résumés LoRA")

    references = [item["reference_summary"] for item in aligned_predictions]
    lora_candidates = [item["lora_summary"] for item in aligned_predictions]

    lora_scores = compute_bertscore(lora_candidates, references)
    feedback_scores = None
    if feedback_predictions is not None:
        feedback_candidates = [item["lora_feedback_summary"] for item in aligned_predictions]
        feedback_scores = compute_bertscore(feedback_candidates, references)

    per_sample = []
    lora_wins = 0
    feedback_wins = 0
    ties = 0

    for index, item in enumerate(aligned_predictions):
        method_f1 = {"lora": lora_scores["f1"][index]}
        if feedback_scores is not None:
            method_f1["lora_feedback"] = feedback_scores["f1"][index]
        best_value = max(method_f1.values())
        best_methods = sorted(
            [method for method, score in method_f1.items() if abs(score - best_value) <= 0.0005]
        )

        if len(best_methods) == 1:
            if best_methods[0] == "lora":
                lora_wins += 1
            elif best_methods[0] == "lora_feedback":
                feedback_wins += 1
        else:
            ties += 1

        row = {
            "conversation_id": item["conversation_id"],
            "reference_summary": item["reference_summary"],
            "lora_summary": item["lora_summary"],
            "lora_bertscore_f1": lora_scores["f1"][index],
            "best_methods": best_methods,
        }
        if feedback_scores is not None:
            row["lora_feedback_summary"] = item["lora_feedback_summary"]
            row["lora_feedback_bertscore_f1"] = feedback_scores["f1"][index]
        per_sample.append(row)

    aggregate = {
        "num_samples": len(aligned_predictions),
        "lora": {
            "bertscore_precision_mean": lora_scores["precision_mean"],
            "bertscore_recall_mean": lora_scores["recall_mean"],
            "bertscore_f1_mean": lora_scores["f1_mean"],
        },
        "comparison": {
            "lora_wins": lora_wins,
            "ties": ties,
        },
    }
    if feedback_scores is not None:
        aggregate["lora_feedback"] = {
            "bertscore_precision_mean": feedback_scores["precision_mean"],
            "bertscore_recall_mean": feedback_scores["recall_mean"],
            "bertscore_f1_mean": feedback_scores["f1_mean"],
        }
        aggregate["comparison"]["lora_feedback_wins"] = feedback_wins

    save_json(aggregate, args.output_dir / "summary_comparison_metrics.json")
    save_json(per_sample, args.output_dir / "summary_comparison_per_sample.json")
    save_manual_review_template(per_sample, args.output_dir / "summary_manual_review_template.csv")

    print(f"[saved] métriques agrégées -> {args.output_dir / 'summary_comparison_metrics.json'}")
    print(
        f"[saved] comparaison par échantillon -> {args.output_dir / 'summary_comparison_per_sample.json'}"
    )
    print(
        f"[saved] modèle de revue manuelle -> {args.output_dir / 'summary_manual_review_template.csv'}"
    )
    print(f"[summary] Moyenne BERTScore F1 LoRA : {aggregate['lora']['bertscore_f1_mean']:.4f}")
    if "lora_feedback" in aggregate:
        print(
            "[summary] Moyenne BERTScore F1 Feedback LoRA : "
            f"{aggregate['lora_feedback']['bertscore_f1_mean']:.4f}"
        )
        print(
            "[summary] Victoires LoRA / Feedback-LoRA / Égalités : "
            f"{aggregate['comparison']['lora_wins']} / "
            f"{aggregate['comparison']['lora_feedback_wins']} / "
            f"{aggregate['comparison']['ties']}"
        )
    else:
        print(
            "[summary] Victoires LoRA / Égalités : "
            f"{aggregate['comparison']['lora_wins']} / "
            f"{aggregate['comparison']['ties']}"
        )



if __name__ == "__main__":
    main()
