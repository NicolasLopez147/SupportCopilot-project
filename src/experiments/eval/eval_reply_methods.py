import argparse
import csv
import json
from pathlib import Path

from bert_score import score as bertscore_score

from src.utils.paths import EXPERIMENTS_OUTPUT_DIR


DEFAULT_BASELINE_PATH = EXPERIMENTS_OUTPUT_DIR / "reply" / "baseline" / "test_replies.json"
DEFAULT_RETRIEVAL_PATH = EXPERIMENTS_OUTPUT_DIR / "reply" / "retrieval" / "test_replies.json"
DEFAULT_LORA_PATH = EXPERIMENTS_OUTPUT_DIR / "reply" / "lora_base" / "test_replies.json"
DEFAULT_LORA_FEEDBACK_PATH = EXPERIMENTS_OUTPUT_DIR / "reply" / "lora_feedback" / "test_replies.json"
DEFAULT_OUTPUT_DIR = EXPERIMENTS_OUTPUT_DIR / "reply" / "comparison"
MANUAL_REVIEW_SAMPLES = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline, retrieval, and LoRA reply generation outputs."
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help=f"Path to the baseline replies JSON. Default: {DEFAULT_BASELINE_PATH}",
    )
    parser.add_argument(
        "--retrieval-path",
        type=Path,
        default=DEFAULT_RETRIEVAL_PATH,
        help=f"Path to the retrieval replies JSON. Default: {DEFAULT_RETRIEVAL_PATH}",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=DEFAULT_LORA_PATH,
        help=f"Path to the LoRA replies JSON. Default: {DEFAULT_LORA_PATH}",
    )
    parser.add_argument(
        "--lora-feedback-path",
        type=Path,
        default=None,
        help=(
            "Optional path to the feedback-retrained LoRA replies JSON. "
            f"Suggested: {DEFAULT_LORA_FEEDBACK_PATH}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save comparison outputs. Default: {DEFAULT_OUTPUT_DIR}",
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
    baseline_predictions: list[dict],
    retrieval_predictions: list[dict],
    lora_predictions: list[dict],
    lora_feedback_predictions: list[dict] | None = None,
) -> list[dict]:
    retrieval_by_id = {item["conversation_id"]: item for item in retrieval_predictions}
    lora_by_id = {item["conversation_id"]: item for item in lora_predictions}
    lora_feedback_by_id = (
        {item["conversation_id"]: item for item in lora_feedback_predictions}
        if lora_feedback_predictions
        else {}
    )

    aligned = []
    for baseline_item in baseline_predictions:
        conversation_id = baseline_item["conversation_id"]
        retrieval_item = retrieval_by_id.get(conversation_id)
        lora_item = lora_by_id.get(conversation_id)
        lora_feedback_item = lora_feedback_by_id.get(conversation_id)

        if retrieval_item is None or lora_item is None:
            continue
        if lora_feedback_predictions is not None and lora_feedback_item is None:
            continue

        row = {
            "conversation_id": conversation_id,
            "scenario": baseline_item.get("scenario"),
            "conversation_text": baseline_item.get("conversation_text"),
            "reference_reply": baseline_item.get("reference_reply"),
            "baseline_reply": baseline_item.get("suggested_reply", ""),
            "retrieval_reply": retrieval_item.get("suggested_reply", ""),
            "lora_reply": lora_item.get("suggested_reply", ""),
            "retrieved_chunks": retrieval_item.get("retrieved_chunks", []),
        }
        if lora_feedback_item is not None:
            row["lora_feedback_reply"] = lora_feedback_item.get("suggested_reply", "")

        aligned.append(row)

    return aligned


def summarize_method_wins(per_sample: list[dict], methods: list[str]) -> dict:
    wins = {method: 0 for method in methods}
    ties = 0

    for item in per_sample:
        best_methods = item["best_methods"]
        if len(best_methods) == 1:
            wins[best_methods[0]] += 1
        else:
            ties += 1

    return {"wins": wins, "ties": ties}


def compute_scenario_summary(per_sample: list[dict]) -> list[dict]:
    grouped = {}
    for item in per_sample:
        scenario = item.get("scenario") or "unknown"
        if scenario not in grouped:
            group = {
                "scenario": scenario,
                "num_samples": 0,
                "baseline_f1_sum": 0.0,
                "retrieval_f1_sum": 0.0,
                "lora_f1_sum": 0.0,
                "baseline_wins": 0,
                "retrieval_wins": 0,
                "lora_wins": 0,
                "ties": 0,
            }
            if "lora_feedback_bertscore_f1" in item:
                group["lora_feedback_f1_sum"] = 0.0
                group["lora_feedback_wins"] = 0
            grouped[scenario] = group
        group = grouped[scenario]
        group["num_samples"] += 1
        group["baseline_f1_sum"] += item["baseline_bertscore_f1"]
        group["retrieval_f1_sum"] += item["retrieval_bertscore_f1"]
        group["lora_f1_sum"] += item["lora_bertscore_f1"]
        if "lora_feedback_bertscore_f1" in item:
            group["lora_feedback_f1_sum"] += item["lora_feedback_bertscore_f1"]

        if len(item["best_methods"]) == 1:
            winner = item["best_methods"][0]
            if winner == "baseline":
                group["baseline_wins"] += 1
            elif winner == "retrieval":
                group["retrieval_wins"] += 1
            elif winner == "lora":
                group["lora_wins"] += 1
            elif winner == "lora_feedback":
                group["lora_feedback_wins"] += 1
        else:
            group["ties"] += 1

    summary = []
    for scenario in sorted(grouped):
        group = grouped[scenario]
        num_samples = group["num_samples"]
        summary.append(
            {
                "scenario": scenario,
                "num_samples": num_samples,
                "baseline_f1_mean": round(group["baseline_f1_sum"] / num_samples, 4),
                "retrieval_f1_mean": round(group["retrieval_f1_sum"] / num_samples, 4),
                "lora_f1_mean": round(group["lora_f1_sum"] / num_samples, 4),
                "baseline_wins": group["baseline_wins"],
                "retrieval_wins": group["retrieval_wins"],
                "lora_wins": group["lora_wins"],
                "ties": group["ties"],
            }
        )
        if "lora_feedback_f1_sum" in group:
            summary[-1]["lora_feedback_f1_mean"] = round(
                group["lora_feedback_f1_sum"] / num_samples, 4
            )
            summary[-1]["lora_feedback_wins"] = group["lora_feedback_wins"]

    return summary


def build_comparison(
    aligned_predictions: list[dict],
    include_lora_feedback: bool = False,
) -> tuple[dict, list[dict], list[dict]]:
    references = [item["reference_reply"] for item in aligned_predictions]
    baseline_candidates = [item["baseline_reply"] for item in aligned_predictions]
    retrieval_candidates = [item["retrieval_reply"] for item in aligned_predictions]
    lora_candidates = [item["lora_reply"] for item in aligned_predictions]
    lora_feedback_candidates = (
        [item["lora_feedback_reply"] for item in aligned_predictions]
        if include_lora_feedback
        else []
    )

    baseline_scores = compute_bertscore(baseline_candidates, references)
    retrieval_scores = compute_bertscore(retrieval_candidates, references)
    lora_scores = compute_bertscore(lora_candidates, references)
    lora_feedback_scores = (
        compute_bertscore(lora_feedback_candidates, references)
        if include_lora_feedback
        else None
    )

    per_sample = []
    methods = ["baseline", "retrieval", "lora"]
    if include_lora_feedback:
        methods.append("lora_feedback")

    for index, item in enumerate(aligned_predictions):
        method_f1 = {
            "baseline": baseline_scores["f1"][index],
            "retrieval": retrieval_scores["f1"][index],
            "lora": lora_scores["f1"][index],
        }
        if include_lora_feedback and lora_feedback_scores is not None:
            method_f1["lora_feedback"] = lora_feedback_scores["f1"][index]
        best_value = max(method_f1.values())
        best_methods = sorted(
            [method for method, score in method_f1.items() if abs(score - best_value) <= 0.0005]
        )

        row = {
            "conversation_id": item["conversation_id"],
            "scenario": item["scenario"],
            "reference_reply": item["reference_reply"],
            "baseline_reply": item["baseline_reply"],
            "retrieval_reply": item["retrieval_reply"],
            "lora_reply": item["lora_reply"],
            "baseline_bertscore_precision": baseline_scores["precision"][index],
            "baseline_bertscore_recall": baseline_scores["recall"][index],
            "baseline_bertscore_f1": baseline_scores["f1"][index],
            "retrieval_bertscore_precision": retrieval_scores["precision"][index],
            "retrieval_bertscore_recall": retrieval_scores["recall"][index],
            "retrieval_bertscore_f1": retrieval_scores["f1"][index],
            "lora_bertscore_precision": lora_scores["precision"][index],
            "lora_bertscore_recall": lora_scores["recall"][index],
            "lora_bertscore_f1": lora_scores["f1"][index],
            "best_methods": best_methods,
            "retrieved_chunks": item["retrieved_chunks"],
        }
        if include_lora_feedback and lora_feedback_scores is not None:
            row["lora_feedback_reply"] = item["lora_feedback_reply"]
            row["lora_feedback_bertscore_precision"] = lora_feedback_scores["precision"][index]
            row["lora_feedback_bertscore_recall"] = lora_feedback_scores["recall"][index]
            row["lora_feedback_bertscore_f1"] = lora_feedback_scores["f1"][index]

        per_sample.append(row)

    win_summary = summarize_method_wins(per_sample, methods)

    aggregate = {
        "num_samples": len(aligned_predictions),
        "baseline": {
            "bertscore_precision_mean": baseline_scores["precision_mean"],
            "bertscore_recall_mean": baseline_scores["recall_mean"],
            "bertscore_f1_mean": baseline_scores["f1_mean"],
        },
        "retrieval": {
            "bertscore_precision_mean": retrieval_scores["precision_mean"],
            "bertscore_recall_mean": retrieval_scores["recall_mean"],
            "bertscore_f1_mean": retrieval_scores["f1_mean"],
        },
        "lora": {
            "bertscore_precision_mean": lora_scores["precision_mean"],
            "bertscore_recall_mean": lora_scores["recall_mean"],
            "bertscore_f1_mean": lora_scores["f1_mean"],
        },
        "comparison": {
            "baseline_wins": win_summary["wins"]["baseline"],
            "retrieval_wins": win_summary["wins"]["retrieval"],
            "lora_wins": win_summary["wins"]["lora"],
            "ties": win_summary["ties"],
        },
    }
    if include_lora_feedback and lora_feedback_scores is not None:
        aggregate["lora_feedback"] = {
            "bertscore_precision_mean": lora_feedback_scores["precision_mean"],
            "bertscore_recall_mean": lora_feedback_scores["recall_mean"],
            "bertscore_f1_mean": lora_feedback_scores["f1_mean"],
        }
        aggregate["comparison"]["lora_feedback_wins"] = win_summary["wins"]["lora_feedback"]

    scenario_summary = compute_scenario_summary(per_sample)
    return aggregate, per_sample, scenario_summary


def save_json(data: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_manual_review_template(per_sample: list[dict], path: Path) -> None:
    fieldnames = [
        "conversation_id",
        "scenario",
        "reference_reply",
        "baseline_reply",
        "retrieval_reply",
        "lora_reply",
        "best_methods_auto",
        "manual_preferred_method",
        "baseline_relevance",
        "retrieval_relevance",
        "lora_relevance",
        "baseline_grounded",
        "retrieval_grounded",
        "lora_grounded",
        "baseline_actionable",
        "retrieval_actionable",
        "lora_actionable",
        "notes",
    ]
    if per_sample and "lora_feedback_reply" in per_sample[0]:
        insert_after = fieldnames.index("lora_reply") + 1
        fieldnames[insert_after:insert_after] = ["lora_feedback_reply"]
        notes_index = fieldnames.index("notes")
        fieldnames[notes_index:notes_index] = [
            "lora_feedback_relevance",
            "lora_feedback_grounded",
            "lora_feedback_actionable",
        ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in per_sample[:MANUAL_REVIEW_SAMPLES]:
            row = {
                "conversation_id": item["conversation_id"],
                "scenario": item["scenario"],
                "reference_reply": item["reference_reply"],
                "baseline_reply": item["baseline_reply"],
                "retrieval_reply": item["retrieval_reply"],
                "lora_reply": item["lora_reply"],
                "best_methods_auto": ",".join(item["best_methods"]),
                "manual_preferred_method": "",
                "baseline_relevance": "",
                "retrieval_relevance": "",
                "lora_relevance": "",
                "baseline_grounded": "",
                "retrieval_grounded": "",
                "lora_grounded": "",
                "baseline_actionable": "",
                "retrieval_actionable": "",
                "lora_actionable": "",
                "notes": "",
            }
            if "lora_feedback_reply" in item:
                row["lora_feedback_reply"] = item["lora_feedback_reply"]
                row["lora_feedback_relevance"] = ""
                row["lora_feedback_grounded"] = ""
                row["lora_feedback_actionable"] = ""
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_predictions = load_predictions(args.baseline_path)
    retrieval_predictions = load_predictions(args.retrieval_path)
    lora_predictions = load_predictions(args.lora_path)
    lora_feedback_predictions = (
        load_predictions(args.lora_feedback_path) if args.lora_feedback_path else None
    )

    aligned_predictions = align_predictions(
        baseline_predictions=baseline_predictions,
        retrieval_predictions=retrieval_predictions,
        lora_predictions=lora_predictions,
        lora_feedback_predictions=lora_feedback_predictions,
    )

    if not aligned_predictions:
        print("[erreur] aucune prédiction alignée trouvée entre les sorties de réponse fournies")
        return

    print(f"[eval] aligned samples: {len(aligned_predictions)}")
    if lora_feedback_predictions is not None:
        print("[eval] calcul du BERTScore pour les réponses baseline, retrieval, LoRA et feedback-LoRA")
    else:
        print("[eval] calcul du BERTScore pour les réponses baseline, retrieval et LoRA")

    aggregate, per_sample, scenario_summary = build_comparison(
        aligned_predictions,
        include_lora_feedback=lora_feedback_predictions is not None,
    )

    save_json(aggregate, args.output_dir / "reply_comparison_metrics.json")
    save_json(per_sample, args.output_dir / "reply_comparison_per_sample.json")
    save_json(scenario_summary, args.output_dir / "reply_comparison_by_scenario.json")
    save_manual_review_template(
        per_sample,
        args.output_dir / "reply_manual_review_template.csv",
    )

    print(f"[saved] métriques agrégées -> {args.output_dir / 'reply_comparison_metrics.json'}")
    print(
        f"[saved] comparaison par échantillon -> "
        f"{args.output_dir / 'reply_comparison_per_sample.json'}"
    )
    print(
        f"[saved] résumé par scénario -> "
        f"{args.output_dir / 'reply_comparison_by_scenario.json'}"
    )
    print(
        f"[saved] modèle de revue manuelle -> "
        f"{args.output_dir / 'reply_manual_review_template.csv'}"
    )
    print(
        "[reply] Moyenne BERTScore F1 baseline : "
        f"{aggregate['baseline']['bertscore_f1_mean']:.4f}"
    )
    print(
        "[reply] Moyenne BERTScore F1 retrieval : "
        f"{aggregate['retrieval']['bertscore_f1_mean']:.4f}"
    )
    print(
        "[reply] Moyenne BERTScore F1 LoRA : "
        f"{aggregate['lora']['bertscore_f1_mean']:.4f}"
    )
    if "lora_feedback" in aggregate:
        print(
            "[reply] Moyenne BERTScore F1 Feedback LoRA : "
            f"{aggregate['lora_feedback']['bertscore_f1_mean']:.4f}"
        )
        print(
            "[reply] Victoires Baseline / Retrieval / LoRA / Feedback-LoRA / Égalités : "
            f"{aggregate['comparison']['baseline_wins']} / "
            f"{aggregate['comparison']['retrieval_wins']} / "
            f"{aggregate['comparison']['lora_wins']} / "
            f"{aggregate['comparison']['lora_feedback_wins']} / "
            f"{aggregate['comparison']['ties']}"
        )
    else:
        print(
            "[reply] Victoires Baseline / Retrieval / LoRA / Égalités : "
            f"{aggregate['comparison']['baseline_wins']} / "
            f"{aggregate['comparison']['retrieval_wins']} / "
            f"{aggregate['comparison']['lora_wins']} / "
            f"{aggregate['comparison']['ties']}"
        )


if __name__ == "__main__":
    main()
