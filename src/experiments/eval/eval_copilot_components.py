import argparse
import json
from pathlib import Path

from src.experiments.eval.eval_reply_methods import (
    align_predictions as align_reply_predictions,
    build_comparison as build_reply_comparison,
    load_predictions as load_reply_predictions,
)
from src.experiments.eval.eval_summary_methods import (
    align_predictions as align_summary_predictions,
    compute_bertscore as compute_summary_bertscore,
    load_predictions as load_summary_predictions,
)
from src.utils.paths import EXPERIMENTS_OUTPUT_DIR


DEFAULT_INTENT_METRICS_PATH = (
    EXPERIMENTS_OUTPUT_DIR / "intent" / "synthetic_embedding" / "metrics.json"
)
DEFAULT_REPLY_BASELINE_PATH = EXPERIMENTS_OUTPUT_DIR / "reply" / "baseline" / "test_replies.json"
DEFAULT_REPLY_RETRIEVAL_PATH = EXPERIMENTS_OUTPUT_DIR / "reply" / "retrieval" / "test_replies.json"
DEFAULT_REPLY_LORA_PATH = EXPERIMENTS_OUTPUT_DIR / "reply" / "lora_base" / "test_replies.json"
DEFAULT_REPLY_FEEDBACK_PATH = EXPERIMENTS_OUTPUT_DIR / "reply" / "lora_feedback" / "test_replies.json"
DEFAULT_SUMMARY_LORA_PATH = EXPERIMENTS_OUTPUT_DIR / "summary" / "lora_base" / "test_predictions.json"
DEFAULT_SUMMARY_FEEDBACK_PATH = EXPERIMENTS_OUTPUT_DIR / "summary" / "lora_feedback" / "test_predictions.json"
DEFAULT_OUTPUT_PATH = EXPERIMENTS_OUTPUT_DIR / "comparison_overview.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Afficher une vue compacte d'évaluation pour l'intention, le résumé et la réponse."
    )
    parser.add_argument("--intent-metrics-path", type=Path, default=DEFAULT_INTENT_METRICS_PATH)
    parser.add_argument("--reply-baseline-path", type=Path, default=DEFAULT_REPLY_BASELINE_PATH)
    parser.add_argument("--reply-retrieval-path", type=Path, default=DEFAULT_REPLY_RETRIEVAL_PATH)
    parser.add_argument("--reply-lora-path", type=Path, default=DEFAULT_REPLY_LORA_PATH)
    parser.add_argument("--reply-feedback-path", type=Path, default=None)
    parser.add_argument("--summary-lora-path", type=Path, default=DEFAULT_SUMMARY_LORA_PATH)
    parser.add_argument("--summary-feedback-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_intent_section(path: Path) -> dict | None:
    if not path.exists():
        return None
    metrics = load_json(path)
    return {
        "valid_accuracy": metrics.get("valid_accuracy"),
        "valid_macro_f1": metrics.get("valid_macro_f1"),
        "test_accuracy": metrics.get("test_accuracy"),
        "test_macro_f1": metrics.get("test_macro_f1"),
        "num_classes": metrics.get("num_classes"),
    }


def build_summary_section(lora_path: Path, feedback_path: Path | None) -> dict | None:
    if not lora_path.exists():
        return None

    lora_predictions = load_summary_predictions(lora_path)
    feedback_predictions = (
        load_summary_predictions(feedback_path)
        if feedback_path is not None and feedback_path.exists()
        else None
    )
    aligned = align_summary_predictions(lora_predictions, feedback_predictions)
    if not aligned:
        return None

    references = [item["reference_summary"] for item in aligned]
    lora_candidates = [item["lora_summary"] for item in aligned]
    lora_scores = compute_summary_bertscore(lora_candidates, references)

    result = {
        "num_samples": len(aligned),
        "lora": {"bertscore_f1_mean": lora_scores["f1_mean"]},
    }

    if feedback_predictions is not None:
        feedback_candidates = [item["lora_feedback_summary"] for item in aligned]
        feedback_scores = compute_summary_bertscore(feedback_candidates, references)
        lora_wins = 0
        feedback_wins = 0
        ties = 0
        for idx in range(len(aligned)):
            left = lora_scores["f1"][idx]
            right = feedback_scores["f1"][idx]
            if abs(left - right) <= 0.0005:
                ties += 1
            elif left > right:
                lora_wins += 1
            else:
                feedback_wins += 1
        result["lora_feedback"] = {"bertscore_f1_mean": feedback_scores["f1_mean"]}
        result["comparison"] = {
            "lora_wins": lora_wins,
            "lora_feedback_wins": feedback_wins,
            "ties": ties,
        }

    return result


def build_reply_section(
    baseline_path: Path,
    retrieval_path: Path,
    lora_path: Path,
    feedback_path: Path | None,
) -> dict | None:
    required = [baseline_path, retrieval_path, lora_path]
    if any(not path.exists() for path in required):
        return None

    baseline_predictions = load_reply_predictions(baseline_path)
    retrieval_predictions = load_reply_predictions(retrieval_path)
    lora_predictions = load_reply_predictions(lora_path)
    feedback_predictions = (
        load_reply_predictions(feedback_path)
        if feedback_path is not None and feedback_path.exists()
        else None
    )

    aligned = align_reply_predictions(
        baseline_predictions=baseline_predictions,
        retrieval_predictions=retrieval_predictions,
        lora_predictions=lora_predictions,
        lora_feedback_predictions=feedback_predictions,
    )
    if not aligned:
        return None

    aggregate, _, _ = build_reply_comparison(
        aligned_predictions=aligned,
        include_lora_feedback=feedback_predictions is not None,
    )
    return aggregate


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if args.reply_feedback_path is None and DEFAULT_REPLY_FEEDBACK_PATH.exists():
        args.reply_feedback_path = DEFAULT_REPLY_FEEDBACK_PATH
    if args.summary_feedback_path is None and DEFAULT_SUMMARY_FEEDBACK_PATH.exists():
        args.summary_feedback_path = DEFAULT_SUMMARY_FEEDBACK_PATH

    overview = {
        "intent": build_intent_section(args.intent_metrics_path),
        "summary": build_summary_section(args.summary_lora_path, args.summary_feedback_path),
        "reply": build_reply_section(
            args.reply_baseline_path,
            args.reply_retrieval_path,
            args.reply_lora_path,
            args.reply_feedback_path,
        ),
    }

    save_json(overview, args.output_path)
    print(f"[saved] vue d'ensemble -> {args.output_path}")

    if overview["intent"] is not None:
        print(
            "[intent] accuracy / macro_f1 validation : "
            f"{overview['intent']['valid_accuracy']:.4f} / {overview['intent']['valid_macro_f1']:.4f}"
        )
        print(
            "[intent] accuracy / macro_f1 test : "
            f"{overview['intent']['test_accuracy']:.4f} / {overview['intent']['test_macro_f1']:.4f}"
        )

    if overview["summary"] is not None:
        print(
            "[summary] Moyenne BERTScore F1 LoRA : "
            f"{overview['summary']['lora']['bertscore_f1_mean']:.4f}"
        )
        if "lora_feedback" in overview["summary"]:
            print(
                "[summary] Moyenne BERTScore F1 Feedback LoRA : "
                f"{overview['summary']['lora_feedback']['bertscore_f1_mean']:.4f}"
            )
            print(
                "[summary] Victoires LoRA / Feedback-LoRA / Égalités : "
                f"{overview['summary']['comparison']['lora_wins']} / "
                f"{overview['summary']['comparison']['lora_feedback_wins']} / "
                f"{overview['summary']['comparison']['ties']}"
            )

    if overview["reply"] is not None:
        print(
            "[reply] F1 Baseline / Retrieval / LoRA : "
            f"{overview['reply']['baseline']['bertscore_f1_mean']:.4f} / "
            f"{overview['reply']['retrieval']['bertscore_f1_mean']:.4f} / "
            f"{overview['reply']['lora']['bertscore_f1_mean']:.4f}"
        )
        if "lora_feedback" in overview["reply"]:
            print(
                "[reply] Moyenne BERTScore F1 Feedback LoRA : "
                f"{overview['reply']['lora_feedback']['bertscore_f1_mean']:.4f}"
            )
            print(
                "[reply] Victoires Baseline / Retrieval / LoRA / Feedback-LoRA / Égalités : "
                f"{overview['reply']['comparison']['baseline_wins']} / "
                f"{overview['reply']['comparison']['retrieval_wins']} / "
                f"{overview['reply']['comparison']['lora_wins']} / "
                f"{overview['reply']['comparison']['lora_feedback_wins']} / "
                f"{overview['reply']['comparison']['ties']}"
            )
        else:
            print(
                "[reply] Victoires Baseline / Retrieval / LoRA / Égalités : "
                f"{overview['reply']['comparison']['baseline_wins']} / "
                f"{overview['reply']['comparison']['retrieval_wins']} / "
                f"{overview['reply']['comparison']['lora_wins']} / "
                f"{overview['reply']['comparison']['ties']}"
            )


if __name__ == "__main__":
    main()
