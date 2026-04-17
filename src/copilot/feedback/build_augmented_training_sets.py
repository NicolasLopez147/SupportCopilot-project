import argparse
import json
from pathlib import Path

from src.utils.paths import DATA_DIR, FEEDBACK_AUGMENTED_DIR, FEEDBACK_CANDIDATES_DIR


SYNTHETIC_DIR = DATA_DIR / "synthetic"
OUTPUT_DIR = FEEDBACK_AUGMENTED_DIR

BASE_INTENT_TRAIN_PATH = SYNTHETIC_DIR / "reply_train.jsonl"
BASE_REPLY_TRAIN_PATH = SYNTHETIC_DIR / "reply_train.jsonl"

INTENT_CANDIDATES_PATH = FEEDBACK_CANDIDATES_DIR / "intent_retraining_candidates.jsonl"
SUMMARY_CANDIDATES_PATH = FEEDBACK_CANDIDATES_DIR / "summary_retraining_candidates.jsonl"
REPLY_CANDIDATES_PATH = FEEDBACK_CANDIDATES_DIR / "reply_retraining_candidates.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge critic-derived retraining candidates with the base datasets to create "
            "offline training batches."
        )
    )
    parser.add_argument(
        "--base-intent-train-path",
        type=Path,
        default=BASE_INTENT_TRAIN_PATH,
        help=f"Base synthetic train split for intent. Default: {BASE_INTENT_TRAIN_PATH}",
    )
    parser.add_argument(
        "--base-reply-train-path",
        type=Path,
        default=BASE_REPLY_TRAIN_PATH,
        help=f"Base synthetic train split for reply. Default: {BASE_REPLY_TRAIN_PATH}",
    )
    parser.add_argument(
        "--intent-candidates-path",
        type=Path,
        default=INTENT_CANDIDATES_PATH,
        help=f"Intent retraining candidates path. Default: {INTENT_CANDIDATES_PATH}",
    )
    parser.add_argument(
        "--summary-candidates-path",
        type=Path,
        default=SUMMARY_CANDIDATES_PATH,
        help=f"Summary retraining candidates path. Default: {SUMMARY_CANDIDATES_PATH}",
    )
    parser.add_argument(
        "--reply-candidates-path",
        type=Path,
        default=REPLY_CANDIDATES_PATH,
        help=f"Reply retraining candidates path. Default: {REPLY_CANDIDATES_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for augmented sets. Default: {OUTPUT_DIR}",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_index(rows: list[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        conversation_id = row.get("conversation_id")
        if conversation_id:
            indexed[str(conversation_id)] = row
    return indexed


def messages_from_conversation_text(conversation_text: str) -> list[dict]:
    messages: list[dict] = []
    for line in conversation_text.splitlines():
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        speaker_part, text_part = stripped.split(":", 1)
        speaker_label = speaker_part.strip().lower()
        text = text_part.strip()
        if not text:
            continue

        if speaker_label == "customer":
            speaker = "customer"
        elif speaker_label == "agent":
            speaker = "agent"
        else:
            speaker = speaker_label

        messages.append({"speaker": speaker, "text": text})
    return messages


def merge_intent_training_set(base_rows: list[dict], candidates: list[dict]) -> tuple[list[dict], dict]:
    merged_rows = [dict(row) for row in base_rows]
    merged_by_id = build_index(merged_rows)

    replaced = 0
    appended = 0

    for candidate in candidates:
        conversation_id = candidate.get("conversation_id")
        target_intent = candidate.get("target_intent")
        if not conversation_id or not target_intent:
            continue

        if conversation_id in merged_by_id:
            merged_by_id[conversation_id]["scenario"] = target_intent
            merged_by_id[conversation_id]["intent_feedback_target"] = target_intent
            merged_by_id[conversation_id]["intent_feedback_issues"] = candidate.get("issues", [])
            merged_by_id[conversation_id]["intent_feedback_source"] = candidate.get("source")
            replaced += 1
        else:
            appended += 1
            merged_rows.append(
                {
                    "conversation_id": conversation_id,
                    "scenario": target_intent,
                    "messages": messages_from_conversation_text(candidate.get("conversation_text", "")),
                    "source": candidate.get("source"),
                    "intent_feedback_target": target_intent,
                    "intent_feedback_issues": candidate.get("issues", []),
                    "intent_feedback_source": candidate.get("source"),
                }
            )

    return merged_rows, {"replaced_rows": replaced, "appended_rows": appended}


def merge_reply_training_set(base_rows: list[dict], candidates: list[dict]) -> tuple[list[dict], dict]:
    merged_rows = [dict(row) for row in base_rows]
    merged_by_id = build_index(merged_rows)

    replaced = 0
    appended = 0

    for candidate in candidates:
        conversation_id = candidate.get("conversation_id")
        target_reply = candidate.get("target_reply")
        if not conversation_id or not target_reply:
            continue

        if conversation_id in merged_by_id:
            merged_by_id[conversation_id]["reference_reply"] = target_reply
            merged_by_id[conversation_id]["reply_feedback_target"] = target_reply
            merged_by_id[conversation_id]["reply_feedback_issues"] = candidate.get("issues", [])
            merged_by_id[conversation_id]["reply_feedback_source"] = candidate.get("source")
            replaced += 1
        else:
            appended += 1
            merged_rows.append(
                {
                    "conversation_id": conversation_id,
                    "scenario": candidate.get("scenario"),
                    "messages": messages_from_conversation_text(candidate.get("conversation_text", "")),
                    "reference_reply": target_reply,
                    "source": candidate.get("source"),
                    "reply_feedback_target": target_reply,
                    "reply_feedback_issues": candidate.get("issues", []),
                    "reply_feedback_source": candidate.get("source"),
                }
            )

    return merged_rows, {"replaced_rows": replaced, "appended_rows": appended}


def build_summary_support_batch(candidates: list[dict]) -> list[dict]:
    batch: list[dict] = []
    for candidate in candidates:
        conversation_text = candidate.get("conversation_text", "")
        target_summary = candidate.get("target_summary")
        conversation_id = candidate.get("conversation_id")
        if not conversation_text or not target_summary or not conversation_id:
            continue

        batch.append(
            {
                "conversation_id": conversation_id,
                "scenario": candidate.get("scenario"),
                "conversation_text": conversation_text,
                "input_text": (
                    "summarize the following customer support conversation: "
                    f"{conversation_text}"
                ),
                "target_text": target_summary,
                "raw_summary": candidate.get("raw_summary", ""),
                "issues": candidate.get("issues", []),
                "source": candidate.get("source"),
            }
        )
    return batch


def build_report(
    base_intent_rows: list[dict],
    base_reply_rows: list[dict],
    intent_candidates: list[dict],
    summary_candidates: list[dict],
    reply_candidates: list[dict],
    intent_merge_stats: dict,
    reply_merge_stats: dict,
    summary_batch: list[dict],
) -> dict:
    return {
        "intent": {
            "base_rows": len(base_intent_rows),
            "candidate_rows": len(intent_candidates),
            "augmented_rows": len(base_intent_rows) + intent_merge_stats.get("appended_rows", 0),
            **intent_merge_stats,
        },
        "summary": {
            "candidate_rows": len(summary_candidates),
            "support_batch_rows": len(summary_batch),
        },
        "reply": {
            "base_rows": len(base_reply_rows),
            "candidate_rows": len(reply_candidates),
            "augmented_rows": len(base_reply_rows) + reply_merge_stats.get("appended_rows", 0),
            **reply_merge_stats,
        },
    }


def main() -> None:
    args = parse_args()

    base_intent_rows = load_jsonl(args.base_intent_train_path)
    base_reply_rows = load_jsonl(args.base_reply_train_path)
    intent_candidates = load_jsonl(args.intent_candidates_path)
    summary_candidates = load_jsonl(args.summary_candidates_path)
    reply_candidates = load_jsonl(args.reply_candidates_path)

    augmented_intent_rows, intent_merge_stats = merge_intent_training_set(
        base_rows=base_intent_rows,
        candidates=intent_candidates,
    )
    augmented_reply_rows, reply_merge_stats = merge_reply_training_set(
        base_rows=base_reply_rows,
        candidates=reply_candidates,
    )
    summary_support_batch = build_summary_support_batch(summary_candidates)

    output_dir = args.output_dir
    intent_output_path = output_dir / "intent_augmented_train.jsonl"
    summary_output_path = output_dir / "summary_support_retraining_batch.jsonl"
    reply_output_path = output_dir / "reply_augmented_train.jsonl"
    report_output_path = output_dir / "augmented_training_report.json"

    save_jsonl(augmented_intent_rows, intent_output_path)
    save_jsonl(summary_support_batch, summary_output_path)
    save_jsonl(augmented_reply_rows, reply_output_path)

    report = build_report(
        base_intent_rows=base_intent_rows,
        base_reply_rows=base_reply_rows,
        intent_candidates=intent_candidates,
        summary_candidates=summary_candidates,
        reply_candidates=reply_candidates,
        intent_merge_stats=intent_merge_stats,
        reply_merge_stats=reply_merge_stats,
        summary_batch=summary_support_batch,
    )
    save_json(report, report_output_path)

    print(f"[saved] augmented intent train -> {intent_output_path}")
    print(f"[saved] support summary batch -> {summary_output_path}")
    print(f"[saved] augmented reply train -> {reply_output_path}")
    print(f"[saved] augmented training report -> {report_output_path}")
    print(f"[summary] intent train rows: {len(augmented_intent_rows)}")
    print(f"[summary] summary support rows: {len(summary_support_batch)}")
    print(f"[summary] reply train rows: {len(augmented_reply_rows)}")


if __name__ == "__main__":
    main()
