import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.utils.paths import FEEDBACK_CANDIDATES_DIR, FEEDBACK_MEMORY_DIR

DEFAULT_OUTPUT_DIR = FEEDBACK_CANDIDATES_DIR

INTENT_FAILURES_PATH = FEEDBACK_MEMORY_DIR / "intent_failures.jsonl"
SUMMARY_FAILURES_PATH = FEEDBACK_MEMORY_DIR / "summary_failures.jsonl"
REPLY_FAILURES_PATH = FEEDBACK_MEMORY_DIR / "reply_failures.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build offline improvement datasets from critic failure memory."
    )
    parser.add_argument(
        "--intent-failures-path",
        type=Path,
        default=INTENT_FAILURES_PATH,
        help=f"Path to intent failure memory. Default: {INTENT_FAILURES_PATH}",
    )
    parser.add_argument(
        "--summary-failures-path",
        type=Path,
        default=SUMMARY_FAILURES_PATH,
        help=f"Path to summary failure memory. Default: {SUMMARY_FAILURES_PATH}",
    )
    parser.add_argument(
        "--reply-failures-path",
        type=Path,
        default=REPLY_FAILURES_PATH,
        help=f"Path to reply failure memory. Default: {REPLY_FAILURES_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where derived retraining sets will be saved. Default: {DEFAULT_OUTPUT_DIR}",
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


def append_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.min


def conversation_key(row: dict) -> str:
    conversation_id = row.get("conversation_id")
    if conversation_id:
        return str(conversation_id)

    scenario = row.get("scenario") or "unknown"
    conversation_text = row.get("conversation_text", "")
    return f"{scenario}:{hash(conversation_text)}"


def deduplicate_latest(rows: list[dict]) -> list[dict]:
    latest_by_key: dict[str, dict] = {}
    for row in rows:
        key = conversation_key(row)
        current = latest_by_key.get(key)
        if current is None or parse_timestamp(row.get("timestamp_utc")) >= parse_timestamp(
            current.get("timestamp_utc")
        ):
            latest_by_key[key] = row
    return list(latest_by_key.values())


def fix_text(text: str | None) -> str:
    if not text:
        return ""

    cleaned = text.strip()
    replacements = {
        "Iâ€™": "I’",
        "â€™": "’",
        "â€œ": '"',
        "â€\x9d": '"',
        "â€“": "-",
        "â€”": "-",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    return cleaned


def issue_counts(rows: list[dict]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for issue in row.get("issues", []):
            counter[str(issue)] += 1
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def build_intent_candidates(rows: list[dict]) -> list[dict]:
    candidates: list[dict] = []
    for row in deduplicate_latest(rows):
        suggested_intent = row.get("suggested_intent") or row.get("final_intent")
        if not suggested_intent:
            continue

        intent_result = row.get("intent_result", {})
        candidates.append(
            {
                "conversation_id": row.get("conversation_id"),
                "scenario": row.get("scenario"),
                "conversation_text": row.get("conversation_text", ""),
                "input_text": intent_result.get("input_text", ""),
                "raw_intent": intent_result.get("predicted_intent"),
                "target_intent": suggested_intent,
                "critic_score": row.get("critic_score"),
                "issues": row.get("issues", []),
                "source": "intent_critic_failure_memory",
                "timestamp_utc": row.get("timestamp_utc"),
            }
        )
    return candidates


def build_summary_candidates(rows: list[dict]) -> list[dict]:
    candidates: list[dict] = []
    for row in deduplicate_latest(rows):
        target_summary = fix_text(row.get("final_summary") or row.get("fallback_summary"))
        if not target_summary:
            continue

        candidates.append(
            {
                "conversation_id": row.get("conversation_id"),
                "scenario": row.get("scenario"),
                "conversation_text": row.get("conversation_text", ""),
                "raw_summary": fix_text(row.get("raw_summary")),
                "target_summary": target_summary,
                "critic_score": row.get("critic_score"),
                "issues": row.get("issues", []),
                "source": "summary_critic_failure_memory",
                "timestamp_utc": row.get("timestamp_utc"),
            }
        )
    return candidates


def build_reply_candidates(rows: list[dict]) -> list[dict]:
    candidates: list[dict] = []
    for row in deduplicate_latest(rows):
        target_reply = fix_text(row.get("final_reply") or row.get("fallback_reply"))
        if not target_reply:
            continue

        intent_result = row.get("intent", {})
        candidates.append(
            {
                "conversation_id": row.get("conversation_id"),
                "scenario": row.get("scenario"),
                "conversation_text": row.get("conversation_text", ""),
                "predicted_intent": intent_result.get("predicted_intent"),
                "intent_input_text": intent_result.get("input_text", ""),
                "summary_text": fix_text(row.get("summary")),
                "raw_reply": fix_text(row.get("raw_reply")),
                "target_reply": target_reply,
                "critic_score": row.get("critic_score"),
                "issues": row.get("issues", []),
                "source": "reply_critic_failure_memory",
                "timestamp_utc": row.get("timestamp_utc"),
            }
        )
    return candidates


def build_summary_report(
    intent_rows: list[dict],
    summary_rows: list[dict],
    reply_rows: list[dict],
    intent_candidates: list[dict],
    summary_candidates: list[dict],
    reply_candidates: list[dict],
) -> dict:
    return {
        "intent": {
            "raw_failure_rows": len(intent_rows),
            "deduplicated_candidates": len(intent_candidates),
            "issue_counts": issue_counts(intent_rows),
        },
        "summary": {
            "raw_failure_rows": len(summary_rows),
            "deduplicated_candidates": len(summary_candidates),
            "issue_counts": issue_counts(summary_rows),
        },
        "reply": {
            "raw_failure_rows": len(reply_rows),
            "deduplicated_candidates": len(reply_candidates),
            "issue_counts": issue_counts(reply_rows),
        },
    }


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    intent_rows = load_jsonl(args.intent_failures_path)
    summary_rows = load_jsonl(args.summary_failures_path)
    reply_rows = load_jsonl(args.reply_failures_path)

    intent_candidates = build_intent_candidates(intent_rows)
    summary_candidates = build_summary_candidates(summary_rows)
    reply_candidates = build_reply_candidates(reply_rows)

    output_dir = args.output_dir
    intent_output_path = output_dir / "intent_retraining_candidates.jsonl"
    summary_output_path = output_dir / "summary_retraining_candidates.jsonl"
    reply_output_path = output_dir / "reply_retraining_candidates.jsonl"
    report_output_path = output_dir / "retraining_candidates_report.json"

    append_jsonl(intent_candidates, intent_output_path)
    append_jsonl(summary_candidates, summary_output_path)
    append_jsonl(reply_candidates, reply_output_path)

    report = build_summary_report(
        intent_rows=intent_rows,
        summary_rows=summary_rows,
        reply_rows=reply_rows,
        intent_candidates=intent_candidates,
        summary_candidates=summary_candidates,
        reply_candidates=reply_candidates,
    )
    save_json(report, report_output_path)

    print(f"[saved] intent candidates -> {intent_output_path}")
    print(f"[saved] summary candidates -> {summary_output_path}")
    print(f"[saved] reply candidates -> {reply_output_path}")
    print(f"[saved] candidate report -> {report_output_path}")
    print(f"[summary] intent candidates: {len(intent_candidates)}")
    print(f"[summary] summary candidates: {len(summary_candidates)}")
    print(f"[summary] reply candidates: {len(reply_candidates)}")


if __name__ == "__main__":
    main()
