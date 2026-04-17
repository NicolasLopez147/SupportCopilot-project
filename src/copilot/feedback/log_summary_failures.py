import json
from datetime import datetime, timezone
from pathlib import Path

from src.utils.paths import FEEDBACK_MEMORY_DIR


SUMMARY_FAILURES_PATH = FEEDBACK_MEMORY_DIR / "summary_failures.jsonl"


def append_jsonl(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_summary_failure(
    conversation_id: str | None,
    scenario: str | None,
    conversation_text: str,
    raw_summary: str,
    summary_review: dict,
) -> None:
    if summary_review.get("passed", True):
        return

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "conversation_id": conversation_id,
        "scenario": scenario,
        "conversation_text": conversation_text,
        "raw_summary": raw_summary,
        "critic_score": summary_review.get("score"),
        "issues": summary_review.get("issues", []),
        "fallback_summary": summary_review.get("fallback_summary"),
        "final_summary": summary_review.get("final_summary"),
        "used_fallback": summary_review.get("used_fallback"),
    }

    append_jsonl(record, SUMMARY_FAILURES_PATH)
