import json
from datetime import datetime, timezone
from pathlib import Path

from src.utils.paths import FEEDBACK_MEMORY_DIR


REPLY_FAILURES_PATH = FEEDBACK_MEMORY_DIR / "reply_failures.jsonl"


def append_jsonl(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_reply_failure(
    conversation_id: str | None,
    scenario: str | None,
    conversation_text: str,
    intent_result: dict,
    summary_text: str,
    raw_reply: str,
    reply_review: dict,
) -> None:
    if reply_review.get("passed", True):
        return

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "conversation_id": conversation_id,
        "scenario": scenario,
        "conversation_text": conversation_text,
        "intent": intent_result,
        "summary": summary_text,
        "raw_reply": raw_reply,
        "critic_score": reply_review.get("score"),
        "issues": reply_review.get("issues", []),
        "fallback_reply": reply_review.get("fallback_reply"),
        "final_reply": reply_review.get("final_reply"),
        "used_fallback": reply_review.get("used_fallback"),
    }

    append_jsonl(record, REPLY_FAILURES_PATH)
