import json
from datetime import datetime, timezone
from pathlib import Path

from src.utils.paths import FEEDBACK_MEMORY_DIR


INTENT_FAILURES_PATH = FEEDBACK_MEMORY_DIR / "intent_failures.jsonl"


def append_jsonl(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_intent_failure(
    conversation_id: str | None,
    scenario: str | None,
    conversation_text: str,
    intent_result: dict,
    intent_review: dict,
) -> None:
    if intent_review.get("passed", True):
        return

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "conversation_id": conversation_id,
        "scenario": scenario,
        "conversation_text": conversation_text,
        "intent_result": intent_result,
        "critic_score": intent_review.get("score"),
        "issues": intent_review.get("issues", []),
        "suggested_intent": intent_review.get("suggested_intent"),
        "final_intent": intent_review.get("final_intent"),
        "used_fallback": intent_review.get("used_fallback"),
    }

    append_jsonl(record, INTENT_FAILURES_PATH)
