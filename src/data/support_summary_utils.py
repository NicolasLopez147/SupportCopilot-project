import html
import json
import re
from pathlib import Path

import emoji

from src.copilot.critics.summary_critic import build_summary_fallback


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def clean_message_text(text: str) -> str:
    cleaned = html.unescape(text)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"@\w+", "", cleaned)
    cleaned = re.sub(r"__\w+__", "[REDACTED]", cleaned)
    cleaned = emoji.replace_emoji(cleaned, replace="")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" .\n\t")


def format_conversation(messages: list[dict]) -> str:
    formatted_lines = []

    for message in messages:
        speaker = message.get("speaker", "").strip().lower()
        text = clean_message_text(message.get("text", ""))

        if not text:
            continue

        if speaker == "customer":
            speaker_label = "Customer"
        elif speaker == "agent":
            speaker_label = "Agent"
        else:
            speaker_label = "Unknown"

        formatted_lines.append(f"{speaker_label}: {text}")

    return "\n".join(formatted_lines)


def build_support_summary_target(sample: dict) -> str:
    for key in [
        "reference_summary",
        "summary_target",
        "summary_text",
        "summary_abstractive",
        "target_text",
        "final_summary",
    ]:
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    conversation_text = sample.get("conversation_text", "")
    if not conversation_text:
        conversation_text = format_conversation(sample.get("messages", []))

    if not conversation_text:
        return ""

    return build_summary_fallback(conversation_text)
