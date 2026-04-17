import re
from pathlib import Path

from src.utils.paths import DATA_DIR


KB_DIR = DATA_DIR / "kb"
CRITICAL_PATTERNS = [
    re.compile(r"\bi'?m not sure\b", re.IGNORECASE),
    re.compile(r"\bi don't know\b", re.IGNORECASE),
    re.compile(r"\bi can't help\b", re.IGNORECASE),
]
ACTION_PATTERNS = [
    re.compile(r"\bnext step\b", re.IGNORECASE),
    re.compile(r"\bI will\b", re.IGNORECASE),
    re.compile(r"\bwe will\b", re.IGNORECASE),
    re.compile(r"\bwe need to\b", re.IGNORECASE),
    re.compile(r"\bplease\b", re.IGNORECASE),
    re.compile(r"\bcheck\b", re.IGNORECASE),
    re.compile(r"\bconfirm\b", re.IGNORECASE),
    re.compile(r"\broute\b", re.IGNORECASE),
]


def load_kb_documents() -> dict[str, str]:
    documents = {}
    for path in KB_DIR.glob("*.md"):
        documents[path.stem] = path.read_text(encoding="utf-8").strip()
    return documents


def normalize_text(text: str) -> str:
    normalized = text.strip()
    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "\u00a0": " ",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def has_repeated_sentence(text: str) -> bool:
    sentences = split_sentences(text.lower())
    if len(sentences) < 2:
        return False
    return len(sentences) != len(set(sentences))


def extract_suggested_reply(kb_text: str) -> str | None:
    pattern = re.compile(
        r"## Suggested Reply\s*(.+?)(?:\n## |\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(kb_text)
    if not match:
        return None
    return normalize_text(match.group(1))


def extract_customer_text(conversation_text: str) -> str:
    customer_lines = []
    for line in conversation_text.splitlines():
        if line.startswith("Customer:"):
            customer_lines.append(line.replace("Customer:", "", 1).strip())
    return " ".join(customer_lines).strip()


def lexical_overlap_ratio(reply: str, customer_text: str) -> float:
    reply_tokens = set(re.findall(r"[a-z0-9']+", reply.lower()))
    customer_tokens = set(re.findall(r"[a-z0-9']+", customer_text.lower()))
    if not reply_tokens or not customer_tokens:
        return 0.0
    return len(reply_tokens & customer_tokens) / max(1, len(reply_tokens))


def build_kb_fallback(predicted_intent: str, kb_documents: dict[str, str]) -> str | None:
    kb_text = kb_documents.get(predicted_intent)
    if not kb_text:
        return None
    return extract_suggested_reply(kb_text)


def critique_reply(
    conversation_text: str,
    predicted_intent: str,
    generated_reply: str,
    summary_text: str | None = None,
) -> dict:
    kb_documents = load_kb_documents()
    reply = normalize_text(generated_reply)
    customer_text = extract_customer_text(conversation_text)

    issues = []
    score = 1.0

    if not reply:
        issues.append("empty_reply")
        score -= 0.6

    if len(reply.split()) < 6:
        issues.append("too_short")
        score -= 0.2

    if has_repeated_sentence(reply):
        issues.append("repetition")
        score -= 0.35

    if any(pattern.search(reply) for pattern in CRITICAL_PATTERNS):
        issues.append("uncertain_or_unhelpful")
        score -= 0.35

    if not any(pattern.search(reply) for pattern in ACTION_PATTERNS):
        issues.append("missing_next_step")
        score -= 0.2

    overlap_ratio = lexical_overlap_ratio(reply, customer_text)
    if overlap_ratio > 0.75:
        issues.append("too_close_to_customer_wording")
        score -= 0.2

    if summary_text:
        summary_overlap = lexical_overlap_ratio(reply, summary_text)
        if summary_overlap > 0.9 and len(reply.split()) < 14:
            issues.append("too_close_to_summary")
            score -= 0.1

    kb_fallback = build_kb_fallback(predicted_intent, kb_documents)

    passed = score >= 0.6 and not {"empty_reply", "repetition", "uncertain_or_unhelpful"} & set(issues)
    final_reply = reply if passed else (kb_fallback or reply)

    return {
        "passed": passed,
        "score": round(max(score, 0.0), 4),
        "issues": issues,
        "fallback_reply": kb_fallback,
        "final_reply": final_reply,
        "used_fallback": final_reply != reply,
    }
