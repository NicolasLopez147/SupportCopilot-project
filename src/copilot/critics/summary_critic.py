import re


SUMMARY_BAD_PATTERNS = [
    re.compile(r"\bcustomer complains that the customer service person\b", re.IGNORECASE),
    re.compile(r"\bthe customer service person\b", re.IGNORECASE),
]

GENERIC_AGENT_PATTERNS = [
    re.compile(r"\bagent (is|was) happy to help\b", re.IGNORECASE),
    re.compile(r"\bagent would be happy to help\b", re.IGNORECASE),
    re.compile(r"\bagent will be happy to help\b", re.IGNORECASE),
    re.compile(r"\bagent will help (him|her|them) figure out what is going on\b", re.IGNORECASE),
    re.compile(r"\bagent asks to help\b", re.IGNORECASE),
]

GENERIC_CUSTOMER_PATTERNS = [
    re.compile(r"\bcustomer has a question about (his|her|their) account\b", re.IGNORECASE),
    re.compile(r"\bcustomer is having some issues with (his|her|their) account\b", re.IGNORECASE),
    re.compile(r"\bcustomer is having some issues with (his|her|their) plan\b", re.IGNORECASE),
]

ACTION_VERBS = {
    "ask",
    "asked",
    "asking",
    "check",
    "checking",
    "checked",
    "route",
    "routing",
    "review",
    "reviewing",
    "verify",
    "verifying",
    "confirm",
    "confirming",
    "restart",
    "restarting",
    "investigate",
    "investigating",
    "troubleshoot",
    "troubleshooting",
    "open",
    "opening",
    "escalate",
    "escalating",
    "determine",
    "determining",
    "continue",
    "continuing",
}


def normalize_text(text: str) -> str:
    normalized = text.strip()
    replacements = {
        "Ã¢â‚¬â„¢": "'",
        "Ã¢â‚¬Ëœ": "'",
        "Ã¢â‚¬Å“": '"',
        "Ã¢â‚¬Â": '"',
        "Ã¢â‚¬â€œ": "-",
        "Ã¢â‚¬â€": "-",
        "\u00a0": " ",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def extract_customer_and_agent_signals(conversation_text: str) -> tuple[list[str], list[str]]:
    customer_parts = []
    agent_parts = []
    for line in conversation_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Customer:"):
            customer_parts.append(stripped.replace("Customer:", "", 1).strip())
        elif stripped.startswith("Agent:"):
            agent_parts.append(stripped.replace("Agent:", "", 1).strip())
    return customer_parts, agent_parts


def build_summary_fallback(conversation_text: str) -> str:
    customer_parts, agent_parts = extract_customer_and_agent_signals(conversation_text)
    customer_text = customer_parts[0] if customer_parts else ""
    agent_text = agent_parts[-1] if agent_parts else ""

    if customer_text and agent_text:
        return (
            f"Customer reports that {customer_text.lower()}. "
            f"Agent responds that {agent_text.lower()}."
        )
    if customer_text:
        return f"Customer reports that {customer_text.lower()}."
    return "Customer requests support and the agent begins troubleshooting."


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def repeated_bigram_ratio(text: str) -> float:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    if len(tokens) < 4:
        return 0.0

    bigrams = list(zip(tokens, tokens[1:]))
    if not bigrams:
        return 0.0

    unique_bigrams = set(bigrams)
    return 1 - (len(unique_bigrams) / len(bigrams))


def contains_action_verb(text: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9']+", text.lower()))
    return bool(tokens & ACTION_VERBS)


def critique_summary(conversation_text: str, generated_summary: str) -> dict:
    summary = normalize_text(generated_summary)
    issues = []
    score = 1.0

    if not summary:
        issues.append("empty_summary")
        score -= 0.7

    word_count = len(summary.split())
    if word_count < 8:
        issues.append("too_short")
        score -= 0.2
    if word_count > 60:
        issues.append("too_long")
        score -= 0.15

    lowered_summary = summary.lower()
    if "customer" not in lowered_summary:
        issues.append("missing_customer_reference")
        score -= 0.15
    if "agent" not in lowered_summary:
        issues.append("missing_agent_reference")
        score -= 0.15

    if any(pattern.search(summary) for pattern in SUMMARY_BAD_PATTERNS):
        issues.append("awkward_phrasing")
        score -= 0.25

    if any(pattern.search(summary) for pattern in GENERIC_AGENT_PATTERNS):
        issues.append("generic_agent_action")
        score -= 0.2

    if any(pattern.search(summary) for pattern in GENERIC_CUSTOMER_PATTERNS):
        issues.append("generic_customer_problem")
        score -= 0.2

    customer_parts, agent_parts = extract_customer_and_agent_signals(conversation_text)
    joined_customer = " ".join(customer_parts).lower()
    joined_agent = " ".join(agent_parts).lower()
    summary_sentences = split_sentences(summary)

    if repeated_bigram_ratio(summary) > 0.22:
        issues.append("redundant_summary")
        score -= 0.15

    if joined_customer:
        overlap_customer = lexical_overlap_ratio(summary, joined_customer)
        if overlap_customer < 0.12:
            issues.append("low_customer_coverage")
            score -= 0.15
        elif overlap_customer < 0.18 and word_count < 14:
            issues.append("low_specificity")
            score -= 0.15

    if joined_agent:
        overlap_agent = lexical_overlap_ratio(summary, joined_agent)
        if overlap_agent < 0.08:
            issues.append("low_agent_coverage")
            score -= 0.15

    if len(summary_sentences) >= 2:
        agent_sentence = summary_sentences[-1]
        if "agent" in agent_sentence.lower() and not contains_action_verb(agent_sentence):
            issues.append("weak_agent_action")
            score -= 0.2

    fallback_summary = build_summary_fallback(conversation_text)
    blocking_issues = {
        "empty_summary",
        "awkward_phrasing",
        "generic_agent_action",
        "generic_customer_problem",
        "weak_agent_action",
    }
    passed = score >= 0.6 and not (blocking_issues & set(issues))
    final_summary = summary if passed else fallback_summary

    return {
        "passed": passed,
        "score": round(max(score, 0.0), 4),
        "issues": issues,
        "fallback_summary": fallback_summary,
        "final_summary": final_summary,
        "used_fallback": final_summary != summary,
    }


def lexical_overlap_ratio(text_a: str, text_b: str) -> float:
    tokens_a = set(re.findall(r"[a-z0-9']+", text_a.lower()))
    tokens_b = set(re.findall(r"[a-z0-9']+", text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(1, len(tokens_a))
