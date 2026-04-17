import re


CONFIDENCE_THRESHOLD = 0.45
MARGIN_THRESHOLD = 0.12
SCENARIO_KEYWORDS = {
    "billing_vs_technical_routing": [
        "bill",
        "billing",
        "invoice",
        "package",
        "plan",
        "price",
        "charged",
        "speed",
        "slow",
    ],
    "fiber_box_red_light": [
        "fiber",
        "red light",
        "red",
        "box",
        "no internet",
        "router",
    ],
    "identity_verification": [
        "verify",
        "verification",
        "account",
        "identity",
        "email",
        "phone number",
        "contact details",
    ],
    "local_outage_check": [
        "outage",
        "area",
        "incident",
        "service down",
        "line",
        "network down",
    ],
    "router_restart_procedure": [
        "restart",
        "reboot",
        "router",
        "modem",
        "turn it off",
        "power cycle",
    ],
    "slow_connection_diagnosis": [
        "slow",
        "buffering",
        "speed",
        "freezing",
        "wifi",
        "download",
        "streaming",
    ],
    "technical_intervention_request": [
        "technician",
        "intervention",
        "visit",
        "repair",
        "send someone",
        "advanced support",
    ],
    "wifi_instability": [
        "wifi",
        "disconnect",
        "unstable",
        "drops",
        "signal",
        "wireless",
    ],
}


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def keyword_scores(intent_text: str) -> dict[str, int]:
    normalized = normalize_text(intent_text)
    scores = {}
    for label, keywords in SCENARIO_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in normalized:
                score += 1
        scores[label] = score
    return scores


def critique_intent(intent_result: dict) -> dict:
    predicted_intent = intent_result.get("predicted_intent")
    confidence = intent_result.get("confidence")
    top_classes = intent_result.get("top_classes", [])
    intent_text = intent_result.get("input_text", "")

    issues = []
    score = 1.0

    if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
        issues.append("low_confidence")
        score -= 0.25

    if len(top_classes) >= 2:
        margin = round(top_classes[0]["score"] - top_classes[1]["score"], 4)
        if margin < MARGIN_THRESHOLD:
            issues.append("ambiguous_top_classes")
            score -= 0.2
    else:
        margin = None

    keyword_match_scores = keyword_scores(intent_text)
    keyword_best_label = max(keyword_match_scores, key=keyword_match_scores.get)
    keyword_best_score = keyword_match_scores[keyword_best_label]
    predicted_keyword_score = keyword_match_scores.get(predicted_intent, 0)

    if keyword_best_score > 0 and keyword_best_label != predicted_intent and predicted_keyword_score == 0:
        issues.append("keyword_mismatch")
        score -= 0.25
        suggested_intent = keyword_best_label
    else:
        suggested_intent = predicted_intent

    passed = score >= 0.6 and "keyword_mismatch" not in issues

    return {
        "passed": passed,
        "score": round(max(score, 0.0), 4),
        "issues": issues,
        "predicted_intent": predicted_intent,
        "suggested_intent": suggested_intent,
        "confidence": confidence,
        "margin_top1_top2": margin,
        "keyword_best_label": keyword_best_label,
        "keyword_scores": keyword_match_scores,
        "final_intent": predicted_intent if passed else suggested_intent,
        "used_fallback": predicted_intent != suggested_intent and not passed,
    }
