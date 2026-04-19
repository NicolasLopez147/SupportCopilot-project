from functools import lru_cache

from src.copilot.critics.intent_critic import critique_intent
from src.copilot.feedback.log_intent_failures import log_intent_failure
from src.copilot.pipeline.service import (
    format_conversation,
    load_intent_components,
    predict_intent,
)


class IntentService:
    def __init__(self) -> None:
        self.encoder, self.classifier = load_intent_components()

    def health(self) -> dict:
        return {
            "status": "ok",
            "service": "intent-service",
            "mode": "service",
        }

    def run(self, sample: dict, persist_feedback: bool = False) -> dict:
        messages = sample.get("messages", [])
        if not messages:
            raise ValueError("Input sample does not contain messages.")

        conversation_text = format_conversation(messages)
        raw_intent_result = predict_intent(messages, self.encoder, self.classifier)
        intent_review = critique_intent(raw_intent_result)
        intent_result = {
            **raw_intent_result,
            "predicted_intent": intent_review["final_intent"],
        }

        if persist_feedback:
            log_intent_failure(
                conversation_id=sample.get("conversation_id"),
                scenario=sample.get("scenario"),
                conversation_text=conversation_text,
                intent_result=raw_intent_result,
                intent_review=intent_review,
            )

        return {
            "conversation_id": sample.get("conversation_id"),
            "scenario": sample.get("scenario"),
            "conversation_text": conversation_text,
            "intent_raw": raw_intent_result,
            "intent_review": intent_review,
            "intent": intent_result,
        }


@lru_cache(maxsize=1)
def get_intent_service() -> IntentService:
    return IntentService()
