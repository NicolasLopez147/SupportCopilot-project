from functools import lru_cache

from src.copilot.critics.reply_critic import critique_reply
from src.copilot.feedback.log_reply_failures import log_reply_failure
from src.copilot.pipeline.service import (
    MAX_REPLY_INPUT_TOKENS,
    MAX_REPLY_NEW_TOKENS,
    REPLY_ADAPTER_DIR,
    format_conversation,
    generate_seq2seq,
    load_peft_seq2seq,
)


class ReplyService:
    def __init__(self) -> None:
        self.model_bundle = load_peft_seq2seq(REPLY_ADAPTER_DIR)

    def health(self) -> dict:
        return {
            "status": "ok",
            "service": "reply-service",
            "mode": "service",
        }

    def run(self, sample: dict, persist_feedback: bool = False) -> dict:
        messages = sample.get("messages", [])
        if not messages:
            raise ValueError("Input sample does not contain messages.")

        predicted_intent = str(sample.get("predicted_intent", "")).strip()
        summary_text = str(sample.get("summary_text", "")).strip()
        if not predicted_intent:
            raise ValueError("Reply service requires a predicted_intent.")
        if not summary_text:
            raise ValueError("Reply service requires a summary_text.")

        conversation_text = format_conversation(messages)
        raw_reply = generate_seq2seq(
            tokenizer=self.model_bundle.tokenizer,
            model=self.model_bundle.model,
            device=self.model_bundle.device,
            prompt=(
                "write the next professional support agent reply based on this conversation: "
                f"{conversation_text}"
            ),
            max_input_tokens=MAX_REPLY_INPUT_TOKENS,
            max_new_tokens=MAX_REPLY_NEW_TOKENS,
        )
        reply_review = critique_reply(
            conversation_text=conversation_text,
            predicted_intent=predicted_intent,
            generated_reply=raw_reply,
            summary_text=summary_text,
        )

        intent_result = {"predicted_intent": predicted_intent}
        if persist_feedback:
            log_reply_failure(
                conversation_id=sample.get("conversation_id"),
                scenario=sample.get("scenario"),
                conversation_text=conversation_text,
                intent_result=intent_result,
                summary_text=summary_text,
                raw_reply=raw_reply,
                reply_review=reply_review,
            )

        return {
            "conversation_id": sample.get("conversation_id"),
            "scenario": sample.get("scenario"),
            "conversation_text": conversation_text,
            "suggested_reply_raw": raw_reply,
            "reply_review": reply_review,
            "suggested_reply": reply_review["final_reply"],
        }


@lru_cache(maxsize=1)
def get_reply_service() -> ReplyService:
    return ReplyService()
