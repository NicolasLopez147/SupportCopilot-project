from functools import lru_cache

from src.copilot.critics.summary_critic import critique_summary
from src.copilot.feedback.log_summary_failures import log_summary_failure
from src.copilot.pipeline.service import (
    MAX_SUMMARY_INPUT_TOKENS,
    MAX_SUMMARY_NEW_TOKENS,
    SUMMARY_ADAPTER_DIR,
    format_conversation,
    generate_seq2seq,
    load_peft_seq2seq,
)


class SummaryService:
    def __init__(self) -> None:
        self.model_bundle = load_peft_seq2seq(SUMMARY_ADAPTER_DIR)

    def health(self) -> dict:
        return {
            "status": "ok",
            "service": "summary-service",
            "mode": "service",
        }

    def run(self, sample: dict, persist_feedback: bool = False) -> dict:
        messages = sample.get("messages", [])
        if not messages:
            raise ValueError("Input sample does not contain messages.")

        conversation_text = format_conversation(messages)
        raw_summary = generate_seq2seq(
            tokenizer=self.model_bundle.tokenizer,
            model=self.model_bundle.model,
            device=self.model_bundle.device,
            prompt=f"summarize the following customer support conversation: {conversation_text}",
            max_input_tokens=MAX_SUMMARY_INPUT_TOKENS,
            max_new_tokens=MAX_SUMMARY_NEW_TOKENS,
        )
        summary_review = critique_summary(
            conversation_text=conversation_text,
            generated_summary=raw_summary,
        )

        if persist_feedback:
            log_summary_failure(
                conversation_id=sample.get("conversation_id"),
                scenario=sample.get("scenario"),
                conversation_text=conversation_text,
                raw_summary=raw_summary,
                summary_review=summary_review,
            )

        return {
            "conversation_id": sample.get("conversation_id"),
            "scenario": sample.get("scenario"),
            "conversation_text": conversation_text,
            "summary_raw": raw_summary,
            "summary_review": summary_review,
            "summary": summary_review["final_summary"],
        }


@lru_cache(maxsize=1)
def get_summary_service() -> SummaryService:
    return SummaryService()
