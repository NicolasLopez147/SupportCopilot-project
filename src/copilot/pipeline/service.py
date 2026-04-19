import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

import emoji
import joblib
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.copilot.critics.intent_critic import critique_intent
from src.copilot.critics.reply_critic import critique_reply
from src.copilot.critics.summary_critic import critique_summary
from src.copilot.feedback.log_intent_failures import INTENT_FAILURES_PATH, log_intent_failure
from src.copilot.feedback.log_reply_failures import REPLY_FAILURES_PATH, log_reply_failure
from src.copilot.feedback.log_summary_failures import SUMMARY_FAILURES_PATH, log_summary_failure
from src.utils.paths import COPILOT_OUTPUT_DIR, EXPERIMENTS_OUTPUT_DIR, PROJECT_ROOT


INTENT_ARTIFACT_PATH = EXPERIMENTS_OUTPUT_DIR / "intent" / "synthetic_embedding" / "intent_synthetic_model.joblib"
SUMMARY_ADAPTER_DIR = EXPERIMENTS_OUTPUT_DIR / "summary" / "lora_base" / "final_model"
REPLY_ADAPTER_DIR = EXPERIMENTS_OUTPUT_DIR / "reply" / "lora_base" / "final_model"
BASE_SEQ2SEQ_MODEL = "google/flan-t5-small"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "synthetic" / "reply_test.jsonl"
DEFAULT_OUTPUT_PATH = COPILOT_OUTPUT_DIR / "runtime" / "pipeline_output.json"
MAX_SUMMARY_INPUT_TOKENS = 512
MAX_SUMMARY_NEW_TOKENS = 128
MAX_REPLY_INPUT_TOKENS = 512
MAX_REPLY_NEW_TOKENS = 128


@dataclass
class LoadedSeq2SeqModel:
    tokenizer: AutoTokenizer
    model: PeftModel
    device: str


class SupportCopilotService:
    def __init__(self) -> None:
        self.intent_encoder, self.intent_classifier = load_intent_components()
        self.summary_model = load_peft_seq2seq(SUMMARY_ADAPTER_DIR)
        self.reply_model = load_peft_seq2seq(REPLY_ADAPTER_DIR)

    def run_sample(self, sample: dict, persist_feedback: bool = True) -> dict:
        messages = sample.get("messages", [])
        if not messages:
            raise ValueError("Input sample does not contain messages.")

        conversation_text = format_conversation(messages)

        raw_intent_result = predict_intent(messages, self.intent_encoder, self.intent_classifier)
        intent_review = critique_intent(raw_intent_result)
        intent_result = {
            **raw_intent_result,
            "predicted_intent": intent_review["final_intent"],
        }

        raw_summary_result = generate_seq2seq(
            tokenizer=self.summary_model.tokenizer,
            model=self.summary_model.model,
            device=self.summary_model.device,
            prompt=f"summarize the following customer support conversation: {conversation_text}",
            max_input_tokens=MAX_SUMMARY_INPUT_TOKENS,
            max_new_tokens=MAX_SUMMARY_NEW_TOKENS,
        )
        summary_review = critique_summary(
            conversation_text=conversation_text,
            generated_summary=raw_summary_result,
        )

        raw_reply_result = generate_seq2seq(
            tokenizer=self.reply_model.tokenizer,
            model=self.reply_model.model,
            device=self.reply_model.device,
            prompt=(
                "write the next professional support agent reply based on this conversation: "
                f"{conversation_text}"
            ),
            max_input_tokens=MAX_REPLY_INPUT_TOKENS,
            max_new_tokens=MAX_REPLY_NEW_TOKENS,
        )
        reply_review = critique_reply(
            conversation_text=conversation_text,
            predicted_intent=intent_result["predicted_intent"],
            generated_reply=raw_reply_result,
            summary_text=summary_review["final_summary"],
        )

        output = {
            "conversation_id": sample.get("conversation_id"),
            "scenario": sample.get("scenario"),
            "conversation_text": conversation_text,
            "intent_raw": raw_intent_result,
            "intent_review": intent_review,
            "intent": intent_result,
            "summary_raw": raw_summary_result,
            "summary_review": summary_review,
            "summary": summary_review["final_summary"],
            "suggested_reply_raw": raw_reply_result,
            "reply_review": reply_review,
            "suggested_reply": reply_review["final_reply"],
        }

        if persist_feedback:
            log_intent_failure(
                conversation_id=sample.get("conversation_id"),
                scenario=sample.get("scenario"),
                conversation_text=conversation_text,
                intent_result=raw_intent_result,
                intent_review=intent_review,
            )

            log_summary_failure(
                conversation_id=sample.get("conversation_id"),
                scenario=sample.get("scenario"),
                conversation_text=conversation_text,
                raw_summary=raw_summary_result,
                summary_review=summary_review,
            )

            log_reply_failure(
                conversation_id=sample.get("conversation_id"),
                scenario=sample.get("scenario"),
                conversation_text=conversation_text,
                intent_result=intent_result,
                summary_text=summary_review["final_summary"],
                raw_reply=raw_reply_result,
                reply_review=reply_review,
            )

        return output

    def run_samples(self, samples: list[dict], persist_feedback: bool = True) -> list[dict]:
        return [self.run_sample(sample, persist_feedback=persist_feedback) for sample in samples]

    def health(self) -> dict:
        return {
            "status": "ok",
            "intent_artifact_exists": INTENT_ARTIFACT_PATH.exists(),
            "summary_adapter_exists": SUMMARY_ADAPTER_DIR.exists(),
            "reply_adapter_exists": REPLY_ADAPTER_DIR.exists(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_samples(
    path: Path,
    conversation_id: str | None = None,
    run_all: bool = False,
    limit: int | None = None,
) -> list[dict]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            sample = json.load(f)
        return [sample]

    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")

    if conversation_id is not None:
        selected = [row for row in rows if row.get("conversation_id") == conversation_id]
        if not selected:
            raise KeyError(f"conversation_id '{conversation_id}' not found in {path}")
        rows = selected
    elif not run_all:
        rows = [rows[0]]

    if limit is not None:
        rows = rows[:limit]

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
        speaker = str(message.get("speaker", "")).strip().lower()
        text = clean_message_text(message.get("text", ""))
        if not text:
            continue
        speaker_label = "Customer" if speaker == "customer" else "Agent"
        formatted_lines.append(f"{speaker_label}: {text}")
    return "\n".join(formatted_lines)


def extract_intent_text(messages: list[dict]) -> str:
    for message in messages:
        speaker = str(message.get("speaker", "")).strip().lower()
        if speaker == "customer":
            text = clean_message_text(message.get("text", ""))
            if text:
                return text

    if messages:
        return clean_message_text(messages[0].get("text", ""))
    return ""


def load_intent_components() -> tuple[SentenceTransformer, object]:
    if not INTENT_ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Intent model artifact not found at {INTENT_ARTIFACT_PATH}. "
            "Run `python -m src.experiments.baselines.train_synthetic_intent` to save it."
        )

    artifact = joblib.load(INTENT_ARTIFACT_PATH)
    encoder_name = artifact["encoder_name"]
    classifier = artifact["classifier"]
    try:
        encoder = SentenceTransformer(encoder_name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load intent encoder '{encoder_name}' from local cache. "
            "Make sure the embedding model is available locally before running the pipeline batch."
        ) from exc
    return encoder, classifier


def predict_intent(
    messages: list[dict],
    encoder: SentenceTransformer,
    classifier: object,
) -> dict:
    intent_text = extract_intent_text(messages)
    if not intent_text:
        raise ValueError("Could not extract text for intent prediction.")

    embedding = encoder.encode([intent_text], show_progress_bar=False)
    label = classifier.predict(embedding)[0]

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(embedding)[0]
        class_names = classifier.classes_.tolist()
        best_idx = int(probabilities.argmax())
        confidence = float(probabilities[best_idx])
        top_classes = sorted(
            [
                {"label": class_names[idx], "score": round(float(score), 4)}
                for idx, score in enumerate(probabilities)
            ],
            key=lambda item: item["score"],
            reverse=True,
        )[:5]
    else:
        confidence = None
        top_classes = []

    return {
        "input_text": intent_text,
        "predicted_intent": label,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "top_classes": top_classes,
    }


def load_peft_seq2seq(adapter_dir: Path) -> LoadedSeq2SeqModel:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_SEQ2SEQ_MODEL)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return LoadedSeq2SeqModel(tokenizer=tokenizer, model=model, device=device)


def generate_seq2seq(
    tokenizer: AutoTokenizer,
    model: PeftModel,
    device: str,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    decoded = re.sub(r"^(Agent|Customer)\s*:\s*", "", decoded, flags=re.IGNORECASE)
    return decoded.strip()


def save_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def reset_feedback_memory() -> None:
    for path in [INTENT_FAILURES_PATH, SUMMARY_FAILURES_PATH, REPLY_FAILURES_PATH]:
        if path.exists():
            path.unlink()
