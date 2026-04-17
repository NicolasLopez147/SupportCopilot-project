import argparse
import html
import json
import re
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full SupportCopilot pipeline on a conversation."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "Path to a JSONL dataset or a single JSON conversation object. "
            f"Default: {DEFAULT_INPUT_PATH}"
        ),
    )
    parser.add_argument(
        "--conversation-id",
        default=None,
        help="Optional conversation_id to select from a JSONL file. Defaults to the first sample.",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Process all conversations in a JSONL file instead of just one sample.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap when using --run-all.",
    )
    parser.add_argument(
        "--reset-feedback",
        action="store_true",
        help="Clear intent/summary/reply failure memory before running the pipeline batch.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to save pipeline output. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_single_sample(path: Path, conversation_id: str | None) -> dict:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")

    if conversation_id is None:
        return rows[0]

    for row in rows:
        if row.get("conversation_id") == conversation_id:
            return row

    raise KeyError(f"conversation_id '{conversation_id}' not found in {path}")


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


def load_peft_seq2seq(adapter_dir: Path) -> tuple[AutoTokenizer, PeftModel, str]:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_SEQ2SEQ_MODEL)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


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


def generate_summary(conversation_text: str) -> str:
    tokenizer, model, device = load_peft_seq2seq(SUMMARY_ADAPTER_DIR)
    prompt = f"summarize the following customer support conversation: {conversation_text}"
    return generate_seq2seq(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt=prompt,
        max_input_tokens=MAX_SUMMARY_INPUT_TOKENS,
        max_new_tokens=MAX_SUMMARY_NEW_TOKENS,
    )


def generate_reply(conversation_text: str) -> str:
    tokenizer, model, device = load_peft_seq2seq(REPLY_ADAPTER_DIR)
    prompt = f"write the next professional support agent reply based on this conversation: {conversation_text}"
    return generate_seq2seq(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt=prompt,
        max_input_tokens=MAX_REPLY_INPUT_TOKENS,
        max_new_tokens=MAX_REPLY_NEW_TOKENS,
    )


def save_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def reset_feedback_memory() -> None:
    for path in [INTENT_FAILURES_PATH, SUMMARY_FAILURES_PATH, REPLY_FAILURES_PATH]:
        if path.exists():
            path.unlink()


def process_sample(
    sample: dict,
    intent_encoder: SentenceTransformer,
    intent_classifier: object,
    summary_tokenizer: AutoTokenizer,
    summary_model: PeftModel,
    summary_device: str,
    reply_tokenizer: AutoTokenizer,
    reply_model: PeftModel,
    reply_device: str,
) -> dict:
    messages = sample.get("messages", [])
    if not messages:
        raise ValueError("Input sample does not contain messages.")

    conversation_text = format_conversation(messages)

    raw_intent_result = predict_intent(messages, intent_encoder, intent_classifier)
    intent_review = critique_intent(raw_intent_result)
    intent_result = {
        **raw_intent_result,
        "predicted_intent": intent_review["final_intent"],
    }

    raw_summary_result = generate_seq2seq(
        tokenizer=summary_tokenizer,
        model=summary_model,
        device=summary_device,
        prompt=f"summarize the following customer support conversation: {conversation_text}",
        max_input_tokens=MAX_SUMMARY_INPUT_TOKENS,
        max_new_tokens=MAX_SUMMARY_NEW_TOKENS,
    )
    summary_review = critique_summary(
        conversation_text=conversation_text,
        generated_summary=raw_summary_result,
    )

    raw_reply_result = generate_seq2seq(
        tokenizer=reply_tokenizer,
        model=reply_model,
        device=reply_device,
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


def main() -> None:
    args = parse_args()
    samples = load_samples(
        path=args.input_path,
        conversation_id=args.conversation_id,
        run_all=args.run_all,
        limit=args.limit,
    )

    if args.reset_feedback:
        reset_feedback_memory()
        print("[pipeline] feedback memory cleared")

    print("[pipeline] loading intent components")
    intent_encoder, intent_classifier = load_intent_components()

    print("[pipeline] loading summary model")
    summary_tokenizer, summary_model, summary_device = load_peft_seq2seq(SUMMARY_ADAPTER_DIR)

    print("[pipeline] loading reply model")
    reply_tokenizer, reply_model, reply_device = load_peft_seq2seq(REPLY_ADAPTER_DIR)

    results = []
    for index, sample in enumerate(samples, start=1):
        conversation_id = sample.get("conversation_id")
        print(f"[pipeline] sample {index}/{len(samples)} -> {conversation_id}")

        output = process_sample(
            sample=sample,
            intent_encoder=intent_encoder,
            intent_classifier=intent_classifier,
            summary_tokenizer=summary_tokenizer,
            summary_model=summary_model,
            summary_device=summary_device,
            reply_tokenizer=reply_tokenizer,
            reply_model=reply_model,
            reply_device=reply_device,
        )
        results.append(output)

        print(f"[intent_final] {output['intent']['predicted_intent']}")
        print(f"[summary_passed] {output['summary_review']['passed']}")
        print(f"[reply_passed] {output['reply_review']['passed']}")

    output_payload: dict | list
    if len(results) == 1:
        output_payload = results[0]
    else:
        output_payload = results

    save_json(output_payload, args.output_path)
    print(f"[saved] pipeline output -> {args.output_path}")

    intent_failures = sum(1 for item in results if not item["intent_review"]["passed"])
    summary_failures = sum(1 for item in results if not item["summary_review"]["passed"])
    reply_failures = sum(1 for item in results if not item["reply_review"]["passed"])

    print(f"[summary] processed samples: {len(results)}")
    print(f"[summary] intent critic failures: {intent_failures}")
    print(f"[summary] summary critic failures: {summary_failures}")
    print(f"[summary] reply critic failures: {reply_failures}")


if __name__ == "__main__":
    main()
