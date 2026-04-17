import html
import json
import re
from pathlib import Path

import emoji
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils.paths import EXPERIMENTS_OUTPUT_DIR, INTERIM_DATA_DIR


MODEL_NAME = "philschmid/bart-large-cnn-samsum"
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 80
NUM_TEST_SAMPLES = 10


def load_jsonl(path: Path) -> list[dict]:
    samples = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    return samples


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


def ensure_output_dir() -> Path:
    output_dir = EXPERIMENTS_OUTPUT_DIR / "summary" / "api_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(data: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def create_prompt(conversation_text: str) -> str:
    return f"Summarize this customer support conversation:\n{conversation_text}"


def build_summary_inputs(samples: list[dict]) -> list[dict]:
    prepared_samples = []

    for sample in samples:
        conversation_id = sample.get("conversation_id")
        messages = sample.get("messages", [])
        reference_summary = sample.get("summary_abstractive")

        if not conversation_id or not messages or not reference_summary:
            continue

        conversation_text = format_conversation(messages)
        if not conversation_text:
            continue

        prepared_samples.append(
            {
                "conversation_id": conversation_id,
                "conversation_text": conversation_text,
                "prompt": create_prompt(conversation_text),
                "reference_summary": reference_summary,
            }
        )

    return prepared_samples


def load_model_and_tokenizer() -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM, str]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_summary(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: str,
    prompt: str,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()


def main() -> None:
    train_path = INTERIM_DATA_DIR / "tweetsum" / "train.jsonl"
    test_path = INTERIM_DATA_DIR / "tweetsum" / "test.jsonl"
    valid_path = INTERIM_DATA_DIR / "tweetsum" / "valid.jsonl"

    train_samples = load_jsonl(train_path)
    test_samples = load_jsonl(test_path)
    valid_samples = load_jsonl(valid_path)

    train_inputs = build_summary_inputs(train_samples)
    test_inputs = build_summary_inputs(test_samples)
    valid_inputs = build_summary_inputs(valid_samples)

    print(f"[data] train samples: {len(train_inputs)}")
    print(f"[data] test samples: {len(test_inputs)}")
    print(f"[data] validation samples: {len(valid_inputs)}")

    output_dir = ensure_output_dir()

    if test_inputs:
        save_json(test_inputs[0], output_dir / "sample_test_input.json")
        print(f"[saved] sample test input -> {output_dir / 'sample_test_input.json'}")

        print("\n[preview] prompt example:\n")
        print(test_inputs[0]["prompt"])
        print("[preview] reference summary:\n")
        print(test_inputs[0]["reference_summary"])

    subset = test_inputs[:NUM_TEST_SAMPLES]
    results = []

    print(f"\n[model] loading seq2seq summarization model: {MODEL_NAME}")
    tokenizer, model, device = load_model_and_tokenizer()
    print(f"[model] running on device: {device}")
    print(f"[generate] running local summarization on {len(subset)} test samples with {MODEL_NAME}")

    for index, sample in enumerate(subset, start=1):
        print(f"[generate] sample {index}/{len(subset)} -> {sample['conversation_id']}")
        predicted_summary = generate_summary(model, tokenizer, device, sample["prompt"])

        results.append(
            {
                "conversation_id": sample["conversation_id"],
                "conversation_text": sample["conversation_text"],
                "prompt": sample["prompt"],
                "reference_summary": sample["reference_summary"],
                "predicted_summary": predicted_summary,
                "model": MODEL_NAME,
            }
        )

    if results:
        save_json(results, output_dir / "test_predictions.json")
        print(f"[saved] predictions -> {output_dir / 'test_predictions.json'}")


if __name__ == "__main__":
    main()
