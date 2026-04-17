import html
import json
import re
import argparse
from pathlib import Path

import emoji
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data.support_summary_utils import build_support_summary_target
from src.utils.paths import DATA_DIR, EXPERIMENTS_OUTPUT_DIR


BASE_MODEL_NAME = "google/flan-t5-small"
OUTPUT_DIR = EXPERIMENTS_OUTPUT_DIR / "summary" / "lora_base"
ADAPTER_DIR = OUTPUT_DIR / "final_model"
DEFAULT_INPUT_PATH = DATA_DIR / "synthetic" / "reply_test.jsonl"
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 128
NUM_TEST_SAMPLES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate summary predictions from a LoRA fine-tuned T5 model."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the summary evaluation dataset. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=ADAPTER_DIR,
        help=f"Path to the LoRA adapter directory. Default: {ADAPTER_DIR}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_DIR / "test_predictions.json",
        help=f"Where to save generated summaries. Default: {OUTPUT_DIR / 'test_predictions.json'}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=NUM_TEST_SAMPLES,
        help="Optional cap on number of evaluation samples. Use 0 or a negative value for all.",
    )
    return parser.parse_args()


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


def build_generation_inputs(samples: list[dict]) -> list[dict]:
    prepared_samples = []

    for sample in samples:
        conversation_id = sample.get("conversation_id")
        messages = sample.get("messages", [])
        reference_summary = build_support_summary_target(sample)
        conversation_text = sample.get("conversation_text", "")
        input_text = sample.get("input_text")

        if not conversation_id or not reference_summary:
            continue

        if messages:
            conversation_text = format_conversation(messages)
        if not conversation_text:
            continue

        if not input_text:
            input_text = (
                "summarize the following customer support conversation: "
                f"{conversation_text}"
            )

        prepared_samples.append(
            {
                "conversation_id": conversation_id,
                "conversation_text": conversation_text,
                "input_text": input_text,
                "reference_summary": reference_summary,
            }
        )

    return prepared_samples


def load_model_and_tokenizer(adapter_dir: Path) -> tuple[AutoTokenizer, PeftModel, str]:
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_summary(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    device: str,
    input_text: str,
) -> str:
    inputs = tokenizer(
        input_text,
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


def save_json(data: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    test_samples = load_jsonl(args.input_path)
    test_inputs = build_generation_inputs(test_samples)

    print(f"[data] test examples: {len(test_inputs)}")

    subset = test_inputs if args.limit <= 0 else test_inputs[: args.limit]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    if subset:
        save_json(subset[0], args.output_path.parent / "sample_test_input.json")
        print(f"[saved] sample input -> {args.output_path.parent / 'sample_test_input.json'}")

    print(f"[model] loading LoRA adapter from {args.adapter_dir}")
    tokenizer, model, device = load_model_and_tokenizer(args.adapter_dir)
    print(f"[model] running on device: {device}")

    results = []

    for index, sample in enumerate(subset, start=1):
        print(f"[generate] sample {index}/{len(subset)} -> {sample['conversation_id']}")
        predicted_summary = generate_summary(model, tokenizer, device, sample["input_text"])

        results.append(
            {
                "conversation_id": sample["conversation_id"],
                "conversation_text": sample["conversation_text"],
                "input_text": sample["input_text"],
                "reference_summary": sample["reference_summary"],
                "predicted_summary": predicted_summary,
                "model": str(args.adapter_dir),
            }
        )

    save_json(results, args.output_path)
    print(f"[saved] predictions -> {args.output_path}")


if __name__ == "__main__":
    main()
