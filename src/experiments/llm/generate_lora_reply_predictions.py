import argparse
import html
import json
import re
from pathlib import Path

import emoji
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils.paths import DATA_DIR, EXPERIMENTS_OUTPUT_DIR


BASE_MODEL_NAME = "google/flan-t5-small"
OUTPUT_DIR = EXPERIMENTS_OUTPUT_DIR / "reply" / "lora_base"
ADAPTER_DIR = OUTPUT_DIR / "final_model"
DEFAULT_INPUT_PATH = DATA_DIR / "synthetic" / "reply_test.jsonl"
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reply predictions from the LoRA fine-tuned T5 model."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the reply test split. Default: {DEFAULT_INPUT_PATH}",
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
        default=OUTPUT_DIR / "test_replies.json",
        help=f"Where to save generated replies. Default: {OUTPUT_DIR / 'test_replies.json'}",
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

        speaker_label = "Customer" if speaker == "customer" else "Agent"
        formatted_lines.append(f"{speaker_label}: {text}")

    return "\n".join(formatted_lines)


def build_generation_inputs(samples: list[dict]) -> list[dict]:
    prepared_samples = []

    for sample in samples:
        conversation_id = sample.get("conversation_id")
        scenario = sample.get("scenario")
        reference_reply = sample.get("reference_reply")
        messages = sample.get("messages", [])

        if not conversation_id or not messages or not reference_reply:
            continue

        conversation_text = format_conversation(messages)
        if not conversation_text:
            continue

        input_text = (
            "write the next professional support agent reply based on this conversation: "
            f"{conversation_text}"
        )

        prepared_samples.append(
            {
                "conversation_id": conversation_id,
                "scenario": scenario,
                "conversation_text": conversation_text,
                "input_text": input_text,
                "reference_reply": reference_reply,
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


def clean_generated_reply(text: str) -> str:
    reply = text.strip()
    reply = re.sub(r"^(Agent|Customer)\s*:\s*", "", reply, flags=re.IGNORECASE)
    return reply.strip()


def generate_reply(
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
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return clean_generated_reply(decoded)


def save_json(data: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    test_samples = load_jsonl(args.input_path)
    test_inputs = build_generation_inputs(test_samples)

    print(f"[data] test examples: {len(test_inputs)}")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[model] loading LoRA adapter from {args.adapter_dir}")
    tokenizer, model, device = load_model_and_tokenizer(args.adapter_dir)
    print(f"[model] running on device: {device}")

    results = []
    for index, sample in enumerate(test_inputs, start=1):
        print(f"[generate] sample {index}/{len(test_inputs)} -> {sample['conversation_id']}")
        suggested_reply = generate_reply(model, tokenizer, device, sample["input_text"])

        results.append(
            {
                "conversation_id": sample["conversation_id"],
                "scenario": sample["scenario"],
                "conversation_text": sample["conversation_text"],
                "reference_reply": sample["reference_reply"],
                "prompt": sample["input_text"],
                "suggested_reply": suggested_reply,
                "model": str(args.adapter_dir),
                "retrieval": False,
                "method": "lora_reply_t5",
            }
        )

    save_json(results, args.output_path)
    print(f"[saved] replies -> {args.output_path}")


if __name__ == "__main__":
    main()
