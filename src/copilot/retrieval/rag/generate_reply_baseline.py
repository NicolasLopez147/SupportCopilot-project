import argparse
import html
import json
import re
from pathlib import Path

import emoji
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils.paths import DATA_DIR, EXPERIMENTS_OUTPUT_DIR


MODEL_NAME = "google/flan-t5-small"
DEFAULT_INPUT_PATH = DATA_DIR / "synthetic" / "synthetic_reply_eval.jsonl"
OUTPUT_DIR = EXPERIMENTS_OUTPUT_DIR / "reply" / "baseline"
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate support reply suggestions without retrieval."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the JSONL dataset. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="How many samples to generate. Default: use the whole input dataset.",
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


def build_reply_inputs(samples: list[dict]) -> list[dict]:
    prepared = []

    for sample in samples:
        conversation_id = sample.get("conversation_id")
        scenario = sample.get("scenario")
        reference_reply = sample.get("reference_reply")
        messages = sample.get("messages", [])
        if not conversation_id or not messages:
            continue

        conversation_text = format_conversation(messages)
        if not conversation_text:
            continue

        prompt = (
            "write the next professional support agent reply based on this conversation: "
            f"{conversation_text}"
        )

        prepared.append(
            {
                "conversation_id": conversation_id,
                "scenario": scenario,
                "conversation_text": conversation_text,
                "reference_reply": reference_reply,
                "prompt": prompt,
            }
        )

    return prepared


def load_model_and_tokenizer() -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM, str]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def clean_generated_reply(text: str) -> str:
    reply = text.strip()
    reply = re.sub(r"^(Agent|Customer)\s*:\s*", "", reply, flags=re.IGNORECASE)
    return reply.strip()


def generate_reply(
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
    dataset_samples = load_jsonl(args.input_path)
    reply_inputs = build_reply_inputs(dataset_samples)

    print(f"[data] loaded samples: {len(dataset_samples)}")
    print(f"[data] usable reply inputs: {len(reply_inputs)}")

    subset = reply_inputs if args.num_samples is None else reply_inputs[: args.num_samples]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[model] loading reply model: {MODEL_NAME}")
    tokenizer, model, device = load_model_and_tokenizer()
    print(f"[model] running on device: {device}")

    results = []
    for index, sample in enumerate(subset, start=1):
        print(f"[generate] sample {index}/{len(subset)} -> {sample['conversation_id']}")
        suggested_reply = generate_reply(model, tokenizer, device, sample["prompt"])
        results.append(
            {
                "conversation_id": sample["conversation_id"],
                "scenario": sample["scenario"],
                "conversation_text": sample["conversation_text"],
                "reference_reply": sample["reference_reply"],
                "prompt": sample["prompt"],
                "suggested_reply": suggested_reply,
                "model": MODEL_NAME,
                "retrieval": False,
                "method": "baseline_t5",
            }
        )

    save_json(results, OUTPUT_DIR / "test_replies.json")
    print(f"[saved] replies -> {OUTPUT_DIR / 'test_replies.json'}")


if __name__ == "__main__":
    main()
