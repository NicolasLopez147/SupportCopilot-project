import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data.support_summary_utils import (
    build_support_summary_target,
    format_conversation,
    load_jsonl,
)
from src.utils.paths import DATA_DIR, EXPERIMENTS_OUTPUT_DIR


OUTPUT_DIR = EXPERIMENTS_OUTPUT_DIR / "summary" / "full_finetune"
MODEL_DIR = OUTPUT_DIR / "final_model"
DEFAULT_INPUT_PATH = DATA_DIR / "synthetic" / "reply_test.jsonl"
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 128
NUM_TEST_SAMPLES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate summary predictions from a fully fine-tuned T5 model."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the summary evaluation dataset. Default: {DEFAULT_INPUT_PATH}",
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


def load_model_and_tokenizer() -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM, str]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_summary(
    model: AutoModelForSeq2SeqLM,
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

    print(f"[model] loading fine-tuned model from {MODEL_DIR}")
    tokenizer, model, device = load_model_and_tokenizer()
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
                "model": "google/flan-t5-small-full-finetuned",
            }
        )

    save_json(results, args.output_path)
    print(f"[saved] predictions -> {args.output_path}")


if __name__ == "__main__":
    main()
