import argparse
import html
import json
import math
import re
from pathlib import Path

import emoji
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.utils.paths import EXPERIMENTS_OUTPUT_DIR, FEEDBACK_AUGMENTED_DIR


BASE_MODEL_NAME = "google/flan-t5-small"
DEFAULT_INPUT_PATH = FEEDBACK_AUGMENTED_DIR / "summary_support_retraining_batch.jsonl"
DEFAULT_BASE_ADAPTER_DIR = EXPERIMENTS_OUTPUT_DIR / "summary" / "lora_base" / "final_model"
DEFAULT_OUTPUT_DIR = EXPERIMENTS_OUTPUT_DIR / "summary" / "lora_feedback"
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue LoRA summary training using the support-domain feedback batch."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Support summary feedback batch path. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--base-adapter-dir",
        type=Path,
        default=DEFAULT_BASE_ADAPTER_DIR,
        help=f"Existing LoRA summary adapter. Default: {DEFAULT_BASE_ADAPTER_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save the feedback-retrained summary adapter. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=2.0,
        help="Additional number of epochs for incremental summary retraining. Default: 2",
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


def clean_text(text: str) -> str:
    cleaned = html.unescape(text)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"@\w+", "", cleaned)
    cleaned = re.sub(r"__\w+__", "[REDACTED]", cleaned)
    cleaned = emoji.replace_emoji(cleaned, replace="")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" .\n\t")


def build_examples(samples: list[dict]) -> list[dict]:
    examples = []

    for sample in samples:
        conversation_id = sample.get("conversation_id")
        conversation_text = clean_text(sample.get("conversation_text", ""))
        input_text = sample.get("input_text")
        target_text = sample.get("target_text")

        if not conversation_id or not conversation_text or not target_text:
            continue

        if not input_text:
            input_text = (
                "summarize the following customer support conversation: "
                f"{conversation_text}"
            )

        examples.append(
            {
                "conversation_id": conversation_id,
                "scenario": sample.get("scenario"),
                "conversation_text": conversation_text,
                "input_text": input_text,
                "target_text": clean_text(target_text),
            }
        )

    return examples


def split_feedback_examples(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    if len(examples) <= 1:
        return examples, examples

    ordered = sorted(examples, key=lambda item: item["conversation_id"])
    valid_count = max(1, math.floor(len(ordered) * 0.2))
    train_count = max(1, len(ordered) - valid_count)

    train_examples = ordered[:train_count]
    valid_examples = ordered[train_count:]
    if not valid_examples:
        valid_examples = train_examples[-1:]

    return train_examples, valid_examples


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def preprocess_batch(batch: dict) -> dict:
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=MAX_SOURCE_LENGTH,
            truncation=True,
        )

        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )


def main() -> None:
    args = parse_args()

    if not args.base_adapter_dir.exists():
        raise FileNotFoundError(
            f"Base summary adapter not found at {args.base_adapter_dir}. "
            "Run the original summary LoRA training first."
        )

    feedback_samples = load_jsonl(args.input_path)
    feedback_examples = build_examples(feedback_samples)
    train_examples, valid_examples = split_feedback_examples(feedback_examples)

    print(f"[data] support feedback examples: {len(feedback_examples)}")
    print(f"[data] train examples: {len(train_examples)}")
    print(f"[data] valid examples: {len(valid_examples)}")

    if train_examples:
        print("\n[preview] support summary input example:\n")
        print(train_examples[0]["input_text"])
        print("\n[preview] support summary target example:\n")
        print(train_examples[0]["target_text"])

    train_dataset = Dataset.from_list(train_examples)
    valid_dataset = Dataset.from_list(valid_examples)

    print(f"\n[model] loading tokenizer from {args.base_adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_adapter_dir)

    print(f"[model] loading base model: {BASE_MODEL_NAME}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)

    print(f"[model] loading trainable LoRA adapter from {args.base_adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.base_adapter_dir, is_trainable=True)
    model.print_trainable_parameters()

    print("[data] tokenizing train/valid datasets")
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    tokenized_valid = tokenize_dataset(valid_dataset, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        weight_decay=0.01,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=data_collator,
    )

    print("\n[train] starting feedback-driven incremental summary retraining")
    trainer.train()

    final_dir = args.output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metadata = {
        "base_model_name": BASE_MODEL_NAME,
        "base_adapter_dir": str(args.base_adapter_dir),
        "input_path": str(args.input_path),
        "feedback_examples": len(feedback_examples),
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "num_train_epochs": args.num_train_epochs,
    }
    with (args.output_dir / "retraining_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[saved] feedback-retrained summary LoRA model -> {final_dir}")
    print(f"[saved] retraining metadata -> {args.output_dir / 'retraining_metadata.json'}")


if __name__ == "__main__":
    main()
