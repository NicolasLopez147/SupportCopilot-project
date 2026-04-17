import argparse
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import emoji
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.utils.paths import DATA_DIR, EXPERIMENTS_OUTPUT_DIR


MODEL_NAME = "google/flan-t5-small"
DEFAULT_INPUT_PATH = DATA_DIR / "synthetic" / "synthetic_reply_eval.jsonl"
DEFAULT_SPLIT_DIR = DATA_DIR / "synthetic"
OUTPUT_DIR = EXPERIMENTS_OUTPUT_DIR / "reply" / "lora_base"
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for support reply generation on the synthetic dataset."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the synthetic reply dataset. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help=f"Where to save train/valid/test splits. Default: {DEFAULT_SPLIT_DIR}",
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


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def build_seq2seq_examples(samples: list[dict]) -> list[dict]:
    examples = []

    for sample in samples:
        conversation_id = sample.get("conversation_id")
        scenario = sample.get("scenario")
        messages = sample.get("messages", [])
        target_text = sample.get("reference_reply")

        if not conversation_id or not messages or not target_text:
            continue

        conversation_text = format_conversation(messages)
        if not conversation_text:
            continue

        input_text = (
            "write the next professional support agent reply based on this conversation: "
            f"{conversation_text}"
        )

        examples.append(
            {
                "conversation_id": conversation_id,
                "scenario": scenario,
                "conversation_text": conversation_text,
                "input_text": input_text,
                "target_text": target_text.strip(),
            }
        )

    return examples


def split_by_scenario(samples: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.get("scenario", "unknown")].append(sample)

    train_rows = []
    valid_rows = []
    test_rows = []

    for scenario in sorted(grouped):
        scenario_samples = sorted(grouped[scenario], key=lambda item: item["conversation_id"])
        n = len(scenario_samples)

        test_count = max(1, math.floor(n * 0.15))
        valid_count = max(1, math.floor(n * 0.15))
        train_count = n - test_count - valid_count

        if train_count < 1:
            train_count = max(1, n - 2)
            remaining = n - train_count
            valid_count = 1 if remaining > 1 else 0
            test_count = remaining - valid_count

        train_rows.extend(scenario_samples[:train_count])
        valid_rows.extend(scenario_samples[train_count : train_count + valid_count])
        test_rows.extend(scenario_samples[train_count + valid_count :])

    return train_rows, valid_rows, test_rows


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
    samples = load_jsonl(args.input_path)
    train_rows, valid_rows, test_rows = split_by_scenario(samples)

    train_path = args.split_dir / "reply_train.jsonl"
    valid_path = args.split_dir / "reply_valid.jsonl"
    test_path = args.split_dir / "reply_test.jsonl"

    save_jsonl(train_rows, train_path)
    save_jsonl(valid_rows, valid_path)
    save_jsonl(test_rows, test_path)

    train_examples = build_seq2seq_examples(train_rows)
    valid_examples = build_seq2seq_examples(valid_rows)

    print(f"[data] total samples: {len(samples)}")
    print(f"[data] train examples: {len(train_examples)}")
    print(f"[data] valid examples: {len(valid_examples)}")
    print(f"[data] test examples: {len(test_rows)}")
    print(f"[saved] train split -> {train_path}")
    print(f"[saved] valid split -> {valid_path}")
    print(f"[saved] test split -> {test_path}")

    if train_examples:
        print("\n[preview] input example:\n")
        print(train_examples[0]["input_text"])
        print("\n[preview] target example:\n")
        print(train_examples[0]["target_text"])

    train_dataset = Dataset.from_list(train_examples)
    valid_dataset = Dataset.from_list(valid_examples)

    print(f"\n[model] loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    print("[data] tokenizing train/valid datasets")
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    tokenized_valid = tokenize_dataset(valid_dataset, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        weight_decay=0.01,
        num_train_epochs=4,
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

    print("\n[train] starting LoRA fine-tuning for reply generation")
    trainer.train()

    final_dir = OUTPUT_DIR / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print(f"[saved] final LoRA model -> {final_dir}")


if __name__ == "__main__":
    main()
