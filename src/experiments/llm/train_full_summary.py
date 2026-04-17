import argparse
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.data.support_summary_utils import (
    build_support_summary_target,
    format_conversation,
    load_jsonl,
)
from src.utils.paths import DATA_DIR, EXPERIMENTS_OUTPUT_DIR


MODEL_NAME = "google/flan-t5-small"
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 128
OUTPUT_DIR = EXPERIMENTS_OUTPUT_DIR / "summary" / "full_finetune"
DEFAULT_TRAIN_PATH = DATA_DIR / "synthetic" / "reply_train.jsonl"
DEFAULT_VALID_PATH = DATA_DIR / "synthetic" / "reply_valid.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full fine-tuning for support summary generation on the synthetic support dataset."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help=f"Path to the synthetic support summary train split. Default: {DEFAULT_TRAIN_PATH}",
    )
    parser.add_argument(
        "--valid-path",
        type=Path,
        default=DEFAULT_VALID_PATH,
        help=f"Path to the synthetic support summary valid split. Default: {DEFAULT_VALID_PATH}",
    )
    return parser.parse_args()


def build_seq2seq_examples(samples: list[dict]) -> list[dict]:
    examples = []

    for sample in samples:
        conversation_id = sample.get("conversation_id")
        messages = sample.get("messages", [])
        conversation_text = sample.get("conversation_text", "")
        target_text = build_support_summary_target(sample)

        if not conversation_id or not target_text:
            continue

        if messages:
            conversation_text = format_conversation(messages)
        if not conversation_text:
            continue

        input_text = (
            "summarize the following customer support conversation: "
            f"{conversation_text}"
        )

        examples.append(
            {
                "conversation_id": conversation_id,
                "input_text": input_text,
                "target_text": target_text.strip(),
            }
        )

    return examples


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

    tokenized_dataset = dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized_dataset


def main() -> None:
    args = parse_args()

    train_samples = load_jsonl(args.train_path)
    valid_samples = load_jsonl(args.valid_path)

    train_examples = build_seq2seq_examples(train_samples)
    valid_examples = build_seq2seq_examples(valid_samples)

    print(f"[data] train examples: {len(train_examples)}")
    print(f"[data] valid examples: {len(valid_examples)}")

    if train_examples:
        print("\n[preview] input example:\n")
        print(train_examples[0]["input_text"])
        print("\n[preview] target example:\n")
        print(train_examples[0]["target_text"])

    train_dataset = Dataset.from_list(train_examples)
    valid_dataset = Dataset.from_list(valid_examples)

    print(f"\n[model] loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

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
        num_train_epochs=2,
        predict_with_generate=True,
        logging_steps=20,
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

    print("\n[train] starting full fine-tuning")
    trainer.train()

    final_dir = OUTPUT_DIR / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print(f"[saved] final model -> {final_dir}")


if __name__ == "__main__":
    main()
