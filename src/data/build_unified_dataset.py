import csv
import json
from pathlib import Path
from typing import List

import yaml
from datasets import load_from_disk

from src.schemas.sample_schema import Message, SupportSample
from src.utils.paths import CONFIGS_DIR, INTERIM_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR


def load_data_config() -> dict:
    config_path = CONFIGS_DIR / "data.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dirs() -> None:
    (INTERIM_DATA_DIR / "banking77").mkdir(parents=True, exist_ok=True)
    (INTERIM_DATA_DIR / "tweetsum").mkdir(parents=True, exist_ok=True)
    (INTERIM_DATA_DIR / "customer_support_tweets").mkdir(parents=True, exist_ok=True)


def save_jsonl(samples: List[SupportSample], output_path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")


def convert_banking77_split(split_name: str, split_data, language: str) -> List[SupportSample]:
    label_feature = split_data.features["label"]
    samples = []

    for idx, row in enumerate(split_data):
        label_id = row["label"]
        label_name = label_feature.int2str(label_id)

        sample = SupportSample(
            conversation_id=f"banking77_{split_name}_{idx:06d}",
            source="banking77",
            language=language,
            channel="query",
            messages=[
                Message(
                    speaker="customer",
                    text=row["text"]
                )
            ],
            intent_label=None,
            summary_structured=None,
            summary_abstractive=None,
            metadata={
                "original_label_id": label_id,
                "original_label_name": label_name,
                "split": split_name,
                "dataset_role": "intent_bootstrap"
            }
        )

        samples.append(sample)

    return samples


def is_valid_support_pair(customer_text: str, agent_text: str) -> bool:
    if not isinstance(customer_text, str) or not isinstance(agent_text, str):
        return False

    customer_text = customer_text.strip()
    agent_text = agent_text.strip()

    if len(customer_text) < 3:
        return False

    if len(agent_text) < 3:
        return False

    return True


def convert_customer_support_tweets_split(split_name: str, split_data, language: str) -> List[SupportSample]:
    samples = []
    skipped_count = 0

    for idx, row in enumerate(split_data):
        customer_text = row["input"]
        agent_text = row["output"]

        if not is_valid_support_pair(customer_text, agent_text):
            skipped_count += 1
            continue

        sample = SupportSample(
            conversation_id=f"customer_support_tweets_{split_name}_{idx:06d}",
            source="customer_support_tweets",
            language=language,
            channel="social_support",
            messages=[
                Message(
                    speaker="customer",
                    text=customer_text.strip()
                ),
                Message(
                    speaker="agent",
                    text=agent_text.strip()
                )
            ],
            intent_label=None,
            summary_structured=None,
            summary_abstractive=None,
            metadata={
                "split": split_name,
                "dataset_role": "noise_pool"
            }
        )

        samples.append(sample)

    print(f"[info] customer_support_tweets - skipped invalid pairs: {skipped_count}")
    return samples


def select_abstractive_summary(abstractive_summaries: List[List[str] | None]) -> tuple[str | None, int | None]:
    for index, summary_sentences in enumerate(abstractive_summaries):
        if not summary_sentences:
            continue

        clean_sentences = [
            sentence.strip()
            for sentence in summary_sentences
            if isinstance(sentence, str) and sentence.strip()
        ]

        if clean_sentences:
            return " ".join(clean_sentences), index

    return None, None


def load_twcs_lookup(twcs_path: Path) -> dict[str, tuple[str, str]]:
    tweet_lookup: dict[str, tuple[str, str]] = {}

    with twcs_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweet_id = str(row["tweet_id"])
            inbound = row["inbound"]
            text = row["text"]
            tweet_lookup[tweet_id] = (inbound, text)

    return tweet_lookup


def parse_sentence_offset(offset: str) -> tuple[int, int]:
    start_str, end_str = offset.replace("[", "").replace("]", "").split(",")
    return int(start_str), int(end_str)


def build_tweetsum_message(tweet_lookup: dict[str, tuple[str, str]], tweet_id: str,
                           sentence_offsets: List[str]) -> Message | None:
    tweet_content = tweet_lookup.get(str(tweet_id))
    if tweet_content is None:
        return None

    inbound, text = tweet_content
    sentences = []
    for offset in sentence_offsets:
        start, end = parse_sentence_offset(offset)
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)

    if not sentences:
        return None

    return Message(
        speaker="agent" if str(inbound).upper() == "FALSE" else "customer",
        text=" ".join(sentences),
    )


def convert_tweetsum_split(split_name: str, input_path: Path, tweet_lookup: dict[str, tuple[str, str]],
                           language: str) -> List[SupportSample]:
    samples = []
    skipped_empty_dialogs = 0
    missing_tweet_messages = 0

    with input_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            row = json.loads(raw_line)
            messages = []

            for tweet_item in row.get("tweet_ids_sentence_offset", []):
                message = build_tweetsum_message(
                    tweet_lookup=tweet_lookup,
                    tweet_id=str(tweet_item["tweet_id"]),
                    sentence_offsets=tweet_item["sentence_offsets"],
                )

                if message is None:
                    missing_tweet_messages += 1
                    continue

                messages.append(message)

            if not messages:
                skipped_empty_dialogs += 1
                continue

            abstractive_summaries = [
                annotation.get("abstractive")
                for annotation in row.get("annotations", [])
            ]
            abstractive_summary, selected_summary_index = select_abstractive_summary(abstractive_summaries)

            extractive_count = sum(
                1 for annotation in row.get("annotations", [])
                if annotation.get("extractive")
            )

            sample = SupportSample(
                conversation_id=f"tweetsum_{split_name}_{row['conversation_id']}",
                source="tweetsum",
                language=language,
                channel="social_support",
                messages=messages,
                intent_label=None,
                summary_structured=None,
                summary_abstractive=abstractive_summary,
                metadata={
                    "split": split_name,
                    "dataset_role": "summary_core",
                    "selected_abstractive_summary_index": selected_summary_index,
                    "num_turns": len(messages),
                    "num_abstractive_summaries": len(abstractive_summaries),
                    "num_extractive_summaries": extractive_count,
                }
            )
            samples.append(sample)

    print(f"[info] tweetsum - skipped empty dialogs: {skipped_empty_dialogs}")
    print(f"[info] tweetsum - skipped missing tweet messages: {missing_tweet_messages}")
    return samples


def main() -> None:
    config = load_data_config()
    language = config.get("language", "en")

    ensure_output_dirs()

    banking77_path = RAW_DATA_DIR / "banking77"
    banking77_dataset = load_from_disk(str(banking77_path))

    for split_name, split_data in banking77_dataset.items():
        print(f"[convert] banking77 - split: {split_name}")
        samples = convert_banking77_split(split_name, split_data, language)

        output_path = INTERIM_DATA_DIR / "banking77" / f"{split_name}.jsonl"
        save_jsonl(samples, output_path)

        print(f"[saved] {len(samples)} samples -> {output_path}")

    tweetsum_cfg = config.get("datasets", {}).get("tweetsum", {})
    tweetsum_paths = tweetsum_cfg.get("local_paths", {})

    split_to_file_key = {
        "train": "train_file",
        "valid": "valid_file",
        "test": "test_file",
    }

    tweetsum_twcs_path = RAW_DATA_DIR / "tweetsum" / "twcs.csv"
    tweetsum_lookup = load_twcs_lookup(tweetsum_twcs_path)
    for split_name, file_key in split_to_file_key.items():
        relative_input_path = tweetsum_paths.get(file_key)
        if not relative_input_path:
            print(f"[warning] tweetsum - missing config path for {file_key}")
            continue

        input_path = PROJECT_ROOT / Path(relative_input_path)
        print(f"[convert] tweetsum - split: {split_name}")
        samples = convert_tweetsum_split(
            split_name=split_name,
            input_path=input_path,
            tweet_lookup=tweetsum_lookup,
            language=language,
        )

        output_path = INTERIM_DATA_DIR / "tweetsum" / f"{split_name}.jsonl"
        save_jsonl(samples, output_path)

        print(f"[saved] {len(samples)} samples -> {output_path}")

    customer_support_path = RAW_DATA_DIR / "customer_support_tweets"
    customer_support_dataset = load_from_disk(str(customer_support_path))

    for split_name, split_data in customer_support_dataset.items():
        print(f"[convert] customer_support_tweets - split: {split_name}")
        samples = convert_customer_support_tweets_split(split_name, split_data, language)

        output_path = INTERIM_DATA_DIR / "customer_support_tweets" / f"{split_name}.jsonl"
        save_jsonl(samples, output_path)

        print(f"[saved] {len(samples)} samples -> {output_path}")


if __name__ == "__main__":
    main()
