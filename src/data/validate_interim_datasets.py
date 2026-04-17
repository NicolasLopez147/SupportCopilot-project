import json
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import ValidationError

from src.schemas.sample_schema import SupportSample
from src.utils.paths import INTERIM_DATA_DIR


def validate_jsonl_file(file_path: Path) -> None:
    relative_path = file_path.relative_to(INTERIM_DATA_DIR)
    print("\n" + "=" * 20 + f" validating: {relative_path} " + "=" * 20)

    total_lines = 0
    valid_samples = 0
    invalid_json_lines = 0
    schema_errors = 0

    message_count_distribution = Counter()
    speaker_distribution = Counter()

    empty_message_texts = 0
    samples_with_intent = 0
    samples_with_structured_summary = 0
    samples_with_abstractive_summary = 0

    source_distribution = Counter()
    channel_distribution = Counter()

    for line_number, line in enumerate(file_path.open("r", encoding="utf-8"), start=1):
        total_lines += 1
        line = line.strip()

        if not line:
            invalid_json_lines += 1
            print(f"[warning] empty line at {line_number}")
            continue

        try:
            raw_sample = json.loads(line)
        except json.JSONDecodeError:
            invalid_json_lines += 1
            print(f"[warning] invalid json at line {line_number}")
            continue

        try:
            sample = SupportSample(**raw_sample)
        except ValidationError as e:
            schema_errors += 1
            print(f"[warning] schema error at line {line_number}")
            print(e)
            continue

        valid_samples += 1

        source_distribution[sample.source] += 1
        channel_distribution[sample.channel] += 1
        message_count_distribution[len(sample.messages)] += 1

        if sample.intent_label is not None:
            samples_with_intent += 1

        if sample.summary_structured is not None:
            samples_with_structured_summary += 1

        if sample.summary_abstractive is not None:
            samples_with_abstractive_summary += 1

        for message in sample.messages:
            speaker_distribution[message.speaker] += 1
            if not message.text.strip():
                empty_message_texts += 1

    print(f"total lines: {total_lines}")
    print(f"valid samples: {valid_samples}")
    print(f"invalid json lines: {invalid_json_lines}")
    print(f"schema errors: {schema_errors}")

    print("\nsource distribution:")
    for source, count in source_distribution.items():
        print(f"  - {source}: {count}")

    print("\nchannel distribution:")
    for channel, count in channel_distribution.items():
        print(f"  - {channel}: {count}")

    print("\nmessage count distribution:")
    for n_messages, count in sorted(message_count_distribution.items()):
        print(f"  - {n_messages} messages: {count}")

    print("\nspeaker distribution:")
    for speaker, count in speaker_distribution.items():
        print(f"  - {speaker}: {count}")

    print("\nfield completion:")
    print(f"  - samples with intent_label: {samples_with_intent}")
    print(f"  - samples with summary_structured: {samples_with_structured_summary}")
    print(f"  - samples with summary_abstractive: {samples_with_abstractive_summary}")

    print("\ncontent quality:")
    print(f"  - empty message texts: {empty_message_texts}")


def validate_all_interim_files() -> None:
    if not INTERIM_DATA_DIR.exists():
        print(f"[error] interim directory does not exist: {INTERIM_DATA_DIR}")
        return

    jsonl_files = sorted(INTERIM_DATA_DIR.rglob("*.jsonl"))

    if not jsonl_files:
        print(f"[error] no jsonl files found in: {INTERIM_DATA_DIR}")
        return

    print(f"found {len(jsonl_files)} jsonl files in {INTERIM_DATA_DIR}")

    for file_path in jsonl_files:
        validate_jsonl_file(file_path)


if __name__ == "__main__":
    validate_all_interim_files()
