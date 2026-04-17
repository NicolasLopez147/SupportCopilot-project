import random
from collections import Counter

from datasets import ClassLabel, load_from_disk

from src.utils.paths import RAW_DATA_DIR


def print_separator(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def inspect_split_basic(split_name: str, split_data) -> None:
    print_separator(f"split: {split_name}")
    print(f"rows: {len(split_data)}")
    print(f"columns: {split_data.column_names}")
    print(f"features: {split_data.features}")


def inspect_label_column(split_data) -> None:
    if "label" not in split_data.column_names:
        print("\n[label inspection] no 'label' column found")
        return

    feature = split_data.features["label"]
    print("\n[label inspection]")
    print(f"label feature type: {type(feature)}")

    labels = split_data["label"]

    if isinstance(feature, ClassLabel):
        label_names = feature.names
        print(f"number of classes: {len(label_names)}")
        print("label names:")
        for idx, name in enumerate(label_names):
            print(f"  {idx}: {name}")

        label_counts = Counter(labels)
        print("\nlabel distribution (top 20 by id):")
        for label_id, count in label_counts.most_common(20):
            label_name = label_names[label_id]
            print(f"  {label_id} ({label_name}): {count}")
    else:
        label_counts = Counter(labels)
        print("label column is not a ClassLabel feature")
        print("label distribution (top 20):")
        for label_value, count in label_counts.most_common(20):
            print(f"  {label_value}: {count}")


def inspect_text_lengths(split_data) -> None:
    print("\n[text length inspection]")

    for column in ["text", "input", "output"]:
        if column not in split_data.column_names:
            continue

        values = split_data[column]

        non_null_values = [v for v in values if isinstance(v, str)]
        empty_count = sum(1 for v in non_null_values if not v.strip())

        if not non_null_values:
            print(f"{column}: no valid string values found")
            continue

        lengths = [len(v) for v in non_null_values]
        word_lengths = [len(v.split()) for v in non_null_values]

        print(f"\ncolumn: {column}")
        print(f"  valid strings: {len(non_null_values)}")
        print(f"  empty strings: {empty_count}")
        print(f"  min chars: {min(lengths)}")
        print(f"  max chars: {max(lengths)}")
        print(f"  avg chars: {sum(lengths) / len(lengths):.2f}")
        print(f"  avg words: {sum(word_lengths) / len(word_lengths):.2f}")


def show_random_examples(split_data, n_examples: int = 3) -> None:
    print("\n[random examples]")

    if len(split_data) == 0:
        print("no examples available")
        return

    indices = random.sample(range(len(split_data)), k=min(n_examples, len(split_data)))

    for i, idx in enumerate(indices, start=1):
        sample = split_data[idx]
        print(f"\nexample {i} (index={idx})")
        for key, value in sample.items():
            print(f"  - {key}: {value}")


def inspect_dataset(dataset_name: str) -> None:
    dataset_path = RAW_DATA_DIR / dataset_name

    if not dataset_path.exists():
        print(f"[error] dataset not found: {dataset_path}")
        return

    dataset = load_from_disk(str(dataset_path))

    print_separator(f"dataset: {dataset_name}")
    print(f"path: {dataset_path}")
    print(f"splits: {list(dataset.keys())}")

    for split_name, split_data in dataset.items():
        inspect_split_basic(split_name, split_data)
        inspect_label_column(split_data)
        inspect_text_lengths(split_data)
        show_random_examples(split_data, n_examples=3)


if __name__ == "__main__":
    inspect_dataset("banking77")
    inspect_dataset("customer_support_tweets")