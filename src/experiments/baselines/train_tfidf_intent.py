import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from src.utils.paths import EXPERIMENTS_OUTPUT_DIR, INTERIM_DATA_DIR


def load_jsonl(path: Path) -> list[dict]:
    samples = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    return samples


def extract_texts_and_labels(samples: list[dict]) -> tuple[list[str], list[str]]:
    texts = []
    labels = []

    for sample in samples:
        messages = sample.get("messages", [])
        metadata = sample.get("metadata", {})

        if not messages:
            continue

        text = messages[0].get("text", "").strip()
        label = metadata.get("original_label_name")

        if not text or not label:
            continue

        texts.append(text)
        labels.append(label)

    return texts, labels


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def ensure_output_dir() -> Path:
    output_dir = EXPERIMENTS_OUTPUT_DIR / "intent" / "tfidf_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    train_path = INTERIM_DATA_DIR / "banking77" / "train.jsonl"
    test_path = INTERIM_DATA_DIR / "banking77" / "test.jsonl"

    train_samples = load_jsonl(train_path)
    test_samples = load_jsonl(test_path)

    X_train, y_train = extract_texts_and_labels(train_samples)
    X_test, y_test = extract_texts_and_labels(test_samples)

    print(f"[data] train samples: {len(X_train)}")
    print(f"[data] test samples: {len(X_test)}")
    print(f"[data] num classes: {len(set(y_train))}")

    model = build_model()

    print("[train] fitting TF-IDF + Logistic Regression baseline")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"[eval] accuracy: {accuracy:.4f}")
    print(f"[eval] macro_f1: {macro_f1:.4f}")

    output_dir = ensure_output_dir()
    save_json(
        {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_classes": len(set(y_train)),
        },
        output_dir / "metrics.json",
    )
    save_json(report, output_dir / "classification_report.json")

    print(f"[saved] metrics -> {output_dir / 'metrics.json'}")
    print(f"[saved] classification report -> {output_dir / 'classification_report.json'}")


if __name__ == "__main__":
    main()
