import html
import json
import re
from pathlib import Path

import emoji
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.utils.paths import DATA_DIR, EXPERIMENTS_OUTPUT_DIR


MODEL_NAME = "all-MiniLM-L6-v2"
TRAIN_PATH = DATA_DIR / "synthetic" / "reply_train.jsonl"
VALID_PATH = DATA_DIR / "synthetic" / "reply_valid.jsonl"
TEST_PATH = DATA_DIR / "synthetic" / "reply_test.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def clean_message_text(text: str) -> str:
    cleaned = html.unescape(text)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"@\w+", "", cleaned)
    cleaned = re.sub(r"__\w+__", "[REDACTED]", cleaned)
    cleaned = emoji.replace_emoji(cleaned, replace="")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" .\n\t")


def format_conversation(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        speaker = str(message.get("speaker", "")).strip().lower()
        text = clean_message_text(message.get("text", ""))
        if not text:
            continue
        speaker_label = "Customer" if speaker == "customer" else "Agent"
        lines.append(f"{speaker_label}: {text}")
    return "\n".join(lines)


def extract_texts_and_labels(samples: list[dict]) -> tuple[list[str], list[str]]:
    texts = []
    labels = []

    for sample in samples:
        messages = sample.get("messages", [])
        scenario = sample.get("scenario")
        if not messages or not scenario:
            continue

        conversation_text = format_conversation(messages)
        if not conversation_text:
            continue

        texts.append(conversation_text)
        labels.append(scenario)

    return texts, labels


def ensure_output_dir() -> Path:
    output_dir = EXPERIMENTS_OUTPUT_DIR / "intent" / "synthetic_embedding"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_artifact(encoder_name: str, clf: LogisticRegression, path: Path) -> None:
    artifact = {
        "encoder_name": encoder_name,
        "classifier": clf,
    }
    joblib.dump(artifact, path)


def evaluate_split(
    clf: LogisticRegression,
    encoder: SentenceTransformer,
    texts: list[str],
    labels: list[str],
) -> dict:
    embeddings = encoder.encode(texts, show_progress_bar=False)
    predictions = clf.predict(embeddings)
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
    }


def main() -> None:
    train_samples = load_jsonl(TRAIN_PATH)
    valid_samples = load_jsonl(VALID_PATH)
    test_samples = load_jsonl(TEST_PATH)

    X_train, y_train = extract_texts_and_labels(train_samples)
    X_valid, y_valid = extract_texts_and_labels(valid_samples)
    X_test, y_test = extract_texts_and_labels(test_samples)

    print(f"[data] train samples: {len(X_train)}")
    print(f"[data] valid samples: {len(X_valid)}")
    print(f"[data] test samples: {len(X_test)}")
    print(f"[data] num classes: {len(set(y_train))}")

    print(f"[embed] loading sentence-transformer encoder: {MODEL_NAME}")
    encoder = SentenceTransformer(MODEL_NAME)

    print("[embed] encoding train split")
    X_train_embeddings = encoder.encode(X_train, show_progress_bar=True)

    print("[train] fitting synthetic support intent classifier")
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_embeddings, y_train)

    print("[eval] validating on reply_valid split")
    valid_metrics = evaluate_split(clf, encoder, X_valid, y_valid)
    print(f"[eval] valid accuracy: {valid_metrics['accuracy']:.4f}")
    print(f"[eval] valid macro_f1: {valid_metrics['macro_f1']:.4f}")

    print("[eval] testing on reply_test split")
    test_metrics = evaluate_split(clf, encoder, X_test, y_test)
    print(f"[eval] test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"[eval] test macro_f1: {test_metrics['macro_f1']:.4f}")

    print("[train] refitting final classifier on train + valid splits")
    X_final = X_train + X_valid
    y_final = y_train + y_valid
    X_final_embeddings = encoder.encode(X_final, show_progress_bar=True)
    final_clf = LogisticRegression(max_iter=2000)
    final_clf.fit(X_final_embeddings, y_final)

    output_dir = ensure_output_dir()
    save_json(
        {
            "encoder_name": MODEL_NAME,
            "num_classes": len(set(y_train)),
            "train_samples": len(X_train),
            "valid_samples": len(X_valid),
            "test_samples": len(X_test),
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
        },
        output_dir / "metrics.json",
    )
    save_json(valid_metrics["classification_report"], output_dir / "valid_classification_report.json")
    save_json(test_metrics["classification_report"], output_dir / "test_classification_report.json")
    save_artifact(MODEL_NAME, final_clf, output_dir / "intent_synthetic_model.joblib")

    print(f"[saved] metrics -> {output_dir / 'metrics.json'}")
    print(f"[saved] valid report -> {output_dir / 'valid_classification_report.json'}")
    print(f"[saved] test report -> {output_dir / 'test_classification_report.json'}")
    print(f"[saved] model artifact -> {output_dir / 'intent_synthetic_model.joblib'}")


if __name__ == "__main__":
    main()
