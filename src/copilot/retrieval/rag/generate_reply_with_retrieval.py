import argparse
import html
import json
import re
from pathlib import Path

import emoji
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils.paths import DATA_DIR, EXPERIMENTS_OUTPUT_DIR


GENERATION_MODEL_NAME = "google/flan-t5-small"
RETRIEVAL_MODEL_NAME = "all-MiniLM-L6-v2"
KB_DIR = DATA_DIR / "kb"
DEFAULT_INPUT_PATH = DATA_DIR / "synthetic" / "synthetic_reply_eval.jsonl"
OUTPUT_DIR = EXPERIMENTS_OUTPUT_DIR / "reply" / "retrieval"
MAX_INPUT_TOKENS = 768
MAX_NEW_TOKENS = 128
TOP_K = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate support reply suggestions with retrieval grounding."
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
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"How many KB chunks to retrieve. Default: {TOP_K}",
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

        prepared.append(
            {
                "conversation_id": conversation_id,
                "scenario": scenario,
                "conversation_text": conversation_text,
                "reference_reply": reference_reply,
            }
        )

    return prepared


def load_kb_documents(kb_dir: Path) -> list[dict]:
    documents = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        documents.append(
            {
                "doc_id": path.stem,
                "path": str(path),
                "text": text,
            }
        )
    return documents


def split_sections(text: str) -> list[str]:
    sections = []
    current_section = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("## ") and current_section:
            sections.append("\n".join(current_section).strip())
            current_section = [stripped]
        else:
            current_section.append(stripped)

    if current_section:
        sections.append("\n".join(current_section).strip())

    return sections


def build_kb_chunks(documents: list[dict]) -> list[dict]:
    chunks = []
    for document in documents:
        sections = split_sections(document["text"])
        if not sections:
            sections = [document["text"]]

        for index, section_text in enumerate(sections):
            chunks.append(
                {
                    "chunk_id": f"{document['doc_id']}_chunk_{index:02d}",
                    "doc_id": document["doc_id"],
                    "path": document["path"],
                    "text": section_text,
                }
            )
    return chunks


def embed_texts(model: SentenceTransformer, texts: list[str]) -> torch.Tensor:
    return model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )


def retrieve_top_k(
    model: SentenceTransformer,
    chunks: list[dict],
    chunk_embeddings: torch.Tensor,
    query: str,
    top_k: int,
) -> list[dict]:
    query_embedding = embed_texts(model, [query])[0]
    similarity_scores = torch.matmul(chunk_embeddings, query_embedding)
    top_indices = torch.topk(similarity_scores, k=min(top_k, len(chunks))).indices.tolist()

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk = chunks[idx]
        results.append(
            {
                "rank": rank,
                "score": round(float(similarity_scores[idx].item()), 4),
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "text": chunk["text"],
            }
        )
    return results


def load_generation_model() -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM, str]:
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME)
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


def build_grounded_prompt(conversation_text: str, retrieved_chunks: list[dict]) -> str:
    kb_context = "\n\n".join(
        f"[Document {item['rank']} - {item['doc_id']}]\n{item['text']}"
        for item in retrieved_chunks
    )

    return (
        "write the next professional support agent reply using the conversation and retrieved support knowledge when relevant.\n\n"
        "Retrieved knowledge:\n"
        f"{kb_context}\n\n"
        "Conversation:\n"
        f"{conversation_text}\n"
    )


def save_json(data: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    dataset_samples = load_jsonl(args.input_path)
    reply_inputs = build_reply_inputs(dataset_samples)

    kb_documents = load_kb_documents(KB_DIR)
    kb_chunks = build_kb_chunks(kb_documents)

    print(f"[data] loaded samples: {len(dataset_samples)}")
    print(f"[data] usable reply inputs: {len(reply_inputs)}")
    print(f"[kb] documents loaded: {len(kb_documents)}")
    print(f"[kb] chunks created: {len(kb_chunks)}")

    print(f"[retrieval] loading embedding model: {RETRIEVAL_MODEL_NAME}")
    try:
        retrieval_model = SentenceTransformer(RETRIEVAL_MODEL_NAME, local_files_only=True)
    except OSError as exc:
        raise RuntimeError(
            f"Could not load retrieval model '{RETRIEVAL_MODEL_NAME}' from local cache. "
            "Run the retrieval pipeline once with internet access or make sure the model is cached locally."
        ) from exc
    chunk_embeddings = embed_texts(retrieval_model, [chunk["text"] for chunk in kb_chunks])

    print(f"[model] loading reply model: {GENERATION_MODEL_NAME}")
    tokenizer, model, device = load_generation_model()
    print(f"[model] running on device: {device}")

    subset = reply_inputs if args.num_samples is None else reply_inputs[: args.num_samples]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for index, sample in enumerate(subset, start=1):
        print(f"[generate] sample {index}/{len(subset)} -> {sample['conversation_id']}")
        retrieved = retrieve_top_k(
            model=retrieval_model,
            chunks=kb_chunks,
            chunk_embeddings=chunk_embeddings,
            query=sample["conversation_text"],
            top_k=args.top_k,
        )
        prompt = build_grounded_prompt(sample["conversation_text"], retrieved)
        suggested_reply = generate_reply(model, tokenizer, device, prompt)

        results.append(
            {
                "conversation_id": sample["conversation_id"],
                "scenario": sample["scenario"],
                "conversation_text": sample["conversation_text"],
                "reference_reply": sample["reference_reply"],
                "retrieved_chunks": retrieved,
                "prompt": prompt,
                "suggested_reply": suggested_reply,
                "model": GENERATION_MODEL_NAME,
                "retrieval": True,
                "method": "retrieval_t5",
            }
        )

    save_json(results, OUTPUT_DIR / "test_replies.json")
    print(f"[saved] replies -> {OUTPUT_DIR / 'test_replies.json'}")


if __name__ == "__main__":
    main()
