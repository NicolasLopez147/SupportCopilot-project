import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from src.utils.paths import DATA_DIR, EXPERIMENTS_OUTPUT_DIR


KB_DIR = DATA_DIR / "kb"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
OUTPUT_DIR = EXPERIMENTS_OUTPUT_DIR / "reply" / "retrieval_debug"


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
    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings


def retrieve_top_k(
    model: SentenceTransformer,
    chunks: list[dict],
    chunk_embeddings: torch.Tensor,
    query: str,
    top_k: int = TOP_K,
) -> list[dict]:
    query_embedding = embed_texts(model, [query])[0]
    similarity_scores = torch.matmul(chunk_embeddings, query_embedding)
    top_indices = torch.topk(similarity_scores, k=min(top_k, len(chunks))).indices.tolist()

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk = chunks[idx]
        score = float(similarity_scores[idx].item())
        results.append(
            {
                "rank": rank,
                "score": round(score, 4),
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "path": chunk["path"],
                "text": chunk["text"],
            }
        )

    return results


def save_json(data: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    documents = load_kb_documents(KB_DIR)
    chunks = build_kb_chunks(documents)

    print(f"[kb] documents loaded: {len(documents)}")
    print(f"[kb] chunks created: {len(chunks)}")

    print(f"[model] loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embed_texts(model, chunk_texts)

    sample_query = (
        "Customer reports a red light on the fiber box, no internet access, "
        "and says the router has already been restarted."
    )

    print("\n[query] sample retrieval query:\n")
    print(sample_query)

    results = retrieve_top_k(
        model=model,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        query=sample_query,
        top_k=TOP_K,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(results, OUTPUT_DIR / "sample_retrieval_results.json")

    print(f"\n[saved] retrieval results -> {OUTPUT_DIR / 'sample_retrieval_results.json'}")
    print("\n[retrieval] top results:\n")
    for item in results:
        print(f"[{item['rank']}] {item['doc_id']} | score={item['score']}")
        print(item["text"])
        print()


if __name__ == "__main__":
    main()
