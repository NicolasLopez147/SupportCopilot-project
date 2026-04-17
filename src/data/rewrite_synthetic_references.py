import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from src.utils.paths import DATA_DIR


DEFAULT_INPUT_PATH = DATA_DIR / "synthetic" / "synthetic_reply_eval.jsonl"
DEFAULT_OUTPUT_PATH = DATA_DIR / "synthetic" / "synthetic_reply_eval_rewritten.jsonl"
KB_DIR = DATA_DIR / "kb"
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL_NAME = "llama3.2"
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite synthetic reference replies into a more consistent style."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the synthetic dataset. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to save the rewritten dataset. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Ollama model name to use. Default: {DEFAULT_MODEL_NAME}",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_kb_documents(kb_dir: Path) -> dict[str, str]:
    kb = {}
    for path in kb_dir.glob("*.md"):
        kb[path.stem] = path.read_text(encoding="utf-8").strip()
    return kb


def format_conversation(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        speaker = str(message.get("speaker", "")).strip().lower()
        text = str(message.get("text", "")).strip()
        label = "Customer" if speaker == "customer" else "Agent"
        if text:
            lines.append(f"{label}: {text}")
    return "\n".join(lines)


def build_prompt(scenario: str, kb_text: str, conversation_text: str) -> str:
    return f"""
You are rewriting a reference reply for a synthetic customer support benchmark.

Your task is to write exactly one high-quality next agent reply for the conversation below.
Use the support knowledge as the source of truth.

Requirements:
- Return valid JSON only.
- Output format:
{{
  "reference_reply": "..."
}}
- Write one short professional support reply.
- Keep it to 1 or 2 sentences.
- It must describe the next best step only.
- Keep the tone neutral, clear, and operational.
- Prefer wording like "The next step is to...", "I will continue by...", or "We will need to...".
- Do not include "Agent:".
- Do not mention internal tools, internal teams, secure links, one-time passwords, or processes unless they are explicitly supported by the knowledge.
- Do not invent checks that have already been completed unless the conversation says so.
- Do not introduce new policies or escalation paths not supported by the knowledge.
- If customer information is needed, ask only for the minimum necessary detail and do not request highly sensitive information.

Scenario:
{scenario}

Support knowledge:
{kb_text}

Conversation:
{conversation_text}
""".strip()


def call_ollama(prompt: str, model_name: str) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
        },
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        body = json.loads(response.read().decode("utf-8"))

    return body["response"].strip()


def extract_reference_reply(text: str) -> str:
    parsed = json.loads(text)
    reference_reply = parsed.get("reference_reply")
    if not isinstance(reference_reply, str) or not reference_reply.strip():
        raise ValueError("Invalid reference_reply in model response.")

    reply = reference_reply.strip()
    reply = re.sub(r"^(Agent|Customer)\s*:\s*", "", reply, flags=re.IGNORECASE)
    reply = re.sub(r"\s+", " ", reply)
    return reply.strip()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_path)
    kb_documents = load_kb_documents(KB_DIR)

    rewritten_rows = []

    print(f"[data] loaded samples: {len(rows)}")
    print(f"[model] using ollama model: {args.model}")

    for index, row in enumerate(rows, start=1):
        scenario = row.get("scenario")
        kb_text = kb_documents.get(scenario)
        if not kb_text:
            raise KeyError(f"No KB document found for scenario: {scenario}")

        conversation_text = format_conversation(row.get("messages", []))
        prompt = build_prompt(
            scenario=scenario,
            kb_text=kb_text,
            conversation_text=conversation_text,
        )

        print(f"[rewrite] sample {index}/{len(rows)} -> {row.get('conversation_id')}")

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw_response = call_ollama(prompt, args.model)
                rewritten_reply = extract_reference_reply(raw_response)
                new_row = dict(row)
                new_row["reference_reply"] = rewritten_reply
                new_row["reference_rewritten_by"] = args.model
                rewritten_rows.append(new_row)
                break
            except (
                ValueError,
                KeyError,
                json.JSONDecodeError,
                urllib.error.URLError,
                TimeoutError,
            ) as exc:
                last_error = exc
                print(
                    f"[retry] conversation={row.get('conversation_id')} "
                    f"attempt={attempt}/{MAX_RETRIES} -> {exc}"
                )
                time.sleep(1.5)
        else:
            raise RuntimeError(
                f"Failed to rewrite reference for {row.get('conversation_id')}: {last_error}"
            )

    save_jsonl(rewritten_rows, args.output_path)
    print(f"[saved] rewritten dataset -> {args.output_path}")


if __name__ == "__main__":
    main()
