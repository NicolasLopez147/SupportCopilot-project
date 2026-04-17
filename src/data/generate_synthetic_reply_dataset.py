import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from src.utils.paths import DATA_DIR, EXPERIMENTS_OUTPUT_DIR


KB_DIR = DATA_DIR / "kb"
OUTPUT_DIR = DATA_DIR / "synthetic"
OUTPUT_PATH = OUTPUT_DIR / "synthetic_reply_eval.jsonl"
PREVIEW_PATH = EXPERIMENTS_OUTPUT_DIR / "data_generation" / "synthetic_reply_preview.json"
FAILURES_PATH = EXPERIMENTS_OUTPUT_DIR / "data_generation" / "synthetic_generation_failures.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL_NAME = "llama3.2"
NUM_SAMPLES_PER_SCENARIO = 3
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 180


def load_kb_documents(kb_dir: Path) -> list[dict]:
    documents = []

    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        documents.append(
            {
                "scenario": path.stem,
                "path": str(path),
                "text": text,
            }
        )

    return documents


def build_prompt(scenario: str, kb_text: str, sample_index: int) -> str:
    return f"""
You are generating synthetic customer support conversations for a retrieval-grounded reply generation benchmark.

Create exactly one realistic conversation for the scenario "{scenario}".
Use the support knowledge below as the ground truth for the scenario.

Requirements:
- The conversation must stay aligned with the support knowledge.
- Use between 3 and 5 messages total.
- Alternate speakers naturally between customer and agent.
- End the conversation before the final agent resolution, so another system can generate the next best agent reply.
- Include realistic but readable customer language.
- Keep the scenario unresolved at the end of the conversation.
- Avoid using exact personal data such as real emails, account numbers, phone numbers, or full addresses.
- Do not add new policies, security procedures, outages, checks, or escalations unless they are clearly supported by the knowledge.
- Prefer plain operational language over corporate phrasing.
- Do not mention the knowledge base explicitly.
- Do not include markdown fences.
- Return valid JSON only.

The JSON object must follow this schema exactly:
{{
  "messages": [
    {{"speaker": "customer", "text": "..."}},
    {{"speaker": "agent", "text": "..."}},
    {{"speaker": "customer", "text": "..."}}
  ],
  "reference_reply": "..."
}}

Constraints for reference_reply:
- It must be the next best agent reply for this conversation.
- Keep it professional, concise, and action-oriented.
- It must be consistent with the support knowledge.
- Do not start it with "Agent:".
- Do not claim that a check has already been completed unless the conversation explicitly says so.
- Prefer stating the next step rather than a final resolution.
- Do not request or reveal sensitive personal data.
- Do not mention internal tools, internal systems, internal teams, or process guarantees unless explicitly stated in the support knowledge.
- Do not promise acceleration, prioritization, or special handling unless explicitly stated in the support knowledge.
- Avoid introducing new actions such as scheduling a call, involving security, using an outage map, or contacting another department unless the support knowledge clearly supports that step.
- Prefer neutral phrasing like "the next step is", "I will continue by", or "we will need to" over stronger claims.

Diversity hint:
- This is variation #{sample_index} for the same scenario, so make it meaningfully different from other likely examples in wording, customer mood, and troubleshooting state.

Support knowledge:
{kb_text}
""".strip()


def call_ollama(prompt: str, model_name: str) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.8,
            "top_p": 0.9,
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


def extract_json_object(text: str) -> dict:
    stripped = text.strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")

    candidate = stripped[start : end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON is not an object.")

    return parsed


def normalize_text(text: str) -> str:
    normalized = text.strip()

    replacements = {
        "{’}": "'",
        "{‘}": "'",
        "{“}": '"',
        "{”}": '"',
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "\u00a0": " ",
    }

    for source, target in replacements.items():
        normalized = normalized.replace(source, target)

    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def validate_example(example: dict, scenario: str, conversation_id: str) -> dict:
    messages = example.get("messages")
    reference_reply = example.get("reference_reply")

    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("Example must contain at least 3 messages.")

    cleaned_messages = []
    valid_speakers = {"customer", "agent"}

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object.")

        speaker = str(message.get("speaker", "")).strip().lower()
        text = normalize_text(str(message.get("text", "")))

        if speaker not in valid_speakers:
            raise ValueError(f"Invalid speaker: {speaker}")
        if not text:
            raise ValueError("Message text cannot be empty.")

        cleaned_messages.append({"speaker": speaker, "text": text})

    if len(cleaned_messages) > 5:
        cleaned_messages = cleaned_messages[:5]

    if cleaned_messages[0]["speaker"] != "customer":
        raise ValueError("Conversation must start with customer.")

    for idx in range(1, len(cleaned_messages)):
        if cleaned_messages[idx]["speaker"] == cleaned_messages[idx - 1]["speaker"]:
            raise ValueError("Speakers must alternate naturally.")

    if not isinstance(reference_reply, str) or not reference_reply.strip():
        raise ValueError("reference_reply must be a non-empty string.")

    reference_reply = normalize_text(reference_reply)
    if reference_reply.lower().startswith("agent:"):
        reference_reply = reference_reply.split(":", 1)[1].strip()

    if "@" in reference_reply:
        raise ValueError("reference_reply should not contain email addresses.")

    for message in cleaned_messages:
        if "@" in message["text"]:
            raise ValueError("messages should not contain email addresses.")

    return {
        "conversation_id": conversation_id,
        "scenario": scenario,
        "messages": cleaned_messages,
        "reference_reply": reference_reply,
        "source": "synthetic_ollama",
        "generator_model": DEFAULT_MODEL_NAME,
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_preview(rows: list[dict], path: Path, limit: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows[:limit], f, indent=2, ensure_ascii=False)


def save_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_examples_for_document(
    document: dict,
    num_samples: int,
    model_name: str,
) -> tuple[list[dict], list[dict]]:
    generated = []
    failures = []

    for sample_index in range(1, num_samples + 1):
        conversation_id = f"{document['scenario']}_{sample_index:03d}"
        prompt = build_prompt(
            scenario=document["scenario"],
            kb_text=document["text"],
            sample_index=sample_index,
        )

        print(
            f"[generate] scenario={document['scenario']} sample={sample_index}/{num_samples}"
        )

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw_response = call_ollama(prompt=prompt, model_name=model_name)
                parsed = extract_json_object(raw_response)
                validated = validate_example(
                    example=parsed,
                    scenario=document["scenario"],
                    conversation_id=conversation_id,
                )
                validated["generator_model"] = model_name
                generated.append(validated)
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
                    f"[retry] scenario={document['scenario']} sample={sample_index} "
                    f"attempt={attempt}/{MAX_RETRIES} -> {exc}"
                )
                time.sleep(1.5)
        else:
            failure = {
                "conversation_id": conversation_id,
                "scenario": document["scenario"],
                "sample_index": sample_index,
                "error": str(last_error),
            }
            failures.append(failure)
            print(
                f"[skip] scenario={document['scenario']} sample={sample_index} "
                f"-> {last_error}"
            )

    return generated, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic reply-generation dataset from the local KB using Ollama."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Ollama model name to use. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--samples-per-scenario",
        type=int,
        default=NUM_SAMPLES_PER_SCENARIO,
        help=f"How many conversations to generate per KB scenario. Default: {NUM_SAMPLES_PER_SCENARIO}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_kb_documents(KB_DIR)
    if not documents:
        raise FileNotFoundError(f"No KB documents found in {KB_DIR}")

    print(f"[kb] documents loaded: {len(documents)}")
    print(f"[model] using ollama model: {args.model}")
    print(f"[config] samples per scenario: {args.samples_per_scenario}")

    all_rows = []
    all_failures = []
    for document in documents:
        rows, failures = generate_examples_for_document(
            document=document,
            num_samples=args.samples_per_scenario,
            model_name=args.model,
        )
        all_rows.extend(rows)
        all_failures.extend(failures)
        write_jsonl(all_rows, OUTPUT_PATH)
        save_preview(all_rows, PREVIEW_PATH)
        save_json(all_failures, FAILURES_PATH)

    write_jsonl(all_rows, OUTPUT_PATH)
    save_preview(all_rows, PREVIEW_PATH)
    save_json(all_failures, FAILURES_PATH)

    print(f"[saved] dataset -> {OUTPUT_PATH}")
    print(f"[saved] preview -> {PREVIEW_PATH}")
    print(f"[saved] failures -> {FAILURES_PATH}")
    print(f"[summary] total synthetic conversations: {len(all_rows)}")
    print(f"[summary] skipped conversations: {len(all_failures)}")


if __name__ == "__main__":
    main()
