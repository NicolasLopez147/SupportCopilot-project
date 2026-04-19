import argparse
from pathlib import Path

from src.copilot.pipeline.service import (
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    SupportCopilotService,
    load_samples,
    reset_feedback_memory,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full SupportCopilot pipeline on a conversation."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "Path to a JSONL dataset or a single JSON conversation object. "
            f"Default: {DEFAULT_INPUT_PATH}"
        ),
    )
    parser.add_argument(
        "--conversation-id",
        default=None,
        help="Optional conversation_id to select from a JSONL file. Defaults to the first sample.",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Process all conversations in a JSONL file instead of just one sample.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap when using --run-all.",
    )
    parser.add_argument(
        "--reset-feedback",
        action="store_true",
        help="Clear intent/summary/reply failure memory before running the pipeline batch.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to save pipeline output. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(
        path=args.input_path,
        conversation_id=args.conversation_id,
        run_all=args.run_all,
        limit=args.limit,
    )

    if args.reset_feedback:
        reset_feedback_memory()
        print("[pipeline] feedback memory cleared")

    print("[pipeline] loading SupportCopilot service")
    service = SupportCopilotService()

    results = []
    for index, sample in enumerate(samples, start=1):
        conversation_id = sample.get("conversation_id")
        print(f"[pipeline] sample {index}/{len(samples)} -> {conversation_id}")

        output = service.run_sample(sample=sample, persist_feedback=True)
        results.append(output)

        print(f"[intent_final] {output['intent']['predicted_intent']}")
        print(f"[summary_passed] {output['summary_review']['passed']}")
        print(f"[reply_passed] {output['reply_review']['passed']}")

    output_payload: dict | list
    if len(results) == 1:
        output_payload = results[0]
    else:
        output_payload = results

    save_json(output_payload, args.output_path)
    print(f"[saved] pipeline output -> {args.output_path}")

    intent_failures = sum(1 for item in results if not item["intent_review"]["passed"])
    summary_failures = sum(1 for item in results if not item["summary_review"]["passed"])
    reply_failures = sum(1 for item in results if not item["reply_review"]["passed"])

    print(f"[summary] processed samples: {len(results)}")
    print(f"[summary] intent critic failures: {intent_failures}")
    print(f"[summary] summary critic failures: {summary_failures}")
    print(f"[summary] reply critic failures: {reply_failures}")


if __name__ == "__main__":
    main()
