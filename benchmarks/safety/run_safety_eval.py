from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.model.inference import generate_response
from src.safety.evaluator import evaluate_response


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

    return records


def run_evaluation(
    model,
    tokenizer,
    test_file: Path,
    output_file: Path,
    limit: int | None = None,
):
    records = load_jsonl(test_file)

    if limit is not None:
        records = records[:limit]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = []

    print(f"Loaded {len(records)} evaluation records.")
    print()

    for index, record in enumerate(records, start=1):
        messages = record["messages"]

        user_message = next(
            message["content"]
            for message in messages
            if message["role"] == "user"
        )
        
        metadata = record.get("metadata", {})
        label = metadata.get("label", "unknown")

        start_time = time.perf_counter()

        result = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=user_message,
            max_new_tokens=256,
        )

        elapsed = time.perf_counter() - start_time

        evaluation = evaluate_response(
            response=result["response"],
            label=label,
        )

        row = {
            "index": index,
            "label": label,
            "source": metadata.get("source"),
            "prompt": user_message,
            "response": result["response"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["total_tokens"],
            "latency_seconds": elapsed,
            **evaluation,
        }

        results.append(row)

        print(
            f"[{index}/{len(records)}] "
            f"{label:16s} "
            f"{elapsed:.2f}s "
            f"{evaluation['preliminary_outcome']}"
        )

    with output_file.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Records evaluated: {len(results)}")
    print(f"Raw results:       {output_file}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/splits/test.jsonl"),
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("results/raw/b0_safety_outputs.jsonl"),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of records for smoke testing.",
    )

    args = parser.parse_args()

    # The model/tokenizer are intentionally supplied by the notebook
    # or execution environment in the first implementation phase.
    raise RuntimeError(
        "Run run_evaluation(...) from the B0 notebook for now. "
        "The CLI model-loading layer will be added after the smoke test."
    )


if __name__ == "__main__":
    main()
