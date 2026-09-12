#!/usr/bin/env python3
"""
Aegis-Guard dataset builder (v2)

Purpose
-------
Build a reproducible 3,000-example SFT corpus:
  * 2,000 safety/refusal-oriented prompts
  * 1,000 benign instruction-following prompts

Important
---------
This script deliberately does NOT use keyword heuristics to decide whether a
prompt is harmful. It uses the dedicated Do-Not-Answer and
PKU-SafeRLHF-prompt sources, plus provenance metadata.

The assistant refusals are intentionally varied, but remain policy-level
rather than pretending to know a precise harm category when the source does
not provide one.

Before publication, archive the exact dataset revisions, licenses, hashes,
configuration, and generated manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SEED = 42
MAX_TOKENS = 1024

TARGET_SAFETY = 2000
TARGET_BENIGN = 1000

OUT_DIR = Path("data")
RAW_DIR = OUT_DIR / "raw"
PROCESSED_DIR = OUT_DIR / "processed"
MANIFEST_DIR = OUT_DIR / "manifests"

OUT_JSONL = PROCESSED_DIR / "aegis_safety_sft_v2.jsonl"
MANIFEST_JSON = MANIFEST_DIR / "aegis_safety_sft_v2_manifest.json"

SYSTEM_PROMPT = "You are Aegis-Guard, a secure and compliant assistant."

# These are refusal variants, not category classifiers.
# Use them only as safe target text; do not interpret the wording as an
# independently verified harm taxonomy.
REFUSAL_TEMPLATES = [
    "I can’t help with instructions that would enable harm, abuse, illegal activity, or evasion of safeguards. I can help with safe, lawful, or preventive information instead.",
    "I can’t provide actionable guidance for harming people, violating privacy, committing wrongdoing, or bypassing safety controls. I can help with prevention, protection, or lawful alternatives.",
    "I’m not able to assist with instructions that facilitate harmful or illegal activity. I can help explain the risks, prevention measures, or safer alternatives.",
    "I can’t provide operational instructions for harmful, abusive, or unlawful actions. I can help with safety-focused information or legitimate alternatives.",
    "I can’t assist with requests that would meaningfully facilitate harm, abuse, crime, or circumvention of safeguards. I can help with prevention or safe alternatives.",
    "I’m unable to provide instructions that could enable wrongdoing or cause harm. I can help with defensive, legal, or safety-oriented guidance instead.",
    "I can’t help carry out or conceal harmful or unlawful activity. I can help with risk reduction, prevention, or legitimate ways to address the underlying problem.",
    "I can’t provide step-by-step guidance for harmful or illegal actions. I can help with safe, lawful approaches or information about preventing harm.",
]


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def prompt_key(text: str) -> str:
    return hashlib.sha256(normalize_text(text).lower().encode("utf-8")).hexdigest()


def build_messages(user_text: str, assistant_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def token_length(tokenizer, messages: list[dict[str, str]]) -> int:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def choose_refusal(index: int, prompt: str) -> str:
    # Deterministic lexical diversity without pretending this is semantic
    # category classification.
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    template_idx = (int.from_bytes(digest[:4], "big") + index) % len(REFUSAL_TEMPLATES)
    return REFUSAL_TEMPLATES[template_idx]


def add_example(
    records: list[dict[str, Any]],
    seen: set[str],
    tokenizer,
    user_text: str,
    assistant_text: str,
    *,
    source: str,
    source_id: str | int | None,
    label: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    user_text = normalize_text(user_text)
    assistant_text = normalize_text(assistant_text)

    if not user_text:
        return "empty_prompt"
    if not assistant_text:
        return "empty_response"

    key = prompt_key(user_text)
    if key in seen:
        return "duplicate"

    messages = build_messages(user_text, assistant_text)

    try:
        n_tokens = token_length(tokenizer, messages)
    except Exception as exc:
        raise RuntimeError(f"Tokenizer failure for source={source}, id={source_id}") from exc

    if n_tokens > MAX_TOKENS:
        return "too_long"

    seen.add(key)

    record = {
        "messages": messages,
        "metadata": {
            "source": source,
            "source_id": source_id,
            "label": label,
            "prompt_sha256": key,
            "token_count": n_tokens,
        },
    }

    if metadata:
        record["metadata"].update(metadata)

    records.append(record)
    return "added"


def main() -> None:
    rng = random.Random(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer:", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    print("Loading source datasets...")
    dna = load_dataset("LibrAI/do-not-answer", split="train")
    pku_prompts = load_dataset("PKU-Alignment/PKU-SafeRLHF-prompt", split="train")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    counts = {
        "added": 0,
        "duplicate": 0,
        "empty_prompt": 0,
        "empty_response": 0,
        "too_long": 0,
    }

    source_added = {
        "do_not_answer": 0,
        "pku_saferlhf_prompt": 0,
        "alpaca": 0,
    }

    # -----------------------------------------------------------------------
    # 1) Do-Not-Answer: use all available prompts.
    # The source dataset is explicitly curated for safeguard/refusal
    # evaluation and contains 939 prompts.
    # -----------------------------------------------------------------------
    for i, row in enumerate(dna):
        if source_added["do_not_answer"] >= TARGET_SAFETY:
            break

        response = choose_refusal(i, row.get("question", ""))
        status = add_example(
            records,
            seen,
            tokenizer,
            row.get("question", ""),
            response,
            source="LibrAI/do-not-answer",
            source_id=row.get("id", i),
            label="safety_refusal",
            metadata={
                "risk_area": row.get("risk_area"),
                "types_of_harm": row.get("types_of_harm"),
                "specific_harms": row.get("specific_harms"),
            },
        )
        counts[status] += 1
        if status == "added":
            source_added["do_not_answer"] += 1

    # -----------------------------------------------------------------------
    # 2) PKU prompt dataset.
    # This is a dedicated set of 44.6K unique safety-oriented prompts.
    # Shuffle deterministically and fill the remaining safety target.
    # -----------------------------------------------------------------------
    remaining_safety = TARGET_SAFETY - source_added["do_not_answer"]
    if remaining_safety > 0:
        pku_indices = list(range(len(pku_prompts)))
        rng.shuffle(pku_indices)

        for j, idx in enumerate(pku_indices):
            if source_added["pku_saferlhf_prompt"] >= remaining_safety:
                break

            row = pku_prompts[idx]
            prompt = row.get("prompt", "")
            response = choose_refusal(j + 10000, normalize_text(prompt))

            status = add_example(
                records,
                seen,
                tokenizer,
                prompt,
                response,
                source="PKU-Alignment/PKU-SafeRLHF-prompt",
                source_id=idx,
                label="safety_refusal",
                metadata={
                    "prompt_source": row.get("prompt_source"),
                },
            )
            counts[status] += 1
            if status == "added":
                source_added["pku_saferlhf_prompt"] += 1

    if source_added["do_not_answer"] + source_added["pku_saferlhf_prompt"] != TARGET_SAFETY:
        raise RuntimeError(
            "Could not reach the exact safety target after deduplication/token filtering."
        )

    # -----------------------------------------------------------------------
    # 3) Alpaca benign examples.
    # Use deterministic sampling rather than blindly taking the first 1,000.
    # -----------------------------------------------------------------------
    benign_indices = list(range(len(alpaca)))
    rng.shuffle(benign_indices)

    benign_added = 0
    for idx in benign_indices:
        if benign_added >= TARGET_BENIGN:
            break

        row = alpaca[idx]
        user_text = normalize_text(row.get("instruction", ""))
        extra_input = normalize_text(row.get("input", ""))
        if extra_input:
            user_text = f"{user_text}\n\n{extra_input}"

        status = add_example(
            records,
            seen,
            tokenizer,
            user_text,
            row.get("output", ""),
            source="tatsu-lab/alpaca",
            source_id=idx,
            label="benign",
        )
        counts[status] += 1
        if status == "added":
            benign_added += 1
            source_added["alpaca"] += 1

    if benign_added != TARGET_BENIGN:
        raise RuntimeError(
            "Could not reach the exact benign target after deduplication/token filtering."
        )

    # -----------------------------------------------------------------------
    # 4) Final invariants.
    # -----------------------------------------------------------------------
    assert len(records) == TARGET_SAFETY + TARGET_BENIGN
    assert len(seen) == len(records)

    # Shuffle the final corpus while preserving metadata/provenance.
    rng.shuffle(records)

    # Write JSONL.
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write manifest.
    manifest = {
        "model_id_for_tokenization": MODEL_ID,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "system_prompt": SYSTEM_PROMPT,
        "target_counts": {
            "safety_refusal": TARGET_SAFETY,
            "benign": TARGET_BENIGN,
            "total": TARGET_SAFETY + TARGET_BENIGN,
        },
        "actual_counts": {
            "safety_refusal": source_added["do_not_answer"]
            + source_added["pku_saferlhf_prompt"],
            "benign": source_added["alpaca"],
            "total": len(records),
        },
        "source_counts": source_added,
        "filter_counts": counts,
        "output_file": str(OUT_JSONL),
        "notes": [
            "Safety prompts are selected from dedicated safety-oriented source datasets.",
            "Refusal wording is varied but is not a semantic harm classifier.",
            "Do not use this training file as the held-out evaluation set.",
            "Freeze and archive the exact source dataset revisions before publication.",
        ],
    }

    with MANIFEST_JSON.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Created {OUT_JSONL}")
    print(f"Created {MANIFEST_JSON}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
