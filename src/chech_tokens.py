import json
from pathlib import Path
from transformers import AutoTokenizer

DATASET_PATH = Path("D:\\aegis-guard\\data\\aegis_safety_sft_final.jsonl")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_TOKENS = 1024

print("Loading Llama-3.1 tokenizer...")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )
except Exception as e:
    print("\nERROR: Could not load the Llama-3.1 tokenizer.")
    print("Make sure you have access to the model/tokenizer.")
    print(f"\nDetails: {e}")
    raise SystemExit(1)


total_records = 0
oversized_records = []
max_tokens = 0
max_record_number = None


with open(DATASET_PATH, "r", encoding="utf-8") as f:

    for line_number, line in enumerate(f, start=1):

        line = line.strip()

        if not line:
            continue

        record = json.loads(line)

        total_records += 1

        # Combine actual message contents.
        text = ""

        for message in record:
            content = message.get("content", "")
            text += content + "\n"

        # Tokenize the actual text.
        tokens = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False
        )

        token_count = len(tokens["input_ids"])

        if token_count > max_tokens:
            max_tokens = token_count
            max_record_number = line_number

        if token_count > MAX_TOKENS:
            oversized_records.append(
                (line_number, token_count)
            )


print("\n========================================")
print("Aegis-Guard Token Length Validation")
print("========================================")
print(f"Total records       : {total_records}")
print(f"Maximum token count : {max_tokens}")

if max_record_number:
    print(f"Longest record      : {max_record_number}")

print(f"Oversized records   : {len(oversized_records)}")
print("========================================")


if oversized_records:

    print("\nFAIL: Oversized records found.")

    print("\nFirst 20 oversized records:")

    for record_number, token_count in oversized_records[:20]:
        print(
            f"Record {record_number}: "
            f"{token_count} tokens"
        )

else:

    print("\nPASS: All records are within 1024 tokens.")
