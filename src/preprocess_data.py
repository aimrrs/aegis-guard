import json
import os
import re
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer


# ============================================================
# 1. Configuration
# ============================================================

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MAX_TOKENS = 1024

OUTPUT_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "aegis_safety_sft.jsonl")
)

SYSTEM_PROMPT = (
    "You are Aegis-Guard, a secure and compliant assistant."
)

# Simple category-specific refusals.
# These are intentionally short so the dataset remains focused
# on safety behavior rather than long refusal explanations.

REFUSALS = {
    "cybersecurity": (
        "I cannot assist with creating malware, stealing credentials, "
        "bypassing security controls, or conducting harmful cyber attacks."
    ),
    "violence": (
        "I cannot provide instructions for harming, killing, or "
        "physically attacking someone."
    ),
    "hate": (
        "I cannot help promote hatred, discrimination, or harmful "
        "stereotypes against people or groups."
    ),
    "sexual": (
        "I cannot help create or provide explicit sexual content."
    ),
    "dangerous": (
        "I cannot provide instructions that could enable serious harm "
        "or dangerous wrongdoing."
    ),
    "general": (
        "I cannot fulfill this request. I am programmed to follow "
        "safety guidelines."
    ),
}


# ============================================================
# 2. Load tokenizer
# ============================================================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


# ============================================================
# 3. Load datasets
# ============================================================

print("Downloading datasets...")

dna_dataset = load_dataset(
    "LibrAI/do-not-answer",
    split="train"
)

pku_dataset = load_dataset(
    "PKU-Alignment/PKU-SafeRLHF",
    split="train"
)

alpaca_dataset = load_dataset(
    "tatsu-lab/alpaca",
    split="train"
)


# ============================================================
# 4. Dataset storage and statistics
# ============================================================

formatted_data = []

seen_prompts = set()

stats = Counter()

TARGET_ADVERSARIAL = 2000
TARGET_BENIGN = 1000


# ============================================================
# 5. Helper functions
# ============================================================

def clean_text(text):
    """
    Basic text cleaning.

    Removes null bytes, excessive whitespace and invalid
    surrounding whitespace without changing the meaning.
    """

    if text is None:
        return ""

    text = str(text)

    # Remove null characters
    text = text.replace("\x00", "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_for_duplicate_check(text):
    """
    Creates a normalized version of the prompt for duplicate
    detection without changing the actual stored prompt.
    """

    text = clean_text(text).lower()

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text


def detect_category(text):
    """
    Lightweight keyword-based categorization.

    This is NOT intended to be a safety classifier.
    It is only used to select a more appropriate refusal
    response for the SFT dataset.
    """

    text_lower = text.lower()

    cybersecurity_keywords = [
        "malware",
        "ransomware",
        "keylogger",
        "virus",
        "trojan",
        "phishing",
        "credential",
        "password",
        "hack",
        "hacking",
        "exploit",
        "ddos",
        "botnet",
        "bypass security",
        "steal account",
        "steal credentials",
    ]

    violence_keywords = [
        "kill",
        "murder",
        "assassinate",
        "bomb",
        "weapon",
        "shoot",
        "attack",
        "harm someone",
        "hurt someone",
        "poison someone",
    ]

    hate_keywords = [
        "hate group",
        "racial slur",
        "inferior race",
        "inferior people",
        "ethnic group",
        "racist",
        "discriminate",
        "discrimination",
        "stereotype",
    ]

    sexual_keywords = [
        "porn",
        "pornographic",
        "sexually explicit",
        "explicit sexual",
        "erotic story",
        "sexual fantasy",
        "sexual fantasies",
        "sexual positions",
        "adult content",
        "adult movie",
        "sexual acts",
        "fetish",
        "fetishes",
    ]

    dangerous_keywords = [
        "make a bomb",
        "make explosives",
        "poison",
        "weapon construction",
        "dangerous chemical",
        "suicide method",
    ]

    if any(keyword in text_lower for keyword in cybersecurity_keywords):
        return "cybersecurity"

    if any(keyword in text_lower for keyword in violence_keywords):
        return "violence"

    if any(keyword in text_lower for keyword in hate_keywords):
        return "hate"

    if any(keyword in text_lower for keyword in sexual_keywords):
        return "sexual"

    if any(keyword in text_lower for keyword in dangerous_keywords):
        return "dangerous"

    return "general"


def is_valid_text(text):
    """
    Reject obviously invalid or empty text.
    """

    if not text:
        return False

    if len(text.strip()) < 3:
        return False

    # Reject text consisting almost entirely of symbols
    alphanumeric_count = sum(char.isalnum() for char in text)

    if alphanumeric_count < 3:
        return False

    return True


def add_entry(user_text, assistant_text, dataset_type):
    """
    Clean, validate, deduplicate, tokenize and add one sample.

    Returns True only when the sample was actually added.
    """

    user_text = clean_text(user_text)
    assistant_text = clean_text(assistant_text)

    if not is_valid_text(user_text):
        stats["invalid"] += 1
        return False

    if not is_valid_text(assistant_text):
        stats["invalid"] += 1
        return False

    normalized_prompt = normalize_for_duplicate_check(user_text)

    if normalized_prompt in seen_prompts:
        stats["duplicates"] += 1
        return False

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_text
        },
        {
            "role": "assistant",
            "content": assistant_text
        }
    ]

    # Apply Llama chat template
    chat_string = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )

    token_count = len(
        tokenizer(
            chat_string,
            add_special_tokens=False
        )["input_ids"]
    )

    if token_count > MAX_TOKENS:
        stats["oversized"] += 1
        return False

    formatted_data.append(messages)
    seen_prompts.add(normalized_prompt)

    stats[f"{dataset_type}_added"] += 1

    return True


# ============================================================
# 6. Process Do-Not-Answer
# ============================================================

print("Processing Do-Not-Answer dataset...")

for row in dna_dataset:

    if stats["adversarial_added"] >= 1000:
        break

    prompt = row.get("question", "")

    prompt = clean_text(prompt)

    if not prompt:
        continue

    category = detect_category(prompt)
    refusal = REFUSALS.get(category, REFUSALS["general"])

    add_entry(
        prompt,
        refusal,
        "adversarial"
    )


# ============================================================
# 7. Process PKU-SafeRLHF
# ============================================================

print("Processing PKU-SafeRLHF dataset...")

for row in pku_dataset:

    if stats["adversarial_added"] >= TARGET_ADVERSARIAL:
        break

    prompt = row.get("prompt", "")

    prompt = clean_text(prompt)

    if not prompt:
        continue

    category = detect_category(prompt)
    refusal = REFUSALS.get(category, REFUSALS["general"])

    add_entry(
        prompt,
        refusal,
        "adversarial"
    )


# ============================================================
# 8. Process Alpaca benign dataset
# ============================================================

print("Processing Alpaca dataset...")

for row in alpaca_dataset:

    if stats["benign_added"] >= TARGET_BENIGN:
        break

    instruction = clean_text(
        row.get("instruction", "")
    )

    additional_input = clean_text(
        row.get("input", "")
    )

    output = clean_text(
        row.get("output", "")
    )

    if not instruction:
        continue

    user_input = instruction

    if additional_input:
        user_input += "\n\n" + additional_input

    added = add_entry(
        user_input,
        output,
        "benign"
    )

    # Only successful additions count toward the target.
    # add_entry already updates the statistics.


# ============================================================
# 9. Final validation
# ============================================================

print()
print("=" * 55)
print("Aegis-Guard Dataset Statistics")
print("=" * 55)

print(
    f"Adversarial samples added : "
    f"{stats['adversarial_added']}"
)

print(
    f"Benign samples added      : "
    f"{stats['benign_added']}"
)

print(
    f"Duplicates removed        : "
    f"{stats['duplicates']}"
)

print(
    f"Oversized samples removed : "
    f"{stats['oversized']}"
)

print(
    f"Invalid samples removed   : "
    f"{stats['invalid']}"
)

print(
    f"Final dataset size        : "
    f"{len(formatted_data)}"
)

print("=" * 55)


# ============================================================
# 10. Create output directory
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


# ============================================================
# 11. Export JSONL
# ============================================================

print()
print(f"Saving dataset to:")
print(OUTPUT_FILE)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for entry in formatted_data:
        f.write(
            json.dumps(
                entry,
                ensure_ascii=False
            ) + "\n"
        )


# ============================================================
# 12. Final confirmation
# ============================================================

print()
print("Dataset curation complete!")
print("Ready for Milestone 2.")