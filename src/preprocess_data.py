import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer

# 1. Configuration & Setup
# Using the Llama-3.1-8B tokenizer to accurately count tokens
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct" 
MAX_TOKENS = 1024 #
OUTPUT_FILE = "../data/aegis_safety_sft.jsonl" #

# Standard safe refusal response[cite: 1]
SAFE_REFUSAL = "I cannot fulfill this request. I am programmed to follow safety guidelines."
SYSTEM_PROMPT = "You are Aegis-Guard, a secure and compliant assistant." #[cite: 1]

print("Loading tokenizer...")
# Note: You will need to be logged in via `huggingface-cli login` for this to work
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2. Data Collection
print("Downloading datasets...")
# Fetching adversarial datasets[cite: 1]
dna_dataset = load_dataset("Librarian-Bot/do-not-answer", split="train")
pku_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")

# Fetching benign dataset to prevent catastrophic forgetting[cite: 1]
alpaca_dataset = load_dataset("tatsu-lab/alpaca", split="train")

formatted_data = []
seen_prompts = set()

def add_entry(user_text, assistant_text):
    """Formats the entry using the Llama-3.1 chat template and checks constraints."""
    if user_text in seen_prompts:
        return # Skip duplicates[cite: 1]
    
    seen_prompts.add(user_text)
    
    # Official Meta Llama-3.1 Template Structure[cite: 1]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text}
    ]
    
    # Apply chat template and check token length[cite: 1]
    chat_string = tokenizer.apply_chat_template(messages, tokenize=False)
    token_count = len(tokenizer.tokenize(chat_string))
    
    if token_count <= MAX_TOKENS:
        formatted_data.append(messages)

# 3. Processing Adversarial Data (Extracting 2,000 samples)[cite: 1]
print("Processing adversarial datasets...")
malicious_count = 0

for row in dna_dataset:
    if malicious_count >= 1000:
        break
    # Assuming 'question' is the column name for the prompt
    add_entry(row['question'], SAFE_REFUSAL)
    malicious_count += 1

for row in pku_dataset:
    if malicious_count >= 2000:
        break
    # Assuming 'prompt' is the column name 
    add_entry(row['prompt'], SAFE_REFUSAL)
    malicious_count += 1

# 4. Processing Benign Data (Extracting 1,000 samples)[cite: 1]
print("Processing Alpaca dataset...")
benign_count = 0

for row in alpaca_dataset:
    if benign_count >= 1000:
        break
    
    # Combine instruction and input if available
    user_input = row['instruction']
    if row.get('input'):
        user_input += f"\n\n{row['input']}"
        
    add_entry(user_input, row['output'])
    benign_count += 1

# 5. Data Validation & Export[cite: 1]
print(f"Total valid samples processed: {len(formatted_data)}")
print(f"Saving to {OUTPUT_FILE}...")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in formatted_data:
        f.write(json.dumps(entry) + "\n")

print("Dataset curation complete! Ready for Milestone 2.")