from __future__ import annotations

import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


def load_base_model(
    model_id: str = MODEL_ID,
    hf_token: str | None = None,
):
    """
    Load the base Llama model and tokenizer for Aegis-Guard experiments.

    B0 configuration:
        - Base Llama-3.1-8B-Instruct
        - 4-bit NF4 quantization
        - No LoRA
        - No constrained decoding

    Returns:
        tuple: (model, tokenizer)
    """

    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "Hugging Face token not found. "
            "Provide hf_token or set the HF_TOKEN environment variable."
        )

    print(f"Loading tokenizer: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer loaded.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading 4-bit base model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )

    model.eval()

    print("Base model loaded successfully.")

    return model, tokenizer
