from __future__ import annotations

from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 128,
    do_sample: bool = False,
) -> dict:
    """
    Generate a response from an instruction-tuned causal language model.

    Returns:
        A dictionary containing the prompt, response, and token counts.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # Move inputs to the same device as the model.
    inputs = inputs.to(model.device)

    input_token_count = inputs.shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][input_token_count:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return {
        "prompt": prompt,
        "response": response,
        "input_tokens": input_token_count,
        "output_tokens": len(generated_tokens),
        "total_tokens": len(outputs[0]),
    }