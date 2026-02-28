import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import COLD_START_SYSTEM_PROMPT
from .utils import get_device


def run_eval(model_path: str, prompt: str):
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    messages = [
        {"role": "system", "content": COLD_START_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick eval")
    parser.add_argument("--model", default="./outputs/grpo")
    parser.add_argument("--prompt", default="If 3x + 5 = 20, what is x?")
    args = parser.parse_args()

    run_eval(model_path=args.model, prompt=args.prompt)
