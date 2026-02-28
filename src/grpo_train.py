import argparse
import math
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

from .data_prep import load_math_dataset
from .rewards import (
    accuracy_reward,
    format_reward,
    reasoning_steps_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
)
from .utils import get_device, maybe_empty_cache


class ScratchGRPOTrainer:
    def __init__(self, model, ref_model, tokenizer, reward_funcs, group_size=2, max_new_tokens=64, temperature=0.7):
        self.model = model
        self.ref_model = ref_model
        self.ref_model.eval()
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    def compute_log_probs(self, model, input_ids, attention_mask, prompt_len):
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_log_probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
        return per_token_log_probs[:, (prompt_len - 1):]

    def train_step(self, batch_prompts, batch_solutions):
        G = self.group_size
        device = self.model.device

        prompt = batch_prompts[0]
        solution = batch_solutions[0]

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=G,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        completion_ids = output_ids[:, prompt_len:]
        completions_text = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        formatted_completions = [[{"content": text}] for text in completions_text]

        rewards = torch.zeros(G, device=device)
        accuracy_scores = None

        for name, func in self.reward_funcs:
            if name == "cosine":
                if accuracy_scores is None:
                    accuracy_scores = accuracy_reward(formatted_completions, solution=[solution] * G)
                scores = func(completions=formatted_completions, solution=[solution] * G, accuracy_rewards=accuracy_scores)
            elif name == "accuracy":
                scores = func(completions=formatted_completions, solution=[solution] * G)
                accuracy_scores = scores
            else:
                scores = func(completions=formatted_completions, solution=[solution] * G)

            rewards += torch.tensor(scores, device=device)

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        curr_log_probs = self.compute_log_probs(self.model, output_ids, None, prompt_len)
        with torch.no_grad():
            ref_log_probs = self.compute_log_probs(self.ref_model, output_ids, None, prompt_len)

        ratio = torch.exp(curr_log_probs - ref_log_probs)
        surrogate_loss = -(ratio * advantages.unsqueeze(1)).mean()

        kl_div = torch.exp(ref_log_probs - curr_log_probs) - (ref_log_probs - curr_log_probs) - 1
        kl_loss = 0.1 * kl_div.mean()

        total_loss = surrogate_loss + kl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        maybe_empty_cache(device)
        return total_loss.item()


def get_reward_functions(names):
    registry = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(),
        "repetition_penalty": get_repetition_penalty_reward(),
    }
    out = []
    for name in names:
        if name not in registry:
            raise ValueError(f"Reward function '{name}' not found in registry.")
        out.append((name, registry[name]))
    return out


def collate_batch(examples):
    return {
        "prompts": [ex["problem"] for ex in examples],
        "solutions": [ex["solution"] for ex in examples],
    }


def run_grpo(base_model: str, limit: int, max_steps: int, group_size: int, max_new_tokens: int, temperature: float):
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True).to(device)

    dataset = load_math_dataset()
    train = dataset["train"].select(range(min(limit, len(dataset["train"]))))

    dataloader = torch.utils.data.DataLoader(
        train,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
    )

    reward_funcs = get_reward_functions(["accuracy"])  # keep simple by default

    trainer = ScratchGRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_funcs=reward_funcs,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    steps = 0
    for batch in dataloader:
        loss = trainer.train_step(batch_prompts=batch["prompts"], batch_solutions=batch["solutions"])
        if steps % 2 == 0:
            print(f"Step {steps} | Loss: {loss:.4f}")
        steps += 1
        if steps >= max_steps:
            break

    output_dir = "./outputs/grpo"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved GRPO model to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO training (debug safe)")
    parser.add_argument("--base-model", default="./outputs/cold_start_sft")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    run_grpo(
        base_model=args.base_model,
        limit=args.limit,
        max_steps=args.max_steps,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
