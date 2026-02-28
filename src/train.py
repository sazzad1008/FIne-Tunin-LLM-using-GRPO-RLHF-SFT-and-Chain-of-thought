import argparse
import os

from .cold_start_sft import run_cold_start
from .grpo_train import run_grpo


def main():
    parser = argparse.ArgumentParser(description="Run cold-start SFT then GRPO")
    parser.add_argument("--cold-start-dir", default="./outputs/cold_start_sft")
    parser.add_argument("--cold-start-limit", type=int, default=256)
    parser.add_argument("--grpo-limit", type=int, default=64)
    parser.add_argument("--grpo-max-steps", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    os.makedirs("./outputs", exist_ok=True)

    run_cold_start(output_dir=args.cold_start_dir, limit=args.cold_start_limit)

    run_grpo(
        base_model=args.cold_start_dir,
        limit=args.grpo_limit,
        max_steps=args.grpo_max_steps,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
