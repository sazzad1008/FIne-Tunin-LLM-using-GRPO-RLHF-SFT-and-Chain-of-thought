import argparse
import inspect
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from .config import COLD_START_SYSTEM_PROMPT, FEW_SHOT_COT_EXAMPLES, MODEL_NAME
from .data_prep import load_math_dataset
from .utils import get_device, get_dtype_for_device


def extract_final_answer(solution_text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^}]*)\}", solution_text)
    if boxed:
        return boxed[-1].strip()

    marker = re.findall(r"####\s*(.+)", solution_text)
    if marker:
        return marker[-1].strip()

    lines = [line.strip() for line in solution_text.splitlines() if line.strip()]
    return lines[-1] if lines else solution_text.strip()


def make_cold_start_text(example, tokenizer):
    messages = [{"role": "system", "content": COLD_START_SYSTEM_PROMPT}]

    for ex in FEW_SHOT_COT_EXAMPLES:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append(
            {
                "role": "assistant",
                "content": f"<think>{ex['think']}</think>\n<answer>{ex['answer']}</answer>",
            }
        )

    messages.append({"role": "user", "content": example["problem"]})
    messages.append(
        {
            "role": "assistant",
            "content": (
                f"<think>{example['solution']}</think>\n"
                f"<answer>{extract_final_answer(example['solution'])}</answer>"
            ),
        }
    )

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


def build_cold_start_dataset(tokenizer, limit: int = 256):
    dataset = load_math_dataset()
    train = dataset["train"]
    limit = min(limit, len(train))
    train = train.select(range(limit))
    train = train.map(lambda ex: make_cold_start_text(ex, tokenizer), remove_columns=train.column_names)
    return train


def run_cold_start(output_dir: str, limit: int = 256):
    device = get_device()
    dtype = get_dtype_for_device(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)

    train_dataset = build_cold_start_dataset(tokenizer=tokenizer, limit=limit)

    raw_args = {
        "output_dir": output_dir,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-5,
        "logging_steps": 5,
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 1,
        "report_to": "none",
        "bf16": False,
        "fp16": False,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
    }

    accepted_args = set(inspect.signature(TrainingArguments.__init__).parameters)
    sft_args = TrainingArguments(**{k: v for k, v in raw_args.items() if k in accepted_args})

    sft_init_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    trainer_kwargs = {
        "model": model,
        "args": sft_args,
        "train_dataset": train_dataset,
    }

    if "tokenizer" in sft_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in sft_init_params:
        trainer_kwargs["processing_class"] = tokenizer

    if "dataset_text_field" in sft_init_params:
        trainer_kwargs["dataset_text_field"] = "text"
    elif "formatting_func" in sft_init_params:
        trainer_kwargs["formatting_func"] = lambda ex: ex["text"]

    if "max_seq_length" in sft_init_params:
        trainer_kwargs["max_seq_length"] = 1024
    if "packing" in sft_init_params:
        trainer_kwargs["packing"] = False

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cold-start SFT training")
    parser.add_argument("--output-dir", default="./outputs/cold_start_sft")
    parser.add_argument("--limit", type=int, default=256)
    args = parser.parse_args()

    run_cold_start(output_dir=args.output_dir, limit=args.limit)
