from datasets import load_dataset
from .config import SYSTEM_PROMPT


def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }


def load_math_dataset():
    dataset = load_dataset(
        "AI-MO/NuminaMath-TIR",
        name="default",
        split=["train", "test"],
    )

    dataset = {
        "train": dataset[0],
        "test": dataset[1],
    }

    for split in dataset:
        dataset[split] = dataset[split].map(make_conversation)
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    return dataset
