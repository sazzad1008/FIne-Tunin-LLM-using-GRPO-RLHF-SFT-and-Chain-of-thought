import math
import re
from typing import Callable, List

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def accuracy_reward(completions, **kwargs) -> List[float]:
    """
    Reward: 1.0 if completion is mathematically correct, else 0.0.
    Falls back to 0.5 when gold solution parsing fails.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    solutions = kwargs.get("solution")

    for content, sol in zip(contents, solutions):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )

        if gold_parsed:
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            reward = float(verify(answer_parsed, gold_parsed))
        else:
            reward = 0.5
            print("Warning: Failed to parse gold solution:", sol)

        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs) -> List[float]:
    """
    Reward: 1.0 if completion has <think>...</think><answer>...</answer> format, else 0.0.
    """
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def reasoning_steps_reward(completions, **kwargs) -> List[float]:
    """
    Reward for step-by-step reasoning signals.
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content, re.MULTILINE)) for content in completion_contents]
    return [min(1.0, count / 3) for count in matches]


def get_cosine_scaled_reward(
    min_value_wrong: float = -0.5,
    max_value_wrong: float = -0.1,
    min_value_correct: float = 0.8,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
) -> Callable:
    """
    Returns a cosine-scaled reward based on completion length and accuracy reward.
    """
    def cosine_scaled_reward(completions, solution, accuracy_rewards, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol, acc_reward in zip(contents, solution, accuracy_rewards):
            gen_len = len(content)
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if acc_reward > 0.5:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -0.1) -> Callable:
    """
    Returns a repetition penalty reward function.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> List[float]:
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for completion in contents:
            if completion == "" or len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)

        return rewards

    return repetition_penalty_reward
