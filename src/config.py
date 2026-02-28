MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

OUTPUT_DIR = "./outputs"
COLD_START_DIR = "./outputs/cold_start_sft"
GRPO_OUT_DIR = "./outputs/grpo"

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
)

COLD_START_SYSTEM_PROMPT = (
    "You are a math tutor. Solve step-by-step, then give the final result. "
    "Use <think>...</think> for reasoning and <answer>...</answer> for final answer."
)

FEW_SHOT_COT_EXAMPLES = [
    {
        "question": "What is 2 + 3 * 4?",
        "think": "Apply order of operations: 3*4=12, then 2+12=14.",
        "answer": "14",
    },
    {
        "question": "Solve 5x - 10 = 0.",
        "think": "Add 10 on both sides to get 5x=10, then divide by 5.",
        "answer": "x = 2",
    },
]
