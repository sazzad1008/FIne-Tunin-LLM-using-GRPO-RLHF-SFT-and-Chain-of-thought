# LLM Fine-Tuning using GRPO (RLHF + SFT + CoT)

This repo mirrors the original notebook workflow as runnable Python modules:

- Cold-start SFT with few-shot long CoT examples
- GRPO-style RLHF loop (lightweight demo trainer)
- Simple evaluation / generation test

## Setup

Create a virtual environment and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Cold-start SFT

```bash
python3 -m src.cold_start_sft --output-dir ./outputs/cold_start_sft --limit 256
```

## GRPO (debug-safe)

```bash
python3 -m src.grpo_train --base-model ./outputs/cold_start_sft --limit 64 --max-steps 20
```

## Quick eval

```bash
python3 -m src.eval --model ./outputs/grpo
```

## Notes

- Defaults are set for Mac safety (small batches, short loops).
- Increase `--limit`, `--max-steps`, and group size once it runs cleanly.
