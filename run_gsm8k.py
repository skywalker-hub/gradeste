"""
Run GRADE-Math experiments on GSM8K.

Usage:
    # Full experiment (all 4 methods):
    python run_gsm8k.py --method all

    # Only GRADE-STE (recommended):
    python run_gsm8k.py --method ste

    # Quick test:
    python run_gsm8k.py --method ste --max_steps 50

    # With local model path:
    python run_gsm8k.py --method ste --model_path /path/to/Qwen2.5-1.5B-Instruct
"""

import argparse
from training_gsm8k import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GRADE-Math on GSM8K")
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["gumbel", "ste", "ppo", "reinforce", "all"],
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./results-gsm8k")
    parser.add_argument("--sft_steps", type=int, default=None)
    parser.add_argument(
        "--skip_sft", action="store_true", help="Skip SFT stage"
    )
    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        method=args.method,
        base_model=args.model_path,
        dataset_path=args.dataset_path,
        max_steps=args.max_steps,
        sft_steps=0 if args.skip_sft else args.sft_steps,
    )
