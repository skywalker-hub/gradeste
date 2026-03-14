"""
Run GRADE experiments on GSM8K.

Usage:
    # Full experiment (all 4 methods):
    python run_grade_math.py --method all

    # Only GRADE-STE (recommended):
    python run_grade_math.py --method ste

    # Quick test:
    python run_grade_math.py --method ste --max_steps 50
"""

import argparse
from training_grade_math import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GRADE experiments on GSM8K")
    parser.add_argument("--method", type=str, default="all",
                        choices=["gumbel", "ste", "ppo", "reinforce", "all"])
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-1.5B")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Local path to GSM8K dataset (optional, downloads from HF if not set)")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./results_math")
    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        method=args.method,
        base_model=args.model_path,
        dataset_path=args.dataset_path,
        max_steps=args.max_steps,
    )
