"""
Run GRADE experiments locally (bypasses Modal).

Usage:
    # Full paper experiment (all 4 methods, 250 steps each):
    python run_grade.py --method all

    # Only GRADE-STE (paper recommended):
    python run_grade.py --method ste

    # Quick test:
    python run_grade.py --method ste --max_steps 50
"""

import argparse
from training_grade import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GRADE experiments")
    parser.add_argument("--method", type=str, default="all",
                        choices=["gumbel", "ste", "ppo", "reinforce", "all"])
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/Qwen3-4B")
    parser.add_argument("--dataset_path", type=str, default="/root/autodl-tmp/imdb_dataset")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        method=args.method,
        base_model=args.model_path,
        dataset_path=args.dataset_path,
        max_steps=args.max_steps,
    )
