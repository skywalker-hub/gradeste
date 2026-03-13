"""
Quick standalone script to run GRADE on a cloud GPU (bypasses Modal).

Usage:
    # If model & dataset are available online (HuggingFace):
    python run_grade.py

    # If model & dataset are downloaded locally (no internet):
    python run_grade.py --model_path /data/models/Qwen3-4B --dataset_path /data/datasets/imdb_dataset
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from training_grade import (
    Config, DataSplits, DifferentiableRewardModel, train_method,
    AutoTokenizer,
)


def main():
    parser = argparse.ArgumentParser(description="Run GRADE training")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/Qwen3-4B",
                        help="HuggingFace model name or local path")
    parser.add_argument("--dataset_path", type=str, default="/root/autodl-tmp/imdb_dataset",
                        help="Local path to IMDB dataset saved via save_to_disk")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Training steps (50 for quick test, 250+ for real run)")
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = Config(
        base_model=args.model_path,
        max_steps=args.max_steps,
        batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        output_dir=args.output_dir,
        seed=seed,
        eval_every=25,
    )
    config.wandb_project = "grade-vs-ppo"

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU (will be very slow)")
        config.device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    data_splits = DataSplits(config, tokenizer, dataset_path=args.dataset_path)

    rm_save_path = Path(args.output_dir) / "reward_model.pt"
    reward_model = DifferentiableRewardModel(
        base_model_name=config.base_model,
        generator_tokenizer=tokenizer,
        device=config.device,
        rm_dataloader=data_splits.get_rm_dataloader(batch_size=16),
        pretrained_rm_path=str(rm_save_path) if rm_save_path.exists() else None,
    )

    if not rm_save_path.exists():
        rm_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(reward_model.model.classifier.state_dict(), rm_save_path)

    results = train_method("gumbel", config, data_splits, reward_model)

    test = results.get("test_eval", {})
    print("\n" + "=" * 60)
    print("GRADE RESULTS")
    print("=" * 60)
    print(f"  TEST Reward:     {test.get('mean_reward', 'N/A')}")
    print(f"  Best VAL Reward: {results.get('best_val_reward', 'N/A')} "
          f"(step {results.get('best_val_step', 'N/A')})")
    print(f"\nResults saved to {args.output_dir}/gumbel/")


if __name__ == "__main__":
    main()
