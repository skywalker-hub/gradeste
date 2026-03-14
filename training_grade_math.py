"""
GRADE for GSM8K: Differentiable RL for Mathematical Reasoning
=============================================================

Adapts GRADE (differentiable RL via Gumbel-Softmax) from sentiment to math reasoning.

Key differences from IMDB version:
1. ORM (Outcome Reward Model) trained on GSM8K ground-truth solutions
   (zero GPU cost) — reward model reads full CoT, providing dense
   gradient signal to the reasoning process
2. Math-specific evaluation (exact match accuracy)
3. Longer generation for chain-of-thought reasoning
4. Lower KL penalty to allow more exploration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset
from tqdm import tqdm
import wandb
from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np
from collections import defaultdict
import json
import re
import random
from pathlib import Path

from training_grade import (
    Config as BaseConfig,
    gumbel_softmax,
    gumbel_softmax_topk,
    DifferentiableGenerator,
    SameVocabRewardModel,
    train_reward_model,
    DifferentiableRewardModel,
    GumbelTrainerMemoryEfficient,
    GumbelTrainer,
    PPOTrainer,
    REINFORCETrainer,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MathConfig(BaseConfig):
    base_model: str = "Qwen/Qwen3-1.5B"

    max_new_tokens: int = 128
    min_new_tokens: int = 16

    tau_start: float = 1.5
    tau_end: float = 0.3
    tau_anneal_steps: int = 3000
    gumbel_topk: int = 128

    kl_coef: float = 0.05

    output_dir: str = "./results_math"

    # Data splits (GSM8K train has 7473 samples)
    orm_questions: int = 2000
    policy_train_size: int = 4000
    val_size: int = 1000
    rm_train_size: int = 2000  # alias for orm_questions in parent

    # ORM data: negative samples per ground-truth solution
    orm_negatives_per_question: int = 3
    orm_cache_dir: str = "./orm_cache"


# ============================================================================
# UTILITIES
# ============================================================================

def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from model output.

    Supports formats:
      - "#### 42"
      - "the answer is 42"
      - Falls back to last number in text
    """
    # Primary: #### separator
    matches = re.findall(r'####\s*([\-]?\d[\d,]*\.?\d*)', text)
    if matches:
        return matches[-1].replace(",", "").strip()

    # Fallback: "the answer is <number>"
    matches = re.findall(r'(?:answer|result)\s*(?:is|=)\s*([\-]?\d[\d,]*\.?\d*)', text, re.I)
    if matches:
        return matches[-1].replace(",", "").strip()

    # Last resort: last number in text
    numbers = re.findall(r'[\-]?\d[\d,]*\.?\d*', text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return None


def format_math_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer: Let's solve this step by step.\n"


# ============================================================================
# DATA MANAGEMENT
# ============================================================================

class MathDataSplits:
    """
    Manages data splits for GSM8K.

    Split strategy:
    - ORM Training:     First orm_questions from GSM8K train → generate solutions, label by correctness
    - Policy Training:  Next policy_train_size from GSM8K train → prompts for Gumbel/PPO/REINFORCE
    - Validation:       Next val_size from GSM8K train → monitor accuracy during training
    - Test:             GSM8K test split → final evaluation only
    """

    def __init__(self, config: MathConfig, tokenizer, dataset_path: Optional[str] = None):
        self.config = config
        self.tokenizer = tokenizer

        print("\n" + "=" * 60)
        print("LOADING GSM8K DATA")
        print("=" * 60)

        if dataset_path and Path(dataset_path).exists():
            from datasets import load_from_disk
            print(f"  Loading from local path: {dataset_path}")
            full_dataset = load_from_disk(dataset_path)
        else:
            full_dataset = load_dataset("openai/gsm8k", "main")

        train_data = full_dataset["train"].shuffle(seed=config.seed)
        self.test_data = full_dataset["test"]

        orm_end = config.orm_questions
        policy_end = orm_end + config.policy_train_size
        val_end = policy_end + config.val_size
        assert val_end <= len(train_data), \
            f"Not enough data: need {val_end}, have {len(train_data)}"

        self.orm_data = train_data.select(range(0, orm_end))
        self.policy_data = train_data.select(range(orm_end, policy_end))
        self.val_data = train_data.select(range(policy_end, val_end))

        print(f"  ORM Training:     {len(self.orm_data)} questions")
        print(f"  Policy Training:  {len(self.policy_data)} prompts")
        print(f"  Validation:       {len(self.val_data)} problems")
        print(f"  Test:             {len(self.test_data)} problems (GSM8K test)")
        print("=" * 60 + "\n")

    # ---- ORM DataLoader ----

    def get_rm_dataloader(self, batch_size: int) -> DataLoader:
        """Build or load ORM training data, return DataLoader compatible with train_reward_model."""
        cache_path = Path(self.config.orm_cache_dir)

        if cache_path.exists() and (cache_path / "dataset_info.json").exists():
            print(f"Loading cached ORM data from {cache_path}")
            from datasets import load_from_disk
            orm_dataset = load_from_disk(str(cache_path))
        else:
            orm_dataset = self._generate_orm_data()
            cache_path.mkdir(parents=True, exist_ok=True)
            orm_dataset.save_to_disk(str(cache_path))
            print(f"Cached ORM data to {cache_path}")

        pos = sum(1 for label in orm_dataset["label"] if label == 1)
        print(f"  ORM dataset: {len(orm_dataset)} samples  "
              f"({pos} correct / {len(orm_dataset) - pos} incorrect)")

        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
            )

        orm_dataset = orm_dataset.map(tokenize, batched=True)
        orm_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        return DataLoader(orm_dataset, batch_size=batch_size, shuffle=True)

    def _generate_orm_data(self) -> Dataset:
        """Build ORM data from GSM8K ground-truth solutions — no model sampling needed.

        Positive: question + ground-truth CoT + correct answer  (label=1)
        Negative: question + ground-truth CoT + wrong answer    (label=0)

        The ORM learns: "does the final answer match the reasoning?"
        This costs zero GPU time — pure string manipulation.
        """
        rng = random.Random(self.config.seed)

        all_gt_answers = []
        for item in self.orm_data:
            ans = extract_answer(item["answer"])
            if ans is not None:
                all_gt_answers.append(ans)

        texts, labels = [], []

        for item in self.orm_data:
            question = item["question"]
            solution = item["answer"]
            gt_answer = extract_answer(solution)
            if gt_answer is None:
                continue

            full_text = format_math_prompt(question) + solution

            # Positive: original ground-truth solution
            texts.append(full_text)
            labels.append(1)

            # Negatives: same CoT but with corrupted final answer
            for _ in range(self.config.orm_negatives_per_question):
                wrong = self._make_wrong_answer(gt_answer, all_gt_answers, rng)
                corrupted = re.sub(
                    r'####\s*[\-]?\d[\d,]*\.?\d*',
                    f'#### {wrong}',
                    full_text,
                )
                texts.append(corrupted)
                labels.append(0)

        return Dataset.from_dict({"text": texts, "label": labels})

    @staticmethod
    def _make_wrong_answer(correct: str, answer_pool: list, rng: random.Random) -> str:
        """Generate a plausible but wrong answer."""
        strategies = ["perturb", "swap_from_pool", "arithmetic"]
        strategy = rng.choice(strategies)

        try:
            correct_num = float(correct)
        except ValueError:
            return str(rng.randint(1, 999))

        if strategy == "perturb":
            offset = rng.choice([-3, -2, -1, 1, 2, 3, 5, 10, -10])
            result = correct_num + offset
        elif strategy == "swap_from_pool":
            candidates = [a for a in answer_pool if a != correct]
            if candidates:
                return rng.choice(candidates)
            result = correct_num + rng.randint(1, 10)
        else:  # arithmetic: multiply or divide
            factor = rng.choice([2, 0.5, 10, 0.1])
            result = correct_num * factor

        if correct_num == int(correct_num):
            return str(int(result))
        return f"{result:.2f}"

    # ---- Policy / Val / Test DataLoaders ----

    def _make_prompt_dataloader(self, data, batch_size: int) -> DataLoader:
        def tokenize(examples):
            prompts = [format_math_prompt(q) for q in examples["question"]]
            return self.tokenizer(
                prompts, truncation=True, max_length=256, padding="max_length",
            )

        dataset = data.map(tokenize, batched=True, remove_columns=data.column_names)
        dataset.set_format("torch")
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def get_policy_train_dataloader(self, batch_size: int) -> DataLoader:
        return self._make_prompt_dataloader(self.policy_data, batch_size)

    def get_val_dataloader(self, batch_size: int) -> DataLoader:
        return self._make_prompt_dataloader(self.val_data, batch_size)

    def get_test_dataloader(self, batch_size: int) -> DataLoader:
        return self._make_prompt_dataloader(self.test_data, batch_size)


# ============================================================================
# MATH-SPECIFIC EVALUATION
# ============================================================================

def evaluate_math(
    model: nn.Module,
    reward_model: DifferentiableRewardModel,
    tokenizer,
    raw_data,
    config: MathConfig,
    num_samples: int = 100,
    split_name: str = "val",
) -> dict:
    """Evaluate using hard generation + exact match accuracy."""
    model.eval()
    correct = 0
    total = 0
    reward_sum = 0.0
    samples = []

    with torch.no_grad():
        for i, item in enumerate(raw_data):
            if i >= num_samples:
                break

            prompt = format_math_prompt(item["question"])
            gt_answer = extract_answer(item["answer"])

            encoding = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256,
            )
            input_ids = encoding.input_ids.to(config.device)
            attention_mask = encoding.attention_mask.to(config.device)

            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_new_tokens,
                min_new_tokens=config.min_new_tokens,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            predicted = extract_answer(response)

            is_correct = predicted is not None and predicted == gt_answer
            if is_correct:
                correct += 1
            total += 1

            output_mask = (output_ids != tokenizer.pad_token_id).long()
            rm_reward = reward_model.forward_hard(output_ids, output_mask).item()
            reward_sum += rm_reward

            if len(samples) < 5:
                samples.append({
                    "question": item["question"][:200],
                    "gt": gt_answer,
                    "pred": predicted,
                    "correct": is_correct,
                    "rm_reward": f"{rm_reward:.3f}",
                    "response": response[:500],
                })

    model.train()
    accuracy = correct / total if total > 0 else 0
    mean_reward = reward_sum / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "mean_reward": mean_reward,
        "correct": correct,
        "total": total,
        "samples": samples,
        "split": split_name,
    }


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_method_math(
    method: Literal["gumbel", "ste", "ppo", "reinforce"],
    config: MathConfig,
    data_splits: MathDataSplits,
    reward_model: DifferentiableRewardModel,
) -> dict:
    """Train a single method on GSM8K with proper evaluation."""

    display_names = {
        "gumbel": "GRADE-Math",
        "ste": "GRADE-STE-Math",
        "ppo": "PPO-Math",
        "reinforce": "REINFORCE-Math",
    }
    display_name = display_names.get(method, method)

    print(f"\n{'=' * 60}")
    print(f"Training: {display_name}")
    print(f"{'=' * 60}\n")

    run = wandb.init(
        project=config.wandb_project if hasattr(config, "wandb_project") else "grade-math",
        name=display_name,
        config=vars(config),
        reinit=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(config.device)
    base_model.gradient_checkpointing_enable()

    if config.use_lora:
        if "qwen" in config.base_model.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "pythia" in config.base_model.lower():
            target_modules = ["query_key_value"]
        elif "gpt2" in config.base_model.lower():
            target_modules = ["c_attn", "c_proj"]
        else:
            target_modules = ["q_proj", "v_proj"]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
        )
        policy = get_peft_model(base_model, lora_config)
        policy.print_trainable_parameters()
    else:
        policy = base_model

    ref_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(config.device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    train_loader = data_splits.get_policy_train_dataloader(config.batch_size)
    # Raw data for exact-match evaluation
    val_raw = data_splits.val_data
    test_raw = data_splits.test_data

    # Initialize trainer (same as IMDB — trainers are task-agnostic)
    if method == "gumbel":
        generator = DifferentiableGenerator(policy, tokenizer, config)
        trainer = GumbelTrainerMemoryEfficient(
            generator, ref_model, reward_model, tokenizer, config, use_ste=False,
        )
    elif method == "ste":
        generator = DifferentiableGenerator(policy, tokenizer, config)
        trainer = GumbelTrainerMemoryEfficient(
            generator, ref_model, reward_model, tokenizer, config, use_ste=True,
        )
    elif method == "ppo":
        trainer = PPOTrainer(policy, ref_model, reward_model, tokenizer, config)
    elif method == "reinforce":
        trainer = REINFORCETrainer(policy, ref_model, reward_model, tokenizer, config)
    else:
        raise ValueError(f"Unknown method: {method}")

    results = defaultdict(list)
    step = 0
    best_val_acc = 0.0
    pbar = tqdm(total=config.max_steps, desc=f"Training {display_name}")

    while step < config.max_steps:
        for batch in train_loader:
            if step >= config.max_steps:
                break

            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)

            metrics = trainer.step(input_ids, attention_mask)

            for k, v in metrics.items():
                results[k].append(v)
            wandb.log({f"{display_name}/train/{k}": v for k, v in metrics.items()}, step=step)

            # Periodic validation: both RM reward and exact-match accuracy
            if step % config.eval_every == 0:
                eval_model = policy if method in ("ppo", "reinforce") else trainer.generator.model
                val_results = evaluate_math(
                    eval_model, reward_model, tokenizer, val_raw, config,
                    num_samples=50, split_name="val",
                )

                wandb.log({
                    f"{display_name}/val/accuracy": val_results["accuracy"],
                    f"{display_name}/val/rm_reward": val_results["mean_reward"],
                }, step=step)

                results["val_accuracy"].append(val_results["accuracy"])
                results["val_reward"].append(val_results["mean_reward"])

                if val_results["accuracy"] > best_val_acc:
                    best_val_acc = val_results["accuracy"]
                    results["best_val_accuracy"] = best_val_acc
                    results["best_val_step"] = step

                tqdm.write(
                    f"  [Step {step}] Val Accuracy: {val_results['accuracy']:.3f}  "
                    f"RM Reward: {val_results['mean_reward']:.3f}"
                )

                if val_results["samples"]:
                    wandb.log({
                        f"{display_name}/val/samples": wandb.Table(
                            columns=["question", "gt", "pred", "correct", "response"],
                            data=[[s["question"], s["gt"], s["pred"],
                                   s["correct"], s["response"]] for s in val_results["samples"]],
                        )
                    }, step=step)

            step += 1
            pbar.update(1)
            pbar.set_postfix({
                "reward": f"{metrics['reward']:.3f}",
                "loss": f"{metrics['loss']:.3f}",
            })

    pbar.close()

    # ---- Final test evaluation ----
    print(f"\n{'=' * 40}")
    print(f"FINAL TEST: {display_name}")
    print(f"{'=' * 40}")

    eval_model = policy if method in ("ppo", "reinforce") else trainer.generator.model
    test_results = evaluate_math(
        eval_model, reward_model, tokenizer, test_raw, config,
        num_samples=200, split_name="test",
    )

    wandb.log({
        f"{display_name}/test/accuracy": test_results["accuracy"],
        f"{display_name}/test/rm_reward": test_results["mean_reward"],
    })
    results["test_eval"] = test_results

    print(f"  Test Accuracy:   {test_results['accuracy']:.4f}  "
          f"({test_results['correct']}/{test_results['total']})")
    print(f"  Test RM Reward:  {test_results['mean_reward']:.4f}")
    print(f"  Best Val Acc:    {best_val_acc:.4f} (step {results.get('best_val_step', 'N/A')})")

    if test_results["samples"]:
        print("\n  Sample predictions:")
        for s in test_results["samples"][:3]:
            mark = "✓" if s["correct"] else "✗"
            print(f"    {mark} Q: {s['question']}")
            print(f"      GT: {s['gt']}  Pred: {s['pred']}")

    # Save results
    output_path = Path(config.output_dir) / method
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "results.json", "w") as f:
        serializable = {}
        for k, v in results.items():
            if isinstance(v, list):
                serializable[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
            else:
                serializable[k] = v
        json.dump(serializable, f, indent=2, default=str)

    if method in ("gumbel", "ste"):
        trainer.generator.model.save_pretrained(output_path / "model")
    else:
        policy.save_pretrained(output_path / "model")

    wandb.finish()
    return dict(results)


# ============================================================================
# MAIN
# ============================================================================

def main(
    output_dir: str = "./results_math",
    method: Optional[str] = None,
    base_model: Optional[str] = None,
    dataset_path: Optional[str] = None,
    max_steps: Optional[int] = None,
):
    config = MathConfig(output_dir=output_dir)
    if method:
        config.method = method
    if base_model:
        config.base_model = base_model
    if max_steps:
        config.max_steps = max_steps
    config.wandb_project = "grade-math"

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        config.device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ---- Step 1: Load and split GSM8K data ----
    data_splits = MathDataSplits(config, tokenizer, dataset_path=dataset_path)

    # ---- Step 2: Train ORM (or load from cache) ----
    # ORM training data is auto-generated by sampling from base model
    rm_save_path = Path(config.output_dir) / "reward_model_math.pt"
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

    # ---- Step 3: Train policy methods ----
    methods = ["gumbel", "ste", "ppo", "reinforce"] if config.method == "all" else [config.method]

    all_results = {}
    for m in methods:
        results = train_method_math(m, config, data_splits, reward_model)
        all_results[m] = results

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY (GSM8K)")
    print("=" * 60)

    display_names = {
        "gumbel": "GRADE-Math", "ste": "GRADE-STE-Math",
        "ppo": "PPO-Math", "reinforce": "REINFORCE-Math",
    }
    for m, res in all_results.items():
        test = res.get("test_eval", {})
        name = display_names.get(m, m)
        print(f"\n{name}:")
        print(f"  Test Accuracy:   {test.get('accuracy', 'N/A')}")
        print(f"  Test RM Reward:  {test.get('mean_reward', 'N/A')}")
        print(f"  Best Val Acc:    {res.get('best_val_accuracy', 'N/A')} "
              f"(step {res.get('best_val_step', 'N/A')})")

    with open(Path(config.output_dir) / "comparison.json", "w") as f:
        json.dump({
            m: {
                "test_accuracy": res.get("test_eval", {}).get("accuracy"),
                "test_rm_reward": res.get("test_eval", {}).get("mean_reward"),
                "best_val_accuracy": res.get("best_val_accuracy"),
                "best_val_step": res.get("best_val_step"),
            }
            for m, res in all_results.items()
        }, f, indent=2)

    print(f"\nResults saved to {config.output_dir}/")


if __name__ == "__main__":
    main()
