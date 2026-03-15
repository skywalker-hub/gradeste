"""
GRADE-Math: Gumbel-Softmax BPTT for Math Reasoning (GSM8K)
============================================================
Core idea: Replace reward model with direct CE loss on answer tokens.
The Gumbel-Softmax chain maintains differentiability through reasoning,
enabling BPTT from answer loss through the entire reasoning chain.

Architecture:
  prompt → Gumbel-Softmax reasoning → detect #### → teacher-forced answer CE loss → backward()
  gradient flows: CE_loss → answer logits → attention → soft_embeds → Gumbel-Softmax → θ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import wandb
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, List
import numpy as np
from collections import defaultdict
import json
from pathlib import Path
import re

# ============================================================================
# CONSTANTS
# ============================================================================

ANSWER_DELIMITER = "\\boxed{"

SYSTEM_PROMPT = (
    "Solve the math problem step by step. Be concise. "
    "Put your final numerical answer inside \\boxed{}. "
    "Example: The total is 3 + 4 = 7. \\boxed{7}"
)

# ============================================================================
# CONFIGURATION 训练设置
# ============================================================================

@dataclass
class Config:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training
    learning_rate: float = 1e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_steps: int = 50
    eval_every: int = 1

    # Generation
    max_prompt_length: int = 256
    max_reasoning_tokens: int = 512
    min_reasoning_tokens: int = 16
    max_answer_tokens: int = 16
    max_eval_tokens: int = 512

    # Gumbel-Softmax
    tau_start: float = 2.0
    tau_end: float = 0.5
    tau_anneal_steps: int = 2000
    gumbel_topk: int = 256

    # PPO specific
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 0.99

    # Regularization
    kl_coef: float = 0.1

    # Infrastructure
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "./results-gsm8k"
    method: str = "all"

    # Data splits
    train_size: int = 6000


# ============================================================================
# MATH UTILITIES
# ============================================================================

def extract_answer_from_text(text: str) -> Optional[str]:
    """Extract numerical answer from generated text using multiple patterns."""

    # Pattern 1: \boxed{number} (highest priority)
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        ans = m.group(1).replace(",", "").strip()
        if ans:
            return ans

    # Pattern 2: "The (final) answer is NUMBER"
    m = re.search(r"[Tt]he (?:final )?answer is[:\s]*(-?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "").strip()

    # Pattern 3: #### followed directly by a number (not by words)
    matches = list(re.finditer(r"####\s*(-?\d[\d,]*\.?\d*)", text))
    if matches:
        return matches[-1].group(1).replace(",", "").strip()

    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.replace(",", "").strip()
    if "." in answer:
        answer = answer.rstrip("0").rstrip(".")
    return answer


def compute_math_reward(
    response_ids: torch.Tensor, tokenizer, gt_answer: str
) -> float:
    """Binary reward: 1.0 if correct, 0.0 otherwise."""
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    predicted = extract_answer_from_text(text)
    if predicted is not None:
        if normalize_answer(predicted) == normalize_answer(gt_answer):
            return 1.0
    return 0.0


# ============================================================================
# GSM8K DATA MANAGEMENT
# ============================================================================

class GSM8KDataset(Dataset):
    def __init__(self, examples: List[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def gsm8k_collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "answer_token_ids": torch.stack([b["answer_token_ids"] for b in batch]),
        "answer_len": torch.tensor([b["answer_len"] for b in batch]),
        "answer_str": [b["answer_str"] for b in batch],
    }


class GSM8KDataSplits:
    """
    Split strategy:
    - Train: first train_size samples from GSM8K train
    - Val:   remaining GSM8K train samples (~1473)
    - Test:  entire GSM8K test split (1319), evaluated only at the end
    """

    def __init__(
        self, config: Config, tokenizer, dataset_path: Optional[str] = None
    ):
        self.config = config
        self.tokenizer = tokenizer

        if dataset_path and Path(dataset_path).exists():
            dataset = load_from_disk(dataset_path)
        else:
            dataset = load_dataset("openai/gsm8k", "main")

        raw_train = list(dataset["train"])
        raw_test = list(dataset["test"])

        print("Processing GSM8K examples...")
        train_examples = [self._process(ex) for ex in tqdm(raw_train, desc="train")]
        test_examples = [self._process(ex) for ex in tqdm(raw_test, desc="test")]

        train_size = min(config.train_size, len(train_examples))
        self.train_data = GSM8KDataset(train_examples[:train_size])
        self.val_data = GSM8KDataset(train_examples[train_size:])
        self.test_data = GSM8KDataset(test_examples)

        print(
            f"GSM8K splits — train: {len(self.train_data)}, "
            f"val: {len(self.val_data)}, test: {len(self.test_data)}"
        )

    def _process(self, example: dict) -> dict:
        question = example["question"]
        answer_text = example["answer"]
        # GSM8K raw data always uses "####" as delimiter (regardless of our ANSWER_DELIMITER)
        final_answer = normalize_answer(answer_text.split("####")[-1])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.config.max_prompt_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Encode "answer}" — the \boxed{ prefix is the delimiter (forced/detected),
        # so we only teacher-force the number + closing brace
        answer_tokens = self.tokenizer.encode(
            final_answer + "}", add_special_tokens=False
        )
        answer_tokens = answer_tokens[: self.config.max_answer_tokens]
        answer_len = len(answer_tokens)
        padded = answer_tokens + [self.tokenizer.pad_token_id] * (
            self.config.max_answer_tokens - answer_len
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "answer_token_ids": torch.tensor(padded, dtype=torch.long),
            "answer_len": answer_len,
            "answer_str": final_answer,
        }

    def get_train_dataloader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=gsm8k_collate_fn,
            drop_last=True,
        )

    def get_val_dataloader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=gsm8k_collate_fn,
        )

    def get_test_dataloader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=gsm8k_collate_fn,
        )


# ============================================================================
# GUMBEL-SOFTMAX UTILITIES
# ============================================================================

def gumbel_softmax(
    logits: torch.Tensor, tau: float = 1.0, hard: bool = False
) -> torch.Tensor:
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y_soft = F.softmax((logits + gumbels) / tau, dim=-1)
    if hard:
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return (y_hard - y_soft).detach() + y_soft
    return y_soft


def gumbel_softmax_topk(
    logits: torch.Tensor, tau: float = 1.0, k: int = 256, hard: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Memory-efficient Gumbel-Softmax over top-k logits only."""
    topk_logits, topk_indices = logits.topk(k, dim=-1)
    gumbels = -torch.log(-torch.log(torch.rand_like(topk_logits) + 1e-10) + 1e-10)
    y_soft_topk = F.softmax((topk_logits + gumbels) / tau, dim=-1)
    y_soft = torch.zeros_like(logits).scatter_(-1, topk_indices, y_soft_topk)
    if hard:
        local_argmax = y_soft_topk.argmax(dim=-1, keepdim=True)
        global_argmax = topk_indices.gather(-1, local_argmax)
        y_hard = torch.zeros_like(logits).scatter_(-1, global_argmax, 1.0)
        return (y_hard - y_soft).detach() + y_soft, topk_indices
    return y_soft, topk_indices


# ============================================================================
# DIFFERENTIABLE GENERATOR
# ============================================================================

class DifferentiableGenerator(nn.Module):
    def __init__(self, model: nn.Module, tokenizer, config: Config):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.embedding = model.get_input_embeddings()

    def get_tau(self, step: int) -> float:
        if step >= self.config.tau_anneal_steps:
            return self.config.tau_end
        ratio = step / self.config.tau_anneal_steps
        return self.config.tau_start - ratio * (
            self.config.tau_start - self.config.tau_end
        )


# ============================================================================
# GRADE MATH TRAINER (CORE)
# ============================================================================

class GumbelMathTrainer:
    """
    GRADE for math: Gumbel-Softmax BPTT with CE loss on answer tokens.

    Phase 1 — Reasoning: generate tokens via Gumbel-Softmax (differentiable).
    Phase 2 — Answer:    teacher-force ground-truth answer, compute CE loss.
    Phase 3 — BPTT:      backward() flows gradients through the full chain.
    """

    def __init__(
        self,
        generator: DifferentiableGenerator,
        ref_model: nn.Module,
        tokenizer,
        config: Config,
        use_ste: bool = False,
    ):
        self.generator = generator
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.use_ste = use_ste
        self.topk = config.gumbel_topk if config.gumbel_topk > 0 else 256

        self.optimizer = torch.optim.AdamW(
            generator.model.parameters(), lr=config.learning_rate
        )
        self.step_count = 0
        self.accum_count = 0

        self.delimiter_text = ANSWER_DELIMITER  # e.g. "\\boxed{"
        self.delimiter_token_ids = tokenizer.encode(
            ANSWER_DELIMITER, add_special_tokens=False
        )
        print(
            f"Delimiter '{ANSWER_DELIMITER}' → token IDs: "
            f"{self.delimiter_token_ids} "
            f"(decoded: '{tokenizer.decode(self.delimiter_token_ids)}')"
        )

    def _policy_forward_step(self, embeds, mask):
        outputs = self.generator.model(
            inputs_embeds=embeds, attention_mask=mask, use_cache=False
        )
        return outputs.logits[:, -1, :]

    def _check_delimiter(self, history: List[int]) -> bool:
        if len(history) < 2:
            return False
        # Decode recent tokens and check for delimiter text
        # This handles tokenization differences (e.g. "\boxed" as 1 vs 2 tokens)
        recent_text = self.tokenizer.decode(history[-8:], skip_special_tokens=True)
        return self.delimiter_text in recent_text

    def step(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        answer_token_ids: torch.Tensor,
        answer_len: torch.Tensor,
    ) -> dict:
        tau = self.generator.get_tau(self.step_count)
        batch_size = prompt_ids.shape[0]
        device = prompt_ids.device

        torch.cuda.empty_cache()

        # ── Phase 1: Gumbel-Softmax reasoning ─────────────────────────
        policy_embeds = self.generator.embedding(prompt_ids)
        policy_mask = prompt_mask.clone()

        with torch.no_grad():
            ref_out = self.ref_model(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                use_cache=True,
            )
            ref_past_kv = ref_out.past_key_values
            del ref_out

        hard_token_history: List[int] = []
        kl_sum = 0.0
        found_delimiter = False
        reasoning_len = 0

        for step_idx in range(self.config.max_reasoning_tokens):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                policy_logits = torch.utils.checkpoint.checkpoint(
                    self._policy_forward_step,
                    policy_embeds,
                    policy_mask,
                    use_reentrant=False,
                ).float()

            soft_token, _ = gumbel_softmax_topk(
                policy_logits, tau=tau, k=self.topk, hard=self.use_ste
            )

            hard_token = soft_token.argmax(dim=-1)
            hard_token_history.append(hard_token[0].item())
            reasoning_len += 1

            # Online KL (detached — monitoring only)
            with torch.no_grad():
                ref_mask_len = prompt_ids.shape[1] + step_idx
                ref_out = self.ref_model(
                    input_ids=hard_token.unsqueeze(-1),
                    attention_mask=torch.cat(
                        [
                            policy_mask[:, :ref_mask_len],
                            torch.ones(batch_size, 1, device=device),
                        ],
                        dim=1,
                    ),
                    past_key_values=ref_past_kv,
                    use_cache=True,
                )
                ref_logits = ref_out.logits[:, -1, :].float()
                ref_past_kv = ref_out.past_key_values

                p_log = F.log_softmax(policy_logits.detach(), dim=-1)
                r_log = F.log_softmax(ref_logits, dim=-1)
                kl_sum += (p_log.exp() * (p_log - r_log)).sum(-1).mean().item()
                del ref_logits

            # Differentiable embedding update (the key link in the BPTT chain)
            next_embed = (
                soft_token.to(self.generator.embedding.weight.dtype)
                @ self.generator.embedding.weight
            ).unsqueeze(1)
            policy_embeds = torch.cat([policy_embeds, next_embed], dim=1)
            policy_mask = torch.cat(
                [policy_mask, torch.ones(batch_size, 1, device=device)], dim=1
            )

            del policy_logits, soft_token

            if (
                step_idx >= self.config.min_reasoning_tokens
                and self._check_delimiter(hard_token_history)
            ):
                found_delimiter = True
                break

        # Force delimiter if model didn't produce it
        if not found_delimiter:
            delim_ids = torch.tensor(
                [self.delimiter_token_ids], device=device
            )
            delim_embeds = self.generator.embedding(delim_ids)
            policy_embeds = torch.cat([policy_embeds, delim_embeds], dim=1)
            policy_mask = torch.cat(
                [
                    policy_mask,
                    torch.ones(
                        batch_size,
                        len(self.delimiter_token_ids),
                        device=device,
                    ),
                ],
                dim=1,
            )

        del ref_past_kv
        torch.cuda.empty_cache()

        # ── Phase 2: Teacher-forced answer CE loss ─────────────────────
        n_ans = answer_len[0].item()
        answer_ce = torch.tensor(0.0, device=device, requires_grad=True)

        for t in range(n_ans):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits_t = torch.utils.checkpoint.checkpoint(
                    self._policy_forward_step,
                    policy_embeds,
                    policy_mask,
                    use_reentrant=False,
                ).float()

            target = answer_token_ids[0, t].unsqueeze(0)
            answer_ce = answer_ce + F.cross_entropy(logits_t, target)

            # Teacher forcing: feed ground truth for next step
            gt_embed = self.generator.embedding(target).unsqueeze(1)
            policy_embeds = torch.cat([policy_embeds, gt_embed], dim=1)
            policy_mask = torch.cat(
                [policy_mask, torch.ones(batch_size, 1, device=device)],
                dim=1,
            )
            del logits_t

        # ── Phase 3: Backward (BPTT through reasoning chain) ──────────
        loss = answer_ce / max(n_ans, 1)
        scaled = loss / self.config.gradient_accumulation_steps
        scaled.backward()

        self.accum_count += 1
        grad_norms: List[float] = []

        if self.accum_count >= self.config.gradient_accumulation_steps:
            grad_norms = [
                p.grad.norm().item()
                for p in self.generator.model.parameters()
                if p.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(
                self.generator.model.parameters(), 1.0
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.accum_count = 0

        self.step_count += 1

        # Decode reasoning tokens for debug output
        reasoning_text = self.tokenizer.decode(
            hard_token_history, skip_special_tokens=True
        )

        result = {
            "loss": loss.item(),
            "kl": kl_sum / max(reasoning_len, 1),
            "tau": tau,
            "reasoning_len": reasoning_len,
            "found_delimiter": float(found_delimiter),
            "grad_norm_mean": np.mean(grad_norms) if grad_norms else 0,
            "grad_norm_std": np.std(grad_norms) if grad_norms else 0,
            "reasoning_text": reasoning_text,
        }

        del answer_ce, policy_embeds, loss
        torch.cuda.empty_cache()
        return result


# ============================================================================
# PPO MATH TRAINER
# ============================================================================

class PPOMathTrainer:
    """PPO baseline adapted for math: binary reward from answer correctness."""

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        tokenizer,
        config: Config,
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.tokenizer = tokenizer
        self.config = config

        hidden_size = policy.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1).to(
            device=config.device, dtype=torch.bfloat16
        )
        self.optimizer = torch.optim.AdamW(
            list(policy.parameters()) + list(self.value_head.parameters()),
            lr=config.learning_rate,
        )

    def _compute_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = 0 if t == len(rewards) - 1 else values[t + 1]
            delta = (
                rewards[t]
                + self.config.gamma * next_val * masks[t]
                - values[t]
            )
            advantages[t] = last_gae = (
                delta
                + self.config.gamma * self.config.gae_lambda * masks[t] * last_gae
            )
        return advantages, advantages + values

    def step(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        answer_str: List[str],
    ) -> dict:
        max_gen = self.config.max_reasoning_tokens + self.config.max_answer_tokens

        with torch.no_grad():
            response_ids = self.policy.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=max_gen,
                min_new_tokens=self.config.min_reasoning_tokens,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response_mask = (response_ids != self.tokenizer.pad_token_id).long()
        prompt_len = prompt_ids.shape[1]
        gen_tokens = response_ids[:, prompt_len:]
        gen_mask = (gen_tokens != self.tokenizer.pad_token_id).float()

        # Binary math reward
        with torch.no_grad():
            reward_list = []
            for i in range(response_ids.shape[0]):
                r = compute_math_reward(
                    response_ids[i], self.tokenizer, answer_str[i]
                )
                reward_list.append(r)
            rewards = torch.tensor(
                reward_list, device=prompt_ids.device, dtype=torch.float32
            )

        # Old log-probs and values
        with torch.no_grad():
            old_out = self.policy(
                response_ids,
                attention_mask=response_mask,
                output_hidden_states=True,
            )
            old_logits = old_out.logits[:, prompt_len - 1 : -1, :]
            old_log_probs = F.log_softmax(old_logits, dim=-1)
            old_token_lp = old_log_probs.gather(
                -1, gen_tokens.unsqueeze(-1)
            ).squeeze(-1)

            old_hidden = old_out.hidden_states[-1][:, prompt_len - 1 : -1, :]
            old_values = self.value_head(old_hidden).squeeze(-1)

            ref_out = self.ref_policy(
                response_ids, attention_mask=response_mask
            )
            ref_logits = ref_out.logits[:, prompt_len - 1 : -1, :]
            ref_lp = F.log_softmax(ref_logits, dim=-1)
            ref_token_lp = ref_lp.gather(
                -1, gen_tokens.unsqueeze(-1)
            ).squeeze(-1)

        log_ratio_kl = old_token_lp - ref_token_lp
        kl_per_seq = (log_ratio_kl * gen_mask).sum(1) / gen_mask.sum(1).clamp(
            min=1
        )

        token_rewards = torch.zeros_like(old_token_lp)
        token_rewards[:, -1] = rewards - self.config.kl_coef * kl_per_seq

        advantages, returns = self._compute_advantages(
            token_rewards, old_values, gen_mask
        )
        adv_mean = (advantages * gen_mask).sum() / gen_mask.sum().clamp(min=1)
        adv_std = (
            ((advantages - adv_mean).pow(2) * gen_mask).sum()
            / gen_mask.sum().clamp(min=1)
        )
        advantages = (advantages - adv_mean) / (adv_std.sqrt() + 1e-8)

        total_loss = 0.0
        all_grad_norms: List[float] = []

        for _ in range(self.config.ppo_epochs):
            out = self.policy(
                response_ids,
                attention_mask=response_mask,
                output_hidden_states=True,
            )
            new_logits = out.logits[:, prompt_len - 1 : -1, :]
            new_lp = F.log_softmax(new_logits, dim=-1)
            new_token_lp = new_lp.gather(
                -1, gen_tokens.unsqueeze(-1)
            ).squeeze(-1)

            new_hidden = out.hidden_states[-1][:, prompt_len - 1 : -1, :]
            new_values = self.value_head(new_hidden).squeeze(-1)

            log_ratio = torch.clamp(new_token_lp - old_token_lp, -2.0, 2.0)
            ratio = torch.exp(log_ratio)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(
                    ratio,
                    1 - self.config.ppo_clip,
                    1 + self.config.ppo_clip,
                )
                * advantages
            )
            policy_loss = -(
                (torch.min(surr1, surr2) * gen_mask).sum()
                / gen_mask.sum().clamp(min=1)
            )

            value_loss = (
                (new_values - returns).pow(2) * gen_mask
            ).sum() / gen_mask.sum().clamp(min=1)

            new_probs = new_lp.exp()
            entropy = (
                -(new_probs * new_lp).sum(-1) * gen_mask
            ).sum() / gen_mask.sum().clamp(min=1)

            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy
            )
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            self.optimizer.zero_grad()
            loss.backward()
            gn = [
                p.grad.norm().item()
                for p in self.policy.parameters()
                if p.grad is not None
            ]
            all_grad_norms.extend(gn)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return {
            "loss": total_loss / self.config.ppo_epochs,
            "reward": rewards.mean().item(),
            "kl": kl_per_seq.mean().item(),
            "grad_norm_mean": np.mean(all_grad_norms) if all_grad_norms else 0,
        }


# ============================================================================
# REINFORCE MATH TRAINER
# ============================================================================

class REINFORCEMathTrainer:
    """REINFORCE baseline with binary math reward."""

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        tokenizer,
        config: Config,
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = torch.optim.AdamW(
            policy.parameters(), lr=config.learning_rate
        )
        self.baseline = 0.0
        self.baseline_momentum = 0.9

    def step(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        answer_str: List[str],
    ) -> dict:
        max_gen = self.config.max_reasoning_tokens + self.config.max_answer_tokens

        with torch.no_grad():
            response_ids = self.policy.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=max_gen,
                min_new_tokens=self.config.min_reasoning_tokens,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response_mask = (response_ids != self.tokenizer.pad_token_id).long()
        prompt_len = prompt_ids.shape[1]
        gen_tokens = response_ids[:, prompt_len:]
        gen_mask = (gen_tokens != self.tokenizer.pad_token_id).float()
        seq_lengths = gen_mask.sum(dim=1).clamp(min=1)

        with torch.no_grad():
            reward_list = []
            for i in range(response_ids.shape[0]):
                r = compute_math_reward(
                    response_ids[i], self.tokenizer, answer_str[i]
                )
                reward_list.append(r)
            rewards = torch.tensor(
                reward_list, device=prompt_ids.device, dtype=torch.float32
            )

        out = self.policy(response_ids, attention_mask=response_mask)
        logits = out.logits[:, prompt_len - 1 : -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)
        seq_lp = (token_lp * gen_mask).sum(dim=1) / seq_lengths

        with torch.no_grad():
            ref_out = self.ref_policy(
                response_ids, attention_mask=response_mask
            )
            ref_logits = ref_out.logits[:, prompt_len - 1 : -1, :]
            ref_lp = F.log_softmax(ref_logits, dim=-1)
            ref_token_lp = ref_lp.gather(
                -1, gen_tokens.unsqueeze(-1)
            ).squeeze(-1)
            kl_per_seq = (
                (token_lp - ref_token_lp) * gen_mask
            ).sum(1) / seq_lengths

        with torch.no_grad():
            adjusted = rewards - self.config.kl_coef * kl_per_seq
            advantage = adjusted - self.baseline
            advantage = (advantage - advantage.mean()) / (
                advantage.std() + 1e-8
            )

        loss = -(seq_lp * advantage).mean()
        self.baseline = (
            self.baseline_momentum * self.baseline
            + (1 - self.baseline_momentum) * rewards.mean().item()
        )

        self.optimizer.zero_grad()
        loss.backward()
        grad_norms = [
            p.grad.norm().item()
            for p in self.policy.parameters()
            if p.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "reward": rewards.mean().item(),
            "kl": kl_per_seq.mean().item(),
            "grad_norm_mean": np.mean(grad_norms) if grad_norms else 0,
            "grad_norm_std": np.std(grad_norms) if grad_norms else 0,
        }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_math(
    model: nn.Module,
    tokenizer,
    dataloader: DataLoader,
    config: Config,
    num_samples: int = 100,
    split_name: str = "val",
) -> dict:
    """Evaluate math accuracy via exact-match on extracted answers."""
    model.eval()
    correct = 0
    total = 0
    samples: List[dict] = []
    max_gen = config.max_eval_tokens

    with torch.no_grad():
        for batch in dataloader:
            if total >= num_samples:
                break

            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            gt_answers = batch["answer_str"]

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

            for i in range(len(outputs)):
                prompt_len = input_ids.shape[1]
                gen_ids = outputs[i][prompt_len:]
                generated = tokenizer.decode(gen_ids, skip_special_tokens=True)
                predicted = extract_answer_from_text(generated)

                is_correct = predicted is not None and normalize_answer(
                    predicted
                ) == normalize_answer(gt_answers[i])
                if is_correct:
                    correct += 1
                total += 1

                if len(samples) < 5:
                    samples.append(
                        {
                            "generated": generated[:1500],
                            "predicted": predicted,
                            "ground_truth": gt_answers[i],
                            "correct": is_correct,
                        }
                    )

    model.train()
    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "samples": samples,
        "split": split_name,
    }


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_method(
    method: Literal["gumbel", "ste", "ppo", "reinforce"],
    config: Config,
    data_splits: GSM8KDataSplits,
) -> dict:
    """Train one method with train/val/test evaluation."""

    display_names = {
        "gumbel": "GRADE",
        "ste": "GRADE-STE",
        "ppo": "PPO",
        "reinforce": "REINFORCE",
    }
    display_name = display_names.get(method, method)

    print(f"\n{'=' * 60}")
    print(f"Training: {display_name}")
    print(f"{'=' * 60}\n")

    run = wandb.init(
        project="grade-math-gsm8k",
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
        elif "llama" in config.base_model.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
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

    train_loader = data_splits.get_train_dataloader(config.batch_size)
    val_loader = data_splits.get_val_dataloader(config.batch_size * 4)
    test_loader = data_splits.get_test_dataloader(config.batch_size * 4)

    # Initialize trainer
    if method in ("gumbel", "ste"):
        generator = DifferentiableGenerator(policy, tokenizer, config)
        trainer = GumbelMathTrainer(
            generator,
            ref_model,
            tokenizer,
            config,
            use_ste=(method == "ste"),
        )
    elif method == "ppo":
        trainer = PPOMathTrainer(policy, ref_model, tokenizer, config)
    elif method == "reinforce":
        trainer = REINFORCEMathTrainer(policy, ref_model, tokenizer, config)

    results = defaultdict(list)
    step = 0
    best_val_acc = 0.0

    pbar = tqdm(total=config.max_steps, desc=f"Training {display_name}")

    while step < config.max_steps:
        for batch in train_loader:
            if step >= config.max_steps:
                break

            prompt_ids = batch["input_ids"].to(config.device)
            prompt_mask = batch["attention_mask"].to(config.device)

            if method in ("gumbel", "ste"):
                answer_token_ids = batch["answer_token_ids"].to(config.device)
                answer_len = batch["answer_len"].to(config.device)
                metrics = trainer.step(
                    prompt_ids, prompt_mask, answer_token_ids, answer_len
                )
            else:
                answer_str = batch["answer_str"]
                metrics = trainer.step(prompt_ids, prompt_mask, answer_str)

            for k, v in metrics.items():
                results[k].append(v)

            # Log non-text metrics to wandb
            log_metrics = {
                k: v for k, v in metrics.items() if k != "reasoning_text"
            }
            wandb.log(
                {f"{display_name}/train/{k}": v for k, v in log_metrics.items()},
                step=step,
            )

            # Print training step details
            gt_ans = batch["answer_str"][0] if "answer_str" in batch else "?"
            print(
                f"\n{'─' * 60}"
                f"\n[Step {step}] loss={metrics['loss']:.3f}  "
                f"reasoning_len={metrics.get('reasoning_len', '?')}  "
                f"delim={metrics.get('found_delimiter', '?')}  "
                f"tau={metrics.get('tau', '?'):.2f}"
            )
            if "reasoning_text" in metrics:
                print(f"  GT answer: {gt_ans}")
                print(f"  Reasoning: {metrics['reasoning_text'][:1200]}")

            # Periodic validation
            if step % config.eval_every == 0:
                eval_model = (
                    policy
                    if method in ("ppo", "reinforce")
                    else trainer.generator.model
                )
                val_res = evaluate_math(
                    eval_model,
                    tokenizer,
                    val_loader,
                    config,
                    num_samples=5,
                    split_name="val",
                )
                wandb.log(
                    {
                        f"{display_name}/val/accuracy": val_res["accuracy"],
                        f"{display_name}/val/correct": val_res["correct"],
                    },
                    step=step,
                )
                results["val_accuracy"].append(val_res["accuracy"])

                if val_res["accuracy"] > best_val_acc:
                    best_val_acc = val_res["accuracy"]
                    results["best_val_accuracy"] = best_val_acc
                    results["best_val_step"] = step

                print(
                    f"\n  === Val ({val_res['correct']}/{val_res['total']} correct) ==="
                )
                for si, s in enumerate(val_res["samples"]):
                    mark = "✓" if s["correct"] else "✗"
                    print(
                        f"  [{si+1}] {mark}  pred={s['predicted']}  gt={s['ground_truth']}"
                    )
                    print(f"      {s['generated'][:1200]}")
                    print()

            step += 1
            pbar.update(1)
            loss_str = f"{metrics['loss']:.3f}"
            extra = (
                f"reward={metrics.get('reward', 'N/A')}"
                if "reward" in metrics
                else f"delim={metrics.get('found_delimiter', 'N/A')}"
            )
            pbar.set_postfix({"loss": loss_str, "info": extra})

    pbar.close()

    # Final test evaluation
    print(f"\n{'=' * 40}")
    print(f"FINAL TEST — {display_name}")
    print(f"{'=' * 40}")

    eval_model = (
        policy
        if method in ("ppo", "reinforce")
        else trainer.generator.model
    )
    test_res = evaluate_math(
        eval_model,
        tokenizer,
        test_loader,
        config,
        num_samples=500,
        split_name="test",
    )
    wandb.log(
        {
            f"{display_name}/test/accuracy": test_res["accuracy"],
            f"{display_name}/test/correct": test_res["correct"],
        }
    )

    results["test_eval"] = test_res
    print(
        f"  Test Accuracy: {test_res['accuracy']:.4f} "
        f"({test_res['correct']}/{test_res['total']})"
    )
    print(
        f"  Best Val Accuracy: {best_val_acc:.4f} "
        f"(step {results.get('best_val_step', 'N/A')})"
    )

    # Save
    output_path = Path(config.output_dir) / method
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "results.json", "w") as f:
        serializable = {
            k: (
                [
                    float(x) if isinstance(x, (np.floating, float)) else x
                    for x in v
                ]
                if isinstance(v, list)
                else v
            )
            for k, v in results.items()
        }
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

@dataclass
class Arguments:
    method: str = "all"
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_steps: int = 50
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-5
    output_dir: str = "./results-gsm8k"
    seed: int = 42


def main(
    output_dir: str = "./results-gsm8k",
    method: Optional[str] = None,
    base_model: Optional[str] = None,
    dataset_path: Optional[str] = None,
    max_steps: Optional[int] = None,
):
    args = Arguments(output_dir=output_dir)
    if method:
        args.method = method
    if base_model:
        args.base_model = base_model
    if max_steps:
        args.max_steps = max_steps

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = Config(
        base_model=args.base_model,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        config.device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Print delimiter tokenization for verification
    delim_tokens = tokenizer.encode(ANSWER_DELIMITER, add_special_tokens=False)
    print(f"\nDelimiter '{ANSWER_DELIMITER}' → token IDs: {delim_tokens}")
    print(f"  Decoded back: '{tokenizer.decode(delim_tokens)}'")
    print(f"  Num tokens: {len(delim_tokens)}\n")

    data_splits = GSM8KDataSplits(config, tokenizer, dataset_path=dataset_path)

    # Run experiments
    methods = (
        ["gumbel", "ste", "ppo", "reinforce"]
        if args.method == "all"
        else [args.method]
    )

    all_results = {}
    for m in methods:
        results = train_method(m, config, data_splits)
        all_results[m] = results

    # Summary
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'=' * 60}")

    display_names = {
        "gumbel": "GRADE",
        "ste": "GRADE-STE",
        "ppo": "PPO",
        "reinforce": "REINFORCE",
    }
    for m, res in all_results.items():
        test = res.get("test_eval", {})
        name = display_names.get(m, m)
        print(f"\n{name}:")
        print(
            f"  TEST Accuracy:     "
            f"{test.get('accuracy', 'N/A'):.4f} "
            f"({test.get('correct', '?')}/{test.get('total', '?')})"
        )
        print(
            f"  Best VAL Accuracy: "
            f"{res.get('best_val_accuracy', 'N/A'):.4f} "
            f"(step {res.get('best_val_step', 'N/A')})"
        )

    with open(Path(config.output_dir) / "comparison.json", "w") as f:
        json.dump(
            {
                m: {
                    "test_accuracy": res.get("test_eval", {}).get("accuracy"),
                    "best_val_accuracy": res.get("best_val_accuracy"),
                    "best_val_step": res.get("best_val_step"),
                }
                for m, res in all_results.items()
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {config.output_dir}/")


if __name__ == "__main__":
    main()
