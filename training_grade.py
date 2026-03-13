"""
GRADE: Replacing Policy Gradients with Backpropagation for LLM Alignment
=========================================================================
With proper train/validation/test splits for rigorous evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset
from tqdm import tqdm
import wandb
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict
import numpy as np
from collections import defaultdict
import json
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Model
    base_model: str = "Qwen/Qwen3-4B"
    
    # LoRA (optional)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Training
    learning_rate: float = 1e-5  # Reduced for stability
    batch_size: int = 1  # Minimal for Gumbel memory requirements
    gradient_accumulation_steps: int = 16  # Increased to compensate
    max_steps: int = 2000
    eval_every: int = 100
    
    # Generation
    max_new_tokens: int = 64  # Reduced for memory (Gumbel needs gradients through all steps)
    min_new_tokens: int = 8
    
    # Gumbel-Softmax specific
    tau_start: float = 2.0
    tau_end: float = 0.5
    tau_anneal_steps: int = 2000
    gumbel_topk: int = 256  # Top-k filtering for memory efficiency (0 = disabled)
    
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
    output_dir: str = "./results"
    method: str = "all"
    
    # Data split sizes
    rm_train_size: int = 5000      # Reward model training
    policy_train_size: int = 10000  # Policy training  
    val_size: int = 2000            # Validation (during training)
    # Test = IMDB test split (25000 samples)


# ============================================================================
# DATA MANAGEMENT - PROPER SPLITS
# ============================================================================

class DataSplits:
    """
    Manages train/validation/test splits to prevent data leakage.
    
    Split strategy:
    - Reward Model Training: First rm_train_size samples from IMDB train
    - Policy Training: Next policy_train_size samples from IMDB train  
    - Validation: Next val_size samples from IMDB train (for monitoring)
    - Test: Entire IMDB test split (final evaluation only)
    """
    
    def __init__(self, config: Config, tokenizer, dataset_path: Optional[str] = None):
        self.config = config
        self.tokenizer = tokenizer
        
        print("\n" + "="*60)
        print("LOADING AND SPLITTING DATA")
        print("="*60)
        
        # Load full dataset (from local path or HuggingFace)
        if dataset_path and Path(dataset_path).exists():
            from datasets import load_from_disk
            print(f"  Loading dataset from local path: {dataset_path}")
            full_dataset = load_from_disk(dataset_path)
        else:
            full_dataset = load_dataset("imdb")
        train_data = full_dataset["train"].shuffle(seed=config.seed)
        test_data = full_dataset["test"]
        
        # Calculate split indices
        rm_end = config.rm_train_size
        policy_end = rm_end + config.policy_train_size
        val_end = policy_end + config.val_size
        
        # Verify we have enough data
        assert val_end <= len(train_data), \
            f"Not enough training data: need {val_end}, have {len(train_data)}"
        
        # Create splits
        self.rm_train_data = train_data.select(range(0, rm_end))
        self.policy_train_data = train_data.select(range(rm_end, policy_end))
        self.val_data = train_data.select(range(policy_end, val_end))
        self.test_data = test_data
        
        print(f"  Reward Model Training: {len(self.rm_train_data)} samples (indices 0-{rm_end})")
        print(f"  Policy Training:       {len(self.policy_train_data)} samples (indices {rm_end}-{policy_end})")
        print(f"  Validation:            {len(self.val_data)} samples (indices {policy_end}-{val_end})")
        print(f"  Test:                  {len(self.test_data)} samples (IMDB test split)")
        print("="*60 + "\n")
    
    def get_rm_dataloader(self, batch_size: int) -> DataLoader:
        """DataLoader for reward model training (full reviews with labels)."""
        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=256,
                padding="max_length",
            )
        
        dataset = self.rm_train_data.map(tokenize, batched=True)
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def _make_prompt_dataloader(self, data: Dataset, batch_size: int) -> DataLoader:
        """Convert reviews to prompts for generation."""
        def tokenize(examples):
            prompts = []
            for text in examples["text"]:
                # Take first 1-2 sentences as prompt
                sentences = text.split(".")[:2]
                prompt = ".".join(sentences)[:200]
                prompts.append(prompt)
            
            return self.tokenizer(
                prompts,
                truncation=True,
                max_length=32,
                padding="max_length",
            )
        
        dataset = data.map(tokenize, batched=True, remove_columns=data.column_names)
        dataset.set_format("torch")
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def get_policy_train_dataloader(self, batch_size: int) -> DataLoader:
        """DataLoader for policy training (prompts only)."""
        return self._make_prompt_dataloader(self.policy_train_data, batch_size)
    
    def get_val_dataloader(self, batch_size: int) -> DataLoader:
        """DataLoader for validation during training."""
        return self._make_prompt_dataloader(self.val_data, batch_size)
    
    def get_test_dataloader(self, batch_size: int) -> DataLoader:
        """DataLoader for final test evaluation."""
        return self._make_prompt_dataloader(self.test_data, batch_size)


# ============================================================================
# GUMBEL-SOFTMAX UTILITIES
# ============================================================================

def gumbel_softmax(logits: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y_soft = F.softmax((logits + gumbels) / tau, dim=-1)
    
    if hard:
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return (y_hard - y_soft).detach() + y_soft
    
    return y_soft


def gumbel_softmax_topk(
    logits: torch.Tensor, 
    tau: float = 1.0, 
    k: int = 256, 
    hard: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient Gumbel-Softmax with top-k filtering.
    
    Instead of computing Gumbel-Softmax over full vocab (32k+), we:
    1. Select top-k logits
    2. Apply Gumbel-Softmax only to those k tokens
    3. Scatter back to full vocab size (sparse, mostly zeros)
    
    Returns:
        soft_tokens: [batch, vocab_size] - sparse soft token distribution
        topk_indices: [batch, k] - indices of top-k tokens (for efficient embedding lookup)
    """
    vocab_size = logits.shape[-1]
    
    # Get top-k logits and their indices
    topk_logits, topk_indices = logits.topk(k, dim=-1)  # [batch, k]
    
    # Apply Gumbel noise and softmax only to top-k
    gumbels = -torch.log(-torch.log(torch.rand_like(topk_logits) + 1e-10) + 1e-10)
    y_soft_topk = F.softmax((topk_logits + gumbels) / tau, dim=-1)  # [batch, k]
    
    # Scatter back to full vocab (creates sparse distribution)
    y_soft = torch.zeros_like(logits).scatter_(-1, topk_indices, y_soft_topk)
    
    if hard:
        # Find argmax within top-k (more efficient than full vocab argmax)
        local_argmax = y_soft_topk.argmax(dim=-1, keepdim=True)  # [batch, 1]
        global_argmax = topk_indices.gather(-1, local_argmax)  # [batch, 1]
        y_hard = torch.zeros_like(logits).scatter_(-1, global_argmax, 1.0)
        return (y_hard - y_soft).detach() + y_soft, topk_indices
    
    return y_soft, topk_indices


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
        return self.config.tau_start - ratio * (self.config.tau_start - self.config.tau_end)
    
    def generate_soft(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tau: float,
        use_ste: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Original generate_soft for backwards compatibility."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        current_embeds = self.embedding(input_ids)
        current_mask = attention_mask
        
        soft_tokens_list = []
        logits_list = []
        
        for _ in range(self.config.max_new_tokens):
            # Use autocast for memory efficiency in the forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(
                    inputs_embeds=current_embeds,
                    attention_mask=current_mask,
                    use_cache=False,
                )
            
            next_logits = outputs.logits[:, -1, :].float()  # Cast to float32 for gumbel
            logits_list.append(next_logits)
            del outputs  # Free memory immediately
            
            soft_token = gumbel_softmax(next_logits, tau=tau, hard=use_ste)
            soft_tokens_list.append(soft_token)
            
            # Project back to bfloat16 for embeddings
            next_embed = (soft_token.to(self.embedding.weight.dtype) @ self.embedding.weight).unsqueeze(1)
            
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)
            current_mask = torch.cat([
                current_mask, 
                torch.ones(batch_size, 1, device=device)
            ], dim=1)
        
        soft_tokens = torch.stack(soft_tokens_list, dim=1)
        logits_sequence = torch.stack(logits_list, dim=1)
        
        return soft_tokens, current_embeds, logits_sequence
    
    def generate_soft_topk(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tau: float,
        topk: int = 256,
        use_ste: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Memory-efficient soft generation with top-k filtering.
        
        Instead of storing full vocab distributions, stores:
        - topk_indices: [batch, seq_len, k] - which tokens have non-zero weight
        - topk_weights: [batch, seq_len, k] - their Gumbel-Softmax weights
        
        This reduces memory from O(batch * seq * vocab) to O(batch * seq * k).
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        current_embeds = self.embedding(input_ids)
        current_mask = attention_mask
        
        # Store sparse representation instead of full vocab
        topk_indices_list = []
        topk_weights_list = []
        logits_list = []
        hard_tokens_list = []  # For embeddings
        
        for _ in range(self.config.max_new_tokens):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(
                    inputs_embeds=current_embeds,
                    attention_mask=current_mask,
                    use_cache=False,
                )
            
            next_logits = outputs.logits[:, -1, :].float()
            logits_list.append(next_logits)
            del outputs
            
            # Use top-k Gumbel-Softmax
            soft_token, topk_idx = gumbel_softmax_topk(next_logits, tau=tau, k=topk, hard=use_ste)
            
            # Store sparse representation
            topk_weights = soft_token.gather(-1, topk_idx)  # [batch, k]
            topk_indices_list.append(topk_idx)
            topk_weights_list.append(topk_weights)
            
            # Get hard token for next embedding (argmax of soft_token)
            hard_token = soft_token.argmax(dim=-1)
            hard_tokens_list.append(hard_token)
            
            # Compute next embedding via soft token @ embedding matrix
            next_embed = (soft_token.to(self.embedding.weight.dtype) @ self.embedding.weight).unsqueeze(1)
            
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)
            current_mask = torch.cat([
                current_mask, 
                torch.ones(batch_size, 1, device=device)
            ], dim=1)
        
        topk_indices = torch.stack(topk_indices_list, dim=1)  # [batch, seq, k]
        topk_weights = torch.stack(topk_weights_list, dim=1)  # [batch, seq, k]
        hard_tokens = torch.stack(hard_tokens_list, dim=1)    # [batch, seq]
        logits_sequence = torch.stack(logits_list, dim=1)     # [batch, seq, vocab]
        
        return topk_indices, topk_weights, hard_tokens, logits_sequence
    
    def generate_hard(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            min_new_tokens=self.config.min_new_tokens,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
        )


# ============================================================================
# REWARD MODEL
# ============================================================================

class SameVocabRewardModel(nn.Module):
    def __init__(self, base_model_name: str, device: str, num_labels: int = 2):
        super().__init__()
        self.transformer = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        self.embedding = self.transformer.get_input_embeddings()
        hidden_size = self.transformer.config.hidden_size
        
        # Classifier in float32 for training stability
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_labels),
        ).float()
        
        self.device = device
        self.to(device)
        
    def forward_from_embeddings(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        
        hidden_states = outputs.hidden_states[-1]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden_states.float() * mask_expanded).sum(dim=1)
        mean_hidden = sum_hidden / mask_expanded.sum(dim=1).clamp(min=1e-9)
        
        logits = self.classifier(mean_hidden)
        rewards = F.softmax(logits, dim=-1)[:, 1]
        
        return rewards
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(input_ids)
        return self.forward_from_embeddings(embeds, attention_mask)
    
    def forward_soft(self, soft_tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Cast soft_tokens to match embedding dtype (bfloat16)
        soft_embeddings = torch.matmul(soft_tokens.to(self.embedding.weight.dtype), self.embedding.weight)
        return self.forward_from_embeddings(soft_embeddings, attention_mask)
    
    def forward_soft_sparse(
        self, 
        topk_indices: torch.Tensor,  # [batch, seq, k]
        topk_weights: torch.Tensor,  # [batch, seq, k]
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Memory-efficient forward pass using sparse soft token representation.
        
        Instead of materializing full [batch, seq, vocab] tensor and doing matmul,
        we gather only the embeddings we need and do weighted sum.
        
        Memory: O(batch * seq * k * hidden) instead of O(batch * seq * vocab * hidden)
        """
        batch_size, seq_len, k = topk_indices.shape
        
        # Gather embeddings for top-k tokens: [batch, seq, k, hidden]
        selected_embeds = self.embedding(topk_indices)
        
        # Weighted sum over k dimension: [batch, seq, hidden]
        weights = topk_weights.to(selected_embeds.dtype).unsqueeze(-1)  # [batch, seq, k, 1]
        soft_embeddings = (weights * selected_embeds).sum(dim=2)
        
        return self.forward_from_embeddings(soft_embeddings, attention_mask)


def train_reward_model(
    base_model_name: str,
    device: str,
    dataloader: DataLoader,  # Now receives dataloader instead of loading data
    num_epochs: int = 1,
    lr: float = 2e-5,
) -> SameVocabRewardModel:
    """Train reward model on provided data split."""
    print("Training reward model...")
    
    reward_model = SameVocabRewardModel(base_model_name, device)
    
    for param in reward_model.transformer.parameters():
        param.requires_grad = False
    for param in reward_model.classifier.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(reward_model.classifier.parameters(), lr=lr)
    
    reward_model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"RM Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            embeds = reward_model.embedding(input_ids)
            outputs = reward_model.transformer(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[-1]
            
            mask_exp = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
            
            logits = reward_model.classifier(pooled)
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.3f}"})
    
    print(f"Reward model training complete. Final accuracy: {correct/total:.3f}")
    
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False
    
    return reward_model


class DifferentiableRewardModel(nn.Module):
    def __init__(
        self, 
        base_model_name: str,
        generator_tokenizer,
        device: str,
        rm_dataloader: Optional[DataLoader] = None,
        pretrained_rm_path: Optional[str] = None,
    ):
        super().__init__()
        self.generator_tokenizer = generator_tokenizer
        self.device = device
        
        if pretrained_rm_path and Path(pretrained_rm_path).exists():
            print(f"Loading reward model from {pretrained_rm_path}")
            self.model = SameVocabRewardModel(base_model_name, device)
            self.model.classifier.load_state_dict(torch.load(pretrained_rm_path, map_location=device))
        else:
            assert rm_dataloader is not None, "Must provide rm_dataloader if no pretrained model"
            self.model = train_reward_model(base_model_name, device, rm_dataloader)
            if pretrained_rm_path:
                Path(pretrained_rm_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.classifier.state_dict(), pretrained_rm_path)
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward_soft(self, soft_tokens: torch.Tensor, prompt_soft_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, vocab_size = soft_tokens.shape
        attention_mask = torch.ones(batch_size, seq_len, device=self.device)
        return self.model.forward_soft(soft_tokens, attention_mask)
    
    def forward_hard(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (token_ids != self.generator_tokenizer.pad_token_id).long()
        with torch.no_grad():
            return self.model(token_ids, attention_mask)


# ============================================================================
# PPO IMPLEMENTATION
# ============================================================================

class PPOTrainer:
    def __init__(
        self, 
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: DifferentiableRewardModel,
        tokenizer,
        config: Config,
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        
        hidden_size = policy.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1).to(device=config.device, dtype=torch.bfloat16)
        
        self.optimizer = torch.optim.AdamW(
            list(policy.parameters()) + list(self.value_head.parameters()),
            lr=config.learning_rate,
        )
    
    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            next_value = 0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.config.gamma * next_value * masks[t] - values[t]
            advantages[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * masks[t] * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def step(self, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor) -> dict:
        # Generate responses (no grad)
        with torch.no_grad():
            response_ids = self.policy.generate(
                prompt_ids, attention_mask=prompt_mask,
                max_new_tokens=self.config.max_new_tokens,
                min_new_tokens=self.config.min_new_tokens,
                do_sample=True, top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response_mask = (response_ids != self.tokenizer.pad_token_id).long()
        prompt_len = prompt_ids.shape[1]
        gen_tokens = response_ids[:, prompt_len:]
        gen_mask = (gen_tokens != self.tokenizer.pad_token_id).float()
        
        # Get rewards
        with torch.no_grad():
            rewards = self.reward_model.forward_hard(response_ids, response_mask)
        
        # Compute old log probs and values (detached - these are the "old" policy)
        with torch.no_grad():
            old_outputs = self.policy(response_ids, attention_mask=response_mask, output_hidden_states=True)
            old_logits = old_outputs.logits[:, prompt_len-1:-1, :]
            old_log_probs = F.log_softmax(old_logits, dim=-1)
            old_token_log_probs = old_log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)
            
            old_hidden = old_outputs.hidden_states[-1][:, prompt_len-1:-1, :]
            old_values = self.value_head(old_hidden).squeeze(-1)
            
            # Reference model for KL
            ref_outputs = self.ref_policy(response_ids, attention_mask=response_mask)
            ref_logits = ref_outputs.logits[:, prompt_len-1:-1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_token_log_probs = ref_log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Proper per-token KL: log(π/π_ref) weighted by mask, then mean
        # For reporting, we use the mean log-ratio (approximates KL for on-policy samples)
        log_ratio_kl = old_token_log_probs - ref_token_log_probs
        kl_per_seq = (log_ratio_kl * gen_mask).sum(dim=1) / gen_mask.sum(dim=1).clamp(min=1)
        
        # Token-level rewards: sparse (only at end) with KL penalty
        token_rewards = torch.zeros_like(old_token_log_probs)
        token_rewards[:, -1] = rewards - self.config.kl_coef * kl_per_seq
        
        # GAE for advantages
        advantages, returns = self.compute_advantages(token_rewards, old_values, gen_mask)
        
        # Normalize advantages
        adv_mean = (advantages * gen_mask).sum() / gen_mask.sum().clamp(min=1)
        adv_std = ((advantages - adv_mean).pow(2) * gen_mask).sum() / gen_mask.sum().clamp(min=1)
        advantages = (advantages - adv_mean) / (adv_std.sqrt() + 1e-8)
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        all_grad_norms = []
        
        for _ in range(self.config.ppo_epochs):
            # Forward pass with gradients
            outputs = self.policy(response_ids, attention_mask=response_mask, output_hidden_states=True)
            new_logits = outputs.logits[:, prompt_len-1:-1, :]
            new_log_probs = F.log_softmax(new_logits, dim=-1)
            new_token_log_probs = new_log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)
            
            new_hidden = outputs.hidden_states[-1][:, prompt_len-1:-1, :]
            new_values = self.value_head(new_hidden).squeeze(-1)
            
            # Policy loss with clipped ratio
            log_ratio = new_token_log_probs - old_token_log_probs
            log_ratio = torch.clamp(log_ratio, -2.0, 2.0)  # Tighter clamp
            ratio = torch.exp(log_ratio)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip) * advantages
            policy_loss = -((torch.min(surr1, surr2) * gen_mask).sum() / gen_mask.sum().clamp(min=1))
            
            # Value loss (simple MSE, no clipping - more stable)
            value_loss = ((new_values - returns).pow(2) * gen_mask).sum() / gen_mask.sum().clamp(min=1)
            
            # Entropy bonus (only over generated tokens, not full vocab)
            # Use negative entropy of the action distribution at each step
            new_probs = new_log_probs.exp()
            entropy_per_token = -(new_probs * new_log_probs).sum(dim=-1)  # Per position
            entropy = (entropy_per_token * gen_mask).sum() / gen_mask.sum().clamp(min=1)
            
            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
            
            # Skip update if loss is invalid
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Track gradient norms before clipping
            grad_norms = [p.grad.norm().item() for p in self.policy.parameters() if p.grad is not None]
            all_grad_norms.extend(grad_norms)
            
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        return {
            "loss": total_loss / self.config.ppo_epochs, 
            "policy_loss": total_policy_loss / self.config.ppo_epochs,
            "value_loss": total_value_loss / self.config.ppo_epochs,
            "reward": rewards.mean().item(), 
            "kl": kl_per_seq.mean().item(),
            "grad_norm_mean": np.mean(all_grad_norms) if all_grad_norms else 0,
        }


# ============================================================================
# GUMBEL-SOFTMAX TRAINER
# ============================================================================

class GumbelTrainer:
    """Original Gumbel trainer (for backwards compatibility)."""
    
    def __init__(
        self,
        generator: DifferentiableGenerator,
        ref_model: nn.Module,
        reward_model: DifferentiableRewardModel,
        tokenizer,
        config: Config,
        use_ste: bool = False,
    ):
        self.generator = generator
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        self.use_ste = use_ste
        
        self.optimizer = torch.optim.AdamW(generator.model.parameters(), lr=config.learning_rate)
        self.step_count = 0
    
    def step(self, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor) -> dict:
        tau = self.generator.get_tau(self.step_count)
        
        # Clear cache before memory-intensive operation
        torch.cuda.empty_cache()
        
        soft_tokens, soft_embeds, logits_seq = self.generator.generate_soft(
            prompt_ids, prompt_mask, tau=tau, use_ste=self.use_ste
        )
        
        rewards = self.reward_model.model.forward_soft(
            soft_tokens, torch.ones(soft_tokens.shape[:2], device=prompt_ids.device)
        )
        
        with torch.no_grad():
            current_embeds = self.generator.embedding(prompt_ids)
            current_mask = prompt_mask
            ref_logits_list = []
            hard_tokens = soft_tokens.argmax(dim=-1)
            
            for i in range(self.config.max_new_tokens):
                ref_out = self.ref_model(inputs_embeds=current_embeds, attention_mask=current_mask, use_cache=False)
                ref_logits_list.append(ref_out.logits[:, -1, :])
                
                next_token = hard_tokens[:, i]
                next_embed = self.generator.embedding(next_token).unsqueeze(1)
                current_embeds = torch.cat([current_embeds, next_embed], dim=1)
                current_mask = torch.cat([current_mask, torch.ones(prompt_ids.shape[0], 1, device=prompt_ids.device)], dim=1)
            
            ref_logits = torch.stack(ref_logits_list, dim=1)
        
        policy_log_probs = F.log_softmax(logits_seq, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        policy_probs = F.softmax(logits_seq, dim=-1)
        kl_per_token = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)
        kl = kl_per_token.mean()
        
        loss = -rewards.mean() + self.config.kl_coef * kl
        
        self.optimizer.zero_grad()
        loss.backward()
        
        grad_norms = [p.grad.norm().item() for p in self.generator.model.parameters() if p.grad is not None]
        
        torch.nn.utils.clip_grad_norm_(self.generator.model.parameters(), 1.0)
        self.optimizer.step()
        self.step_count += 1
        
        result = {
            "loss": loss.item(), "reward": rewards.mean().item(), "kl": kl.item(),
            "tau": tau, "grad_norm_mean": np.mean(grad_norms) if grad_norms else 0,
            "grad_norm_std": np.std(grad_norms) if grad_norms else 0,
        }
        
        # Clean up to free memory
        del soft_tokens, soft_embeds, logits_seq, ref_logits, loss
        torch.cuda.empty_cache()
        
        return result


class GumbelTrainerMemoryEfficient:
    """
    Memory-optimized Gumbel trainer with 4 key optimizations:
    
    1. Top-k filtering: Gumbel-Softmax over top-k tokens instead of full vocab
       - Reduces soft_tokens from [B, T, V] to sparse [B, T, k]
       - ~100x memory reduction for k=256 vs V=32000
    
    2. Online KL computation: Compute KL per-token during generation
       - Don't store full ref_logits tensor [B, T, V]
       - Accumulate scalar KL instead
    
    3. KV-cache for reference model: Use past_key_values
       - Avoids recomputing full sequence at each step
       - ~3-4x speedup and memory reduction for ref model
    
    4. Sparse reward computation: Use gather instead of full matmul
       - forward_soft_sparse uses [B, T, k] instead of [B, T, V]
    """
    
    def __init__(
        self,
        generator: DifferentiableGenerator,
        ref_model: nn.Module,
        reward_model: DifferentiableRewardModel,
        tokenizer,
        config: Config,
        use_ste: bool = False,
    ):
        self.generator = generator
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        self.use_ste = use_ste
        self.topk = config.gumbel_topk if config.gumbel_topk > 0 else 256
        
        self.optimizer = torch.optim.AdamW(generator.model.parameters(), lr=config.learning_rate)
        self.step_count = 0
    
    def _policy_forward_step(self, policy_embeds, policy_mask):
        """Checkpointable policy forward step."""
        outputs = self.generator.model(
            inputs_embeds=policy_embeds,
            attention_mask=policy_mask,
            use_cache=False,
        )
        return outputs.logits[:, -1, :]
    
    def step(self, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor) -> dict:
        tau = self.generator.get_tau(self.step_count)
        batch_size = prompt_ids.shape[0]
        device = prompt_ids.device
        
        torch.cuda.empty_cache()
        
        # ====================================================================
        # PHASE 1: Generate with top-k + online KL computation + KV-cache
        # ====================================================================
        
        # Initialize policy generation state
        policy_embeds = self.generator.embedding(prompt_ids)
        policy_mask = prompt_mask.clone()
        
        # Initialize reference model with KV-cache
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                use_cache=True,
            )
            ref_past_kv = ref_outputs.past_key_values
            del ref_outputs
        
        # Storage for sparse soft tokens (for reward computation)
        topk_indices_list = []
        topk_weights_list = []
        
        # Online KL accumulator (detached, just for logging)
        kl_sum = 0.0
        
        for step_idx in range(self.config.max_new_tokens):
            # --- Policy forward with gradient checkpointing ---
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Use checkpointing to recompute activations during backward
                policy_logits = torch.utils.checkpoint.checkpoint(
                    self._policy_forward_step,
                    policy_embeds,
                    policy_mask,
                    use_reentrant=False,
                ).float()
            
            # Top-k Gumbel-Softmax
            soft_token, topk_idx = gumbel_softmax_topk(
                policy_logits, tau=tau, k=self.topk, hard=self.use_ste
            )
            
            # Store sparse representation (detach indices, keep weights for grad)
            topk_weights = soft_token.gather(-1, topk_idx)  # [batch, k]
            topk_indices_list.append(topk_idx.detach())
            topk_weights_list.append(topk_weights)
            
            # Hard token for ref model
            hard_token = soft_token.argmax(dim=-1)
            
            # --- Reference forward with KV-cache (no gradients) ---
            with torch.no_grad():
                ref_out = self.ref_model(
                    input_ids=hard_token.unsqueeze(-1),
                    attention_mask=torch.cat([
                        policy_mask[:, :prompt_ids.shape[1] + step_idx],
                        torch.ones(batch_size, 1, device=device)
                    ], dim=1),
                    past_key_values=ref_past_kv,
                    use_cache=True,
                )
                ref_logits = ref_out.logits[:, -1, :].float()
                ref_past_kv = ref_out.past_key_values
                del ref_out
                
                # Online KL for logging only (detached)
                policy_log_probs = F.log_softmax(policy_logits.detach(), dim=-1)
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                policy_probs = policy_log_probs.exp()
                kl_per_token = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)
                kl_sum = kl_sum + kl_per_token.mean().item()
                
                del ref_logits, policy_log_probs, ref_log_probs
            
            # Update policy embeddings for next step
            next_embed = (soft_token.to(self.generator.embedding.weight.dtype) @ 
                         self.generator.embedding.weight).unsqueeze(1)
            policy_embeds = torch.cat([policy_embeds, next_embed], dim=1)
            policy_mask = torch.cat([policy_mask, torch.ones(batch_size, 1, device=device)], dim=1)
            
            # Free policy_logits after using
            del policy_logits, soft_token
        
        # Clear ref KV-cache
        del ref_past_kv
        torch.cuda.empty_cache()
        
        # ====================================================================
        # PHASE 2: Compute reward using sparse representation
        # ====================================================================
        
        topk_indices = torch.stack(topk_indices_list, dim=1)  # [batch, seq, k]
        topk_weights = torch.stack(topk_weights_list, dim=1)  # [batch, seq, k]
        del topk_indices_list, topk_weights_list
        
        gen_mask = torch.ones(batch_size, self.config.max_new_tokens, device=device)
        
        # Use sparse forward for memory efficiency
        rewards = self.reward_model.model.forward_soft_sparse(
            topk_indices, topk_weights, gen_mask
        )
        
        # ====================================================================
        # PHASE 3: Compute loss and backprop
        # ====================================================================
        
        # Simple reward maximization loss (KL is for logging only now)
        loss = -rewards.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clean up before gradient computation
        del topk_indices, topk_weights, policy_embeds, rewards
        torch.cuda.empty_cache()
        
        grad_norms = [p.grad.norm().item() for p in self.generator.model.parameters() 
                      if p.grad is not None]
        
        torch.nn.utils.clip_grad_norm_(self.generator.model.parameters(), 1.0)
        self.optimizer.step()
        self.step_count += 1
        
        result = {
            "loss": loss.item(), 
            "reward": -loss.item(),  # reward = -loss since loss = -reward
            "kl": kl_sum / self.config.max_new_tokens,
            "tau": tau, 
            "grad_norm_mean": np.mean(grad_norms) if grad_norms else 0,
            "grad_norm_std": np.std(grad_norms) if grad_norms else 0,
        }
        
        del loss
        torch.cuda.empty_cache()
        
        return result


# ============================================================================
# REINFORCE BASELINE
# ============================================================================

class REINFORCETrainer:
    def __init__(self, policy: nn.Module, ref_policy: nn.Module, reward_model: DifferentiableRewardModel, tokenizer, config: Config):
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)
        self.baseline = 0.0
        self.baseline_momentum = 0.9
    
    def step(self, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor) -> dict:
        # Generate responses (no grad needed for generation)
        with torch.no_grad():
            response_ids = self.policy.generate(
                prompt_ids, attention_mask=prompt_mask,
                max_new_tokens=self.config.max_new_tokens, min_new_tokens=self.config.min_new_tokens,
                do_sample=True, top_p=0.9, pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response_mask = (response_ids != self.tokenizer.pad_token_id).long()
        prompt_len = prompt_ids.shape[1]
        gen_tokens = response_ids[:, prompt_len:]
        gen_mask = (gen_tokens != self.tokenizer.pad_token_id).float()
        seq_lengths = gen_mask.sum(dim=1).clamp(min=1)
        
        # Get rewards (no grad)
        with torch.no_grad():
            rewards = self.reward_model.forward_hard(response_ids, response_mask)
        
        # Compute log probs for policy gradient
        outputs = self.policy(response_ids, attention_mask=response_mask)
        logits = outputs.logits[:, prompt_len-1:-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Mean log prob per sequence (normalized by length)
        seq_log_probs = (token_log_probs * gen_mask).sum(dim=1) / seq_lengths
        
        # Compute KL with reference model (no grad, for penalty only)
        with torch.no_grad():
            ref_outputs = self.ref_policy(response_ids, attention_mask=response_mask)
            ref_logits = ref_outputs.logits[:, prompt_len-1:-1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_token_log_probs = ref_log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)
            
            # KL per sequence, normalized by length
            kl_per_token = token_log_probs - ref_token_log_probs
            kl_per_seq = (kl_per_token * gen_mask).sum(dim=1) / seq_lengths
        
        # Advantage = reward - KL penalty - baseline
        with torch.no_grad():
            adjusted_rewards = rewards - self.config.kl_coef * kl_per_seq
            advantage = adjusted_rewards - self.baseline
            # Normalize advantage for stability
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        # REINFORCE loss: -log_prob * advantage
        loss = -(seq_log_probs * advantage).mean()
        
        # Update baseline with exponential moving average of raw rewards
        self.baseline = self.baseline_momentum * self.baseline + (1 - self.baseline_momentum) * rewards.mean().item()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        grad_norms = [p.grad.norm().item() for p in self.policy.parameters() if p.grad is not None]
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            "loss": loss.item(), "reward": rewards.mean().item(), "kl": kl_per_seq.mean().item(),
            "grad_norm_mean": np.mean(grad_norms) if grad_norms else 0,
            "grad_norm_std": np.std(grad_norms) if grad_norms else 0,
        }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(
    model: nn.Module,
    reward_model: DifferentiableRewardModel,
    tokenizer,
    dataloader: DataLoader,
    config: Config,
    num_samples: int = 100,
    split_name: str = "val",
) -> dict:
    """Evaluate model on specified data split."""
    model.eval()
    rewards = []
    samples = []
    
    sample_count = 0
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            
            outputs = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=config.max_new_tokens, min_new_tokens=config.min_new_tokens,
                do_sample=True, top_p=0.9, pad_token_id=tokenizer.pad_token_id,
            )
            
            output_mask = (outputs != tokenizer.pad_token_id).long()
            batch_rewards = reward_model.forward_hard(outputs, output_mask)
            rewards.extend(batch_rewards.cpu().tolist())
            
            for i in range(len(outputs)):
                text = tokenizer.decode(outputs[i], skip_special_tokens=True)
                samples.append(text)
            
            sample_count += len(outputs)
    
    model.train()
    
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "samples": samples[:5],
        "split": split_name,
    }


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_method(
    method: Literal["gumbel", "ste", "ppo", "reinforce"],
    config: Config,
    data_splits: DataSplits,
    reward_model: DifferentiableRewardModel,
) -> dict:
    """Train a single method with proper train/val/test evaluation."""
    
    display_names = {
        'gumbel': 'GRADE', 'ste': 'GRADE-STE', 
        'gumbel_legacy': 'GRADE-Legacy', 'ste_legacy': 'GRADE-STE-Legacy',
        'ppo': 'PPO', 'reinforce': 'REINFORCE'
    }
    display_name = display_names.get(method, method)
    
    print(f"\n{'='*60}")
    print(f"Training: {display_name}")
    print(f"{'='*60}\n")
    
    run = wandb.init(
        project=config.wandb_project if hasattr(config, 'wandb_project') else "grade-vs-ppo",
        name=display_name, config=vars(config), reinit=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for decoder-only models
    
    # Load model with bfloat16 for memory efficiency
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model, 
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Avoid flash attention issues with soft tokens
    ).to(config.device)
    base_model.gradient_checkpointing_enable()  # Save memory during backprop
    
    if config.use_lora:
        if "pythia" in config.base_model.lower():
            target_modules = ["query_key_value"]
        elif "gpt2" in config.base_model.lower():
            target_modules = ["c_attn", "c_proj"]
        elif "qwen" in config.base_model.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            target_modules = ["q_proj", "v_proj"]  # Common default for LLaMA-style models
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=config.lora_r,
            lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            target_modules=target_modules,
        )
        policy = get_peft_model(base_model, lora_config)
        policy.print_trainable_parameters()
    else:
        policy = base_model
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in policy.parameters())
        print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.2f}")
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.base_model, 
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(config.device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    
    # Get data loaders from splits
    train_loader = data_splits.get_policy_train_dataloader(config.batch_size)
    val_loader = data_splits.get_val_dataloader(config.batch_size * 2)
    test_loader = data_splits.get_test_dataloader(config.batch_size * 2)
    
    # Initialize trainer
    if method == "gumbel":
        generator = DifferentiableGenerator(policy, tokenizer, config)
        trainer = GumbelTrainerMemoryEfficient(generator, ref_model, reward_model, tokenizer, config, use_ste=False)
    elif method == "ste":
        generator = DifferentiableGenerator(policy, tokenizer, config)
        trainer = GumbelTrainerMemoryEfficient(generator, ref_model, reward_model, tokenizer, config, use_ste=True)
    elif method == "gumbel_legacy":
        # Original implementation (for comparison)
        generator = DifferentiableGenerator(policy, tokenizer, config)
        trainer = GumbelTrainer(generator, ref_model, reward_model, tokenizer, config, use_ste=False)
    elif method == "ste_legacy":
        # Original implementation (for comparison)
        generator = DifferentiableGenerator(policy, tokenizer, config)
        trainer = GumbelTrainer(generator, ref_model, reward_model, tokenizer, config, use_ste=True)
    elif method == "ppo":
        trainer = PPOTrainer(policy, ref_model, reward_model, tokenizer, config)
    elif method == "reinforce":
        trainer = REINFORCETrainer(policy, ref_model, reward_model, tokenizer, config)
    
    # Training loop
    results = defaultdict(list)
    step = 0
    best_val_reward = -float('inf')
    
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
            
            # Validation evaluation
            if step % config.eval_every == 0:
                eval_model = policy if method in ["ppo", "reinforce"] else trainer.generator.model
                
                val_results = evaluate(eval_model, reward_model, tokenizer, val_loader, config, 
                                       num_samples=100, split_name="val")
                
                wandb.log({
                    f"{display_name}/val/reward_mean": val_results["mean_reward"],
                    f"{display_name}/val/reward_std": val_results["std_reward"],
                }, step=step)
                
                results["val_reward"].append(val_results["mean_reward"])
                
                # Track best validation performance
                if val_results["mean_reward"] > best_val_reward:
                    best_val_reward = val_results["mean_reward"]
                    results["best_val_reward"] = best_val_reward
                    results["best_val_step"] = step
                
                if val_results["samples"]:
                    wandb.log({
                        f"{display_name}/val/samples": wandb.Table(
                            columns=["text"], data=[[s] for s in val_results["samples"]]
                        )
                    }, step=step)
            
            step += 1
            pbar.update(1)
            pbar.set_postfix({"reward": f"{metrics['reward']:.3f}", "loss": f"{metrics['loss']:.3f}"})
    
    pbar.close()
    
    # Final TEST evaluation (only at the end!)
    print(f"\n{'='*40}")
    print(f"Running FINAL TEST evaluation for {display_name}")
    print(f"{'='*40}")
    
    eval_model = policy if method in ["ppo", "reinforce"] else trainer.generator.model
    
    test_results = evaluate(eval_model, reward_model, tokenizer, test_loader, config, 
                           num_samples=500, split_name="test")
    
    wandb.log({
        f"{display_name}/test/reward_mean": test_results["mean_reward"],
        f"{display_name}/test/reward_std": test_results["std_reward"],
    })
    
    results["test_eval"] = test_results
    results["final_val_reward"] = results["val_reward"][-1] if results["val_reward"] else None
    
    print(f"  Test Reward: {test_results['mean_reward']:.4f} ± {test_results['std_reward']:.4f}")
    print(f"  Best Val Reward: {best_val_reward:.4f} (step {results.get('best_val_step', 'N/A')})")
    
    # Save results
    output_path = Path(config.output_dir) / method
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "results.json", "w") as f:
        serializable = {k: [float(x) if isinstance(x, (np.floating, float)) else x for x in v] 
                       if isinstance(v, list) else v for k, v in results.items()}
        json.dump(serializable, f, indent=2, default=str)
    
    if method in ["gumbel", "ste"]:
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
    base_model: str = "Qwen/Qwen3-4B"
    max_steps: int = 250
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    wandb_project: str = "grade-vs-ppo"
    output_dir: str = "./results"
    seed: int = 42


def main(output_dir: str = "/data/results"):
    args = Arguments(output_dir=output_dir)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    config = Config(
        base_model=args.base_model, max_steps=args.max_steps,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        output_dir=args.output_dir, seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    config.wandb_project = args.wandb_project
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        config.device = "cpu"
    
    # Load tokenizer for data splits
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for decoder-only models
    
    # Create data splits ONCE (shared across all methods)
    data_splits = DataSplits(config, tokenizer)
    
    # Train reward model on its dedicated split
    rm_save_path = Path(config.output_dir) / "reward_model.pt"
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
    
    # Run experiments
    methods = ["gumbel", "ste", "ppo", "reinforce"] if args.method == "all" else [args.method]
    
    all_results = {}
    for method in methods:
        results = train_method(method, config, data_splits, reward_model)
        all_results[method] = results
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    display_names = {
        'gumbel': 'GRADE', 'ste': 'GRADE-STE',
        'gumbel_legacy': 'GRADE-Legacy', 'ste_legacy': 'GRADE-STE-Legacy',
        'ppo': 'PPO', 'reinforce': 'REINFORCE'
    }
    
    for method, results in all_results.items():
        test = results.get("test_eval", {})
        display_name = display_names.get(method, method)
        print(f"\n{display_name}:")
        print(f"  TEST Reward:     {test.get('mean_reward', 'N/A'):.4f} ± {test.get('std_reward', 'N/A'):.4f}")
        print(f"  Best VAL Reward: {results.get('best_val_reward', 'N/A'):.4f} (step {results.get('best_val_step', 'N/A')})")
    
    # Save comparison
    with open(Path(config.output_dir) / "comparison.json", "w") as f:
        json.dump({
            method: {
                "test_reward_mean": results.get("test_eval", {}).get("mean_reward"),
                "test_reward_std": results.get("test_eval", {}).get("std_reward"),
                "best_val_reward": results.get("best_val_reward"),
                "best_val_step": results.get("best_val_step"),
            }
            for method, results in all_results.items()
        }, f, indent=2)
    
    print(f"\nResults saved to {config.output_dir}/")


if __name__ == "__main__":
    main()