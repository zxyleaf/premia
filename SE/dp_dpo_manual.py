#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Manual DP-DPO for LoRA / QLoRA models.

Purpose
-------
A simple baseline for training-time privacy defense:
  SFT reference model -> DPO with per-example gradient clipping + Gaussian noise.

This script is designed for StackExchange-Paired style data with columns:
  question, response_j, response_k
and also supports transformed columns such as response_j1 / response_k1.

Important
---------
1) This is an exact but slow implementation: it computes one backward pass per sample.
2) DP is applied only to trainable parameters, normally the SFT LoRA adapter parameters.
3) It logs an approximate epsilon if Opacus is installed. The accounting assumes uniform
   minibatch sampling and should be used as a baseline reporting aid, not as a formal audit.

Example
-------
accelerate launch --num_processes 1 dp_dpo_manual.py \
  --base_model_name_or_path /data1/zxy/Premia/origin/open_llama_3b_v2 \
  --sft_peft_path /data1/zxy/Premia/HH/train_sft/open_llama_3b_v2_sft_lora \
  --dataset_name /data1/zxy/Premia/dataset/stack_exchange_paired \
  --rl_filename train-00000-of-00001.parquet \
  --output_dir /data1/zxy/Premia/HH/train_dpo_dp/open_llama_3b_dp_dpo_sigma1_c01 \
  --num_train_epochs 3 \
  --dp_batch_size 8 \
  --learning_rate 5e-5 \
  --beta 0.4 \
  --dp_clip_norm 0.1 \
  --dp_noise_multiplier 1.0 \
  --train_sample_size 80000 \
  --eval_sample_size 2000 \
  --max_length 1024 \
  --max_prompt_length 256 \
  --load_in_4bit \
  --dtype bfloat16
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
)
from peft import PeftModel, prepare_model_for_kbit_training


# -----------------------------
# Args
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser("DP-DPO baseline with per-example clipping and Gaussian noise")

    # model / adapter
    p.add_argument("--base_model_name_or_path", type=str, required=True,
                   help="Original base model path. For LoRA SFT, this is the base model.")
    p.add_argument("--sft_peft_path", type=str, default=None,
                   help="SFT LoRA adapter path. If omitted, base_model_name_or_path is trained directly.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--attn_implementation", type=str, default=None)

    # quantization
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    p.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    p.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--gradient_checkpointing", action="store_true")

    # data
    p.add_argument("--dataset_name", type=str, required=True,
                   help="Local data root, local /rl folder, or HF dataset id like lvwerra/stack-exchange-paired.")
    p.add_argument("--rl_filename", type=str, default="train-00000-of-00001.parquet")
    p.add_argument("--eval_filename", type=str, default="train-00000-of-00001.parquet")
    p.add_argument("--train_sample_size", type=int, default=None)
    p.add_argument("--eval_sample_size", type=int, default=None)
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--prompt_col", type=str, default="question")
    p.add_argument("--chosen_col", type=str, default="response_j",
                   help="Use response_j for original DP-DPO baseline; use response_j1 for perturbed-data experiments.")
    p.add_argument("--rejected_col", type=str, default="response_k",
                   help="Use response_k for original DP-DPO baseline; use response_k1 for perturbed-data experiments.")

    # DPO
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--optim", type=str, default="adamw", choices=["adamw", "paged_adamw_32bit"])
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_steps", type=int, default=-1)

    # DP
    p.add_argument("--dp_batch_size", type=int, default=8,
                   help="Logical batch size per noisy DP update. One backward pass is run per sample.")
    p.add_argument("--dp_clip_norm", type=float, default=0.1)
    p.add_argument("--dp_noise_multiplier", type=float, default=1.0,
                   help="Gaussian noise multiplier sigma. Noise std = sigma * clip_norm before division by batch size.")
    p.add_argument("--dp_delta", type=float, default=None,
                   help="If unset, defaults to 1 / train_size.")
    p.add_argument("--disable_dp_noise", action="store_true",
                   help="Debug only: clip gradients but do not add noise.")

    # tokenization / formatting
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--add_eos_token", action="store_true")
    p.add_argument("--template", type=str, default="plain", choices=["plain", "chat"])
    p.add_argument("--plain_sep", type=str, default="\n\n")

    # eval / save / logging
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--report_to", type=str, default="none", choices=["none", "wandb"])
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--max_eval_batches", type=int, default=100)

    return p.parse_args()


def str_to_dtype(s: str):
    if s == "auto":
        return "auto"
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


# -----------------------------
# Dataset loading
# -----------------------------

def _build_local_paths(base_path: str, rl_filename: str, eval_filename: str) -> Tuple[str, str]:
    base = os.path.abspath(base_path)
    if base.endswith("/rl"):
        rl_glob = os.path.join(base, rl_filename)
        eval_glob = os.path.join(os.path.dirname(base), "evaluation", eval_filename)
    elif base.endswith("/evaluation"):
        eval_glob = os.path.join(base, eval_filename)
        rl_glob = os.path.join(os.path.dirname(base), "rl", rl_filename)
    else:
        rl_glob = os.path.join(base, "rl", rl_filename)
        eval_glob = os.path.join(base, "evaluation", eval_filename)
    return rl_glob, eval_glob


def _sample_dataset(ds: Dataset, n: Optional[int], seed: int) -> Dataset:
    if n is None or n <= 0 or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def load_preference_data(args) -> DatasetDict:
    """Load StackExchange-style data from local parquet root or HF hub."""
    if os.path.exists(args.dataset_name):
        rl_path, eval_path = _build_local_paths(args.dataset_name, args.rl_filename, args.eval_filename)
        print(f"[DATA] local train={rl_path}")
        print(f"[DATA] local eval ={eval_path}")
        train = load_dataset("parquet", data_files={"train": rl_path}, split="train")
        eval_ = load_dataset("parquet", data_files={"train": eval_path}, split="train")
    else:
        # Primary attempt: HF dataset with data_dir. If your datasets version hits ExpectedMoreSplits,
        # use the URL parquet fallback below.
        try:
            print(f"[DATA] HF dataset={args.dataset_name} data_dir=data/rl")
            train = load_dataset(args.dataset_name, data_dir="data/rl", split="train")
            print(f"[DATA] HF dataset={args.dataset_name} data_dir=data/evaluation")
            eval_ = load_dataset(args.dataset_name, data_dir="data/evaluation", split="train")
        except Exception as e:
            print(f"[DATA] HF data_dir loading failed: {repr(e)}")
            print("[DATA] Falling back to direct parquet URLs on the Hub.")
            base_url = f"https://huggingface.co/datasets/{args.dataset_name}/resolve/main"
            train_url = f"{base_url}/data/rl/{args.rl_filename}"
            eval_url = f"{base_url}/data/evaluation/{args.eval_filename}"
            print(f"[DATA] train_url={train_url}")
            print(f"[DATA] eval_url ={eval_url}")
            train = load_dataset("parquet", data_files={"train": train_url}, split="train")
            eval_ = load_dataset("parquet", data_files={"train": eval_url}, split="train")

    train = _sample_dataset(train, args.train_sample_size, args.sample_seed)
    eval_ = _sample_dataset(eval_, args.eval_sample_size, args.sample_seed)

    def to_triplet(ex):
        # Allow prompt/chosen/rejected already prepared, otherwise use specified columns.
        prompt = ex.get("prompt", ex.get(args.prompt_col, ""))
        chosen = ex.get("chosen", ex.get(args.chosen_col, ""))
        rejected = ex.get("rejected", ex.get(args.rejected_col, ""))
        return {
            "prompt": str(prompt or "").strip(),
            "chosen": str(chosen or "").strip(),
            "rejected": str(rejected or "").strip(),
        }

    def check_cols(ds: Dataset, split_name: str):
        cols = set(ds.column_names)
        if {"prompt", "chosen", "rejected"}.issubset(cols):
            return
        missing = []
        for c in [args.prompt_col, args.chosen_col, args.rejected_col]:
            if c not in cols:
                missing.append(c)
        if missing:
            raise ValueError(f"[{split_name}] missing columns {missing}; available={ds.column_names}")

    check_cols(train, "train")
    check_cols(eval_, "eval")

    train = train.map(to_triplet, remove_columns=list(train.column_names), desc="to DPO triplet")
    eval_ = eval_.map(to_triplet, remove_columns=list(eval_.column_names), desc="to DPO triplet")
    print(f"[DATA] final sizes: train={len(train)} eval={len(eval_)}")
    print(f"[DATA] first train row: {train[0]}")
    return DatasetDict({"train": train, "test": eval_})


# -----------------------------
# Model loading
# -----------------------------

def get_model_kwargs(args, trainable: bool):
    kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    if args.load_in_4bit:
        compute_dtype = str_to_dtype(args.bnb_4bit_compute_dtype)
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        dtype = str_to_dtype(args.dtype)
        if dtype != "auto":
            kwargs["torch_dtype"] = dtype
    return kwargs


def load_policy_and_ref(args):
    print(f"[MODEL] Loading policy base: {args.base_model_name_or_path}")
    policy_base = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        **get_model_kwargs(args, trainable=True),
    )
    if args.gradient_checkpointing:
        policy_base.config.use_cache = False
        policy_base.gradient_checkpointing_enable()

    if args.load_in_4bit:
        policy_base = prepare_model_for_kbit_training(
            policy_base,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    if args.sft_peft_path:
        print(f"[MODEL] Loading trainable SFT LoRA adapter: {args.sft_peft_path}")
        policy = PeftModel.from_pretrained(policy_base, args.sft_peft_path, is_trainable=True)
    else:
        print("[WARN] --sft_peft_path not set. This will train all requires_grad parameters of the base model.")
        policy = policy_base

    print(f"[MODEL] Loading reference base: {args.base_model_name_or_path}")
    ref_base = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        **get_model_kwargs(args, trainable=False),
    )
    if args.sft_peft_path:
        print(f"[MODEL] Loading frozen SFT LoRA adapter for reference: {args.sft_peft_path}")
        ref = PeftModel.from_pretrained(ref_base, args.sft_peft_path, is_trainable=False)
    else:
        ref = ref_base

    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    # Tokenizer should come from the base model or SFT directory if it exists there.
    tok_path = args.sft_peft_path if args.sft_peft_path and os.path.exists(os.path.join(args.sft_peft_path, "tokenizer_config.json")) else args.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print_trainable_params(policy)
    return policy, ref, tokenizer


def print_trainable_params(model):
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"[MODEL] trainable params={trainable:,} / total params={total:,} ({100*trainable/max(1,total):.4f}%)")


# -----------------------------
# DPO logprob / loss
# -----------------------------

def model_device(model) -> torch.device:
    return next((p.device for p in model.parameters() if p.device.type != "meta"), torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def format_prompt(args, prompt: str) -> str:
    prompt = str(prompt or "").strip()
    if args.template == "chat":
        return f"<user>\n{prompt}\n</user>\n<assistant>\n"
    return prompt + args.plain_sep


def encode_prompt_answer(tokenizer, args, prompt: str, answer: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    prompt_text = format_prompt(args, prompt)
    answer = str(answer or "").strip()

    p_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    a_ids = tokenizer(answer, add_special_tokens=False).input_ids

    # Truncate prompt first, then answer to max_length.
    if len(p_ids) > args.max_prompt_length:
        p_ids = p_ids[-args.max_prompt_length:]

    ids = []
    if tokenizer.bos_token_id is not None:
        ids.append(tokenizer.bos_token_id)
    start = len(ids) + len(p_ids)
    ids.extend(p_ids)
    ids.extend(a_ids)
    if args.add_eos_token and tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)

    if len(ids) > args.max_length:
        ids = ids[:args.max_length]

    if start >= len(ids):
        return None
    labels = torch.tensor(ids, dtype=torch.long)
    labels[:start] = -100
    if int((labels != -100).sum().item()) <= 0:
        return None
    input_ids = torch.tensor(ids, dtype=torch.long)
    return input_ids.unsqueeze(0), labels.unsqueeze(0)


def sequence_logp(model, tokenizer, args, prompt: str, answer: str) -> Optional[torch.Tensor]:
    enc = encode_prompt_answer(tokenizer, args, prompt, answer)
    if enc is None:
        return None
    input_ids, labels = enc
    dev = model_device(model)
    input_ids = input_ids.to(dev)
    labels = labels.to(dev)
    attention_mask = torch.ones_like(input_ids, device=dev)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    ans_len = (labels != -100).sum().to(out.loss.dtype)
    return -out.loss * ans_len


def dpo_loss_one(model, ref_model, tokenizer, args, ex: Dict[str, str]) -> Optional[Tuple[torch.Tensor, Dict[str, float]]]:
    prompt = ex["prompt"]
    chosen = ex["chosen"]
    rejected = ex["rejected"]

    pi_ch = sequence_logp(model, tokenizer, args, prompt, chosen)
    pi_rj = sequence_logp(model, tokenizer, args, prompt, rejected)
    if pi_ch is None or pi_rj is None:
        return None

    with torch.no_grad():
        ref_ch = sequence_logp(ref_model, tokenizer, args, prompt, chosen)
        ref_rj = sequence_logp(ref_model, tokenizer, args, prompt, rejected)
    if ref_ch is None or ref_rj is None:
        return None

    pi_logratio = pi_ch - pi_rj
    ref_logratio = ref_ch - ref_rj
    margin = pi_logratio - ref_logratio
    logits = args.beta * margin
    loss = -F.logsigmoid(logits)
    metrics = {
        "loss": float(loss.detach().cpu()),
        "margin": float(margin.detach().cpu()),
        "acc": float((margin.detach() > 0).float().cpu()),
        "pi_logratio": float(pi_logratio.detach().cpu()),
        "ref_logratio": float(ref_logratio.detach().cpu()),
    }
    return loss, metrics


# -----------------------------
# DP update
# -----------------------------

def trainable_parameters(model) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def grad_global_norm(params: List[torch.nn.Parameter]) -> float:
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach().float()
            total_sq += float(torch.sum(g * g).cpu())
    return math.sqrt(total_sq)


def zero_model_grad(model):
    model.zero_grad(set_to_none=True)


def make_optimizer(args, params):
    if args.optim == "paged_adamw_32bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW32bit(
                params,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
        except Exception as e:
            print(f"[WARN] bitsandbytes PagedAdamW32bit unavailable: {repr(e)}; falling back to torch AdamW.")
    return torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )


class PrivacyLogger:
    def __init__(self, noise_multiplier: float, sample_rate: float, delta: float):
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.delta = delta
        self.steps = 0
        self.accountant = None
        try:
            from opacus.accountants import RDPAccountant
            self.accountant = RDPAccountant()
            print("[DP] Opacus RDPAccountant enabled for approximate epsilon logging.")
        except Exception as e:
            print(f"[DP] Opacus not available; epsilon will not be logged. ({repr(e)})")

    def step(self):
        self.steps += 1
        if self.accountant is not None and self.noise_multiplier > 0:
            self.accountant.step(noise_multiplier=self.noise_multiplier, sample_rate=self.sample_rate)

    def epsilon(self):
        if self.accountant is None or self.noise_multiplier <= 0:
            return None
        try:
            return float(self.accountant.get_epsilon(delta=self.delta))
        except Exception:
            return None


@torch.no_grad()
def evaluate(model, ref_model, tokenizer, args, eval_ds: Dataset) -> Dict[str, float]:
    model.eval()
    losses, margins, accs = [], [], []
    n_seen = 0
    max_items = args.max_eval_batches * args.dp_batch_size
    for ex in eval_ds:
        if n_seen >= max_items:
            break
        out = dpo_loss_one(model, ref_model, tokenizer, args, ex)
        if out is None:
            continue
        _, m = out
        losses.append(m["loss"])
        margins.append(m["margin"])
        accs.append(m["acc"])
        n_seen += 1
    model.train()
    return {
        "eval_loss": float(sum(losses) / max(1, len(losses))),
        "eval_margin": float(sum(margins) / max(1, len(margins))),
        "eval_acc": float(sum(accs) / max(1, len(accs))),
        "eval_n": float(len(losses)),
    }


def save_checkpoint(model, tokenizer, args, step: int, checkpoints: List[str]):
    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    checkpoints.append(ckpt_dir)
    print(f"[SAVE] {ckpt_dir}")
    while args.save_total_limit and len(checkpoints) > args.save_total_limit:
        old = checkpoints.pop(0)
        try:
            import shutil
            shutil.rmtree(old)
            print(f"[SAVE] removed old checkpoint: {old}")
        except Exception as e:
            print(f"[WARN] failed to remove {old}: {repr(e)}")


# -----------------------------
# Main train loop
# -----------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # Reproducibility
    random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.sample_seed)

    wandb = None
    if args.report_to == "wandb":
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(project=os.environ.get("WANDB_PROJECT", "premia_dp_dpo"), name=args.run_name, config=vars(args))
        except Exception as e:
            print(f"[WANDB] disabled because init failed: {repr(e)}")
            wandb = None

    data = load_preference_data(args)
    policy, ref_model, tokenizer = load_policy_and_ref(args)
    policy.train()

    params = trainable_parameters(policy)
    if not params:
        raise RuntimeError("No trainable parameters found. Did you load SFT LoRA with is_trainable=True?")

    optimizer = make_optimizer(args, params)
    updates_per_epoch = math.ceil(len(data["train"]) / args.dp_batch_size)
    total_updates = int(math.ceil(args.num_train_epochs * updates_per_epoch))
    if args.max_steps and args.max_steps > 0:
        total_updates = min(total_updates, args.max_steps)
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_updates,
    )

    delta = args.dp_delta if args.dp_delta is not None else 1.0 / max(1, len(data["train"]))
    privacy = PrivacyLogger(
        noise_multiplier=0.0 if args.disable_dp_noise else args.dp_noise_multiplier,
        sample_rate=args.dp_batch_size / max(1, len(data["train"])),
        delta=delta,
    )
    print(f"[DP] clip_norm={args.dp_clip_norm} noise_multiplier={args.dp_noise_multiplier} delta={delta}")
    print(f"[TRAIN] total_updates={total_updates}, updates_per_epoch={updates_per_epoch}")

    loader = DataLoader(data["train"], batch_size=args.dp_batch_size, shuffle=True, collate_fn=lambda xs: xs)
    step = 0
    epoch = 0
    checkpoints: List[str] = []
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    t0 = time.time()

    while step < total_updates:
        epoch += 1
        pbar = tqdm(loader, desc=f"DP-DPO epoch {epoch}")
        for batch in pbar:
            if step >= total_updates:
                break

            # buffers for clipped per-example gradients
            accum_grads = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in params]
            batch_metrics = []
            used = 0
            skipped = 0

            for ex in batch:
                zero_model_grad(policy)
                out = dpo_loss_one(policy, ref_model, tokenizer, args, ex)
                if out is None:
                    skipped += 1
                    continue
                loss, metrics = out
                loss.backward()

                norm = grad_global_norm(params)
                clip_coef = min(1.0, args.dp_clip_norm / (norm + 1e-12))

                for i, p in enumerate(params):
                    if p.grad is not None:
                        accum_grads[i].add_(p.grad.detach(), alpha=clip_coef)

                metrics["grad_norm"] = float(norm)
                metrics["clip_coef"] = float(clip_coef)
                batch_metrics.append(metrics)
                used += 1

            if used == 0:
                print("[WARN] entire DP batch skipped due to tokenization/truncation.")
                continue

            # assign averaged noisy gradients
            zero_model_grad(policy)
            for p, gsum in zip(params, accum_grads):
                if args.disable_dp_noise or args.dp_noise_multiplier <= 0:
                    noisy = gsum
                else:
                    noise = torch.randn_like(gsum) * (args.dp_noise_multiplier * args.dp_clip_norm)
                    noisy = gsum + noise
                p.grad = (noisy / float(used)).to(p.dtype)

            optimizer.step()
            scheduler.step()
            zero_model_grad(policy)
            privacy.step()
            step += 1

            mean = lambda key: sum(m[key] for m in batch_metrics) / max(1, len(batch_metrics))
            log = {
                "step": step,
                "epoch": epoch,
                "train_loss": mean("loss"),
                "train_margin": mean("margin"),
                "train_acc": mean("acc"),
                "grad_norm": mean("grad_norm"),
                "clip_coef": mean("clip_coef"),
                "used": used,
                "skipped": skipped,
                "lr": float(scheduler.get_last_lr()[0]),
                "epsilon_approx": privacy.epsilon(),
                "delta": delta,
                "time_sec": time.time() - t0,
            }

            if step % args.logging_steps == 0:
                pbar.set_postfix({
                    "loss": f"{log['train_loss']:.4f}",
                    "acc": f"{log['train_acc']:.3f}",
                    "eps": "NA" if log["epsilon_approx"] is None else f"{log['epsilon_approx']:.2f}",
                })
                print("[LOG] " + json.dumps(log, ensure_ascii=False))

            if args.eval_steps > 0 and step % args.eval_steps == 0:
                ev = evaluate(policy, ref_model, tokenizer, args, data["test"])
                log.update(ev)
                print("[EVAL] " + json.dumps({"step": step, **ev}, ensure_ascii=False))

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")

            if wandb is not None:
                wandb.log(log, step=step)

            if args.save_steps > 0 and step % args.save_steps == 0:
                save_checkpoint(policy, tokenizer, args, step, checkpoints)

    print("[FINAL] saving final model/adapter")
    policy.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    final_privacy = {"epsilon_approx": privacy.epsilon(), "delta": delta, "steps": step}
    with open(os.path.join(args.output_dir, "privacy_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_privacy, f, indent=2, ensure_ascii=False)
    print("[FINAL] " + json.dumps(final_privacy, ensure_ascii=False))

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
