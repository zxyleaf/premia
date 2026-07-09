#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build PROPS training parquet files for lvwerra/stack-exchange-paired style data.

What this script does:
  Stage 1: split train data -> D1, apply Randomized Response (RR) to preference labels.
  Stage 2: split train data -> D2, apply RR, use Stage-1 DPO model to rank responses,
           estimate model error rate from RR/model disagreement, fuse labels by MLE.

Output layout is compatible with the user's DPO training script:
  <output_dir>/rl/train_props_stage1_rr.parquet      or train_props_stage2_mle.parquet
  <output_dir>/evaluation/train_props_eval.parquet

The output training parquet always contains at least:
  question, response_j1, response_k
where response_j1 is the current chosen/preferred response and response_k is rejected.

Example:
  python make_props_stackexchange.py \
    --stage 1 \
    --dataset_path /path/to/stack_rl_eval/data \
    --output_dir /path/to/props_eps05_stage1_data \
    --epsilon 0.5 \
    --train_sample_size 80000 \
    --sample_seed 42

  python make_props_stackexchange.py \
    --stage 2 \
    --dataset_path /path/to/stack_rl_eval/data \
    --output_dir /path/to/props_eps05_stage2_data \
    --epsilon 0.5 \
    --train_sample_size 80000 \
    --sample_seed 42 \
    --base_model_name_or_path /path/to/base_model \
    --stage1_peft_path /path/to/props_stage1_model \
    --ref_peft_path /path/to/sft_lora \
    --load_in_4bit
"""

import argparse
import gc
import glob
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None


# -----------------------------
# Dataset utilities
# -----------------------------


def build_se_paths(base_path: str, rl_filename: Optional[str], eval_filename: Optional[str]) -> Tuple[str, str]:
    """Infer StackExchange paired local parquet globs.

    Accepted dataset_path forms:
      /.../data
      /.../data/rl
      /.../data/evaluation
    """
    base = os.path.abspath(base_path)
    rl_filename = rl_filename or "train-*.parquet"
    eval_filename = eval_filename or "train-*.parquet"

    if base.endswith("/rl"):
        rl_glob = os.path.join(base, rl_filename)
        eval_glob = os.path.join(os.path.dirname(base), "evaluation", eval_filename)
    elif base.endswith("/evaluation"):
        eval_glob = os.path.join(base, eval_filename)
        rl_glob = os.path.join(os.path.dirname(base), "rl", rl_filename)
    else:
        rl_glob = os.path.join(base, "rl", rl_filename)
        eval_glob = os.path.join(base, "evaluation", eval_filename)

    if len(glob.glob(rl_glob)) == 0:
        raise FileNotFoundError(f"No RL parquet matched: {rl_glob}")
    if len(glob.glob(eval_glob)) == 0:
        raise FileNotFoundError(f"No eval parquet matched: {eval_glob}")
    return rl_glob, eval_glob


def choose_column(column_names: Sequence[str], explicit: Optional[str], candidates: Sequence[str], role: str) -> str:
    if explicit and explicit != "auto":
        if explicit not in column_names:
            raise ValueError(f"Requested {role}_col={explicit!r}, but columns are: {list(column_names)}")
        return explicit
    for c in candidates:
        if c in column_names:
            return c
    raise ValueError(f"Could not infer {role} column. Tried {list(candidates)}; columns={list(column_names)}")


def shuffle_and_cap(ds: Dataset, n: Optional[int], seed: int) -> Dataset:
    ds = ds.shuffle(seed=seed)
    if n is not None and int(n) > 0 and int(n) < len(ds):
        ds = ds.select(range(int(n)))
    return ds


def partition_dataset(ds: Dataset, stage: int, num_stages: int) -> Dataset:
    if not (1 <= stage <= num_stages):
        raise ValueError(f"stage must be in [1, {num_stages}], got {stage}")
    n = len(ds)
    start = (stage - 1) * n // num_stages
    end = stage * n // num_stages
    return ds.select(range(start, end))


def normalize_eval_dataset(ds: Dataset, question_col: str, chosen_col: str, rejected_col: str) -> Dataset:
    """Keep eval labels clean; output question/response_j1/response_k for DPO eval."""
    def map_row(ex):
        return {
            "question": (ex.get(question_col) or "").strip(),
            "response_j1": (ex.get(chosen_col) or "").strip(),
            "response_k": (ex.get(rejected_col) or "").strip(),
        }

    return ds.map(map_row, remove_columns=list(ds.column_names), desc="[eval] normalize columns")


def rr_flip_probability(epsilon: float) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0. At epsilon=0, gamma=0.5 and PROPS cannot estimate model error reliably.")
    return 1.0 / (1.0 + math.exp(float(epsilon)))


def make_rr_dataset(
    ds: Dataset,
    question_col: str,
    chosen_col: str,
    rejected_col: str,
    epsilon: float,
    rr_seed: int,
    stage: int,
    keep_metadata: bool = True,
) -> Tuple[Dataset, np.ndarray, np.ndarray]:
    """Apply randomized response to pairwise preference labels.

    Original label z*=+1 means original chosen_col is preferred to rejected_col.
    RR label z_rr=+1 keeps chosen/rejected; z_rr=-1 swaps them.
    """
    gamma = rr_flip_probability(epsilon)
    rng = np.random.default_rng(rr_seed)
    flips = rng.random(len(ds)) < gamma
    z_rr = np.where(flips, -1, 1).astype(np.int8)

    out: Dict[str, List] = {
        "question": [],
        "response_j1": [],
        "response_k": [],
    }
    if keep_metadata:
        out.update({
            "orig_response_j1": [],
            "orig_response_k": [],
            "props_stage": [],
            "props_epsilon": [],
            "props_gamma_rr": [],
            "props_rr_flip": [],
            "props_rr_label": [],
        })

    for i, ex in enumerate(tqdm(ds, desc=f"[stage {stage}] applying RR")):
        q = (ex.get(question_col) or "").strip()
        yj = (ex.get(chosen_col) or "").strip()
        yk = (ex.get(rejected_col) or "").strip()
        if z_rr[i] == 1:
            chosen, rejected = yj, yk
        else:
            chosen, rejected = yk, yj
        out["question"].append(q)
        out["response_j1"].append(chosen)
        out["response_k"].append(rejected)
        if keep_metadata:
            out["orig_response_j1"].append(yj)
            out["orig_response_k"].append(yk)
            out["props_stage"].append(stage)
            out["props_epsilon"].append(float(epsilon))
            out["props_gamma_rr"].append(float(gamma))
            out["props_rr_flip"].append(bool(flips[i]))
            out["props_rr_label"].append(int(z_rr[i]))

    return Dataset.from_dict(out), z_rr, flips.astype(np.bool_)


# -----------------------------
# Scoring utilities for Stage 2
# -----------------------------


def get_dtype(dtype_name: str):
    if dtype_name in (None, "auto"):
        return "auto"
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    if hasattr(torch, dtype_name):
        return getattr(torch, dtype_name)
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_causal_lm_with_optional_peft(
    base_model_name_or_path: str,
    peft_path: Optional[str],
    torch_dtype: str = "auto",
    load_in_4bit: bool = False,
    device_map: str = "auto",
    trust_remote_code: bool = True,
):
    kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": get_dtype(torch_dtype),
    }
    if device_map:
        kwargs["device_map"] = device_map
    if load_in_4bit:
        if BitsAndBytesConfig is None:
            raise ImportError("BitsAndBytesConfig is not available. Install bitsandbytes/transformers with 4-bit support.")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch_dtype == "bf16" else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, **kwargs)
    if peft_path:
        model = PeftModel.from_pretrained(model, peft_path, is_trainable=False)
    model.eval()
    return model


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_tokenizer(base_model_name_or_path: str, trust_remote_code: bool = True):
    tok = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=trust_remote_code, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def encode_prompt_response(
    tokenizer,
    prompt: str,
    response: str,
    max_prompt_length: int,
    max_length: int,
    add_eos_token: bool,
) -> Tuple[List[int], List[int], List[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids

    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]

    max_resp_len = max(1, max_length - len(prompt_ids))
    if add_eos_token and tokenizer.eos_token_id is not None:
        # Reserve one token for EOS if possible.
        max_resp_body_len = max(1, max_resp_len - 1)
        response_ids = response_ids[:max_resp_body_len] + [tokenizer.eos_token_id]
    else:
        response_ids = response_ids[:max_resp_len]

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, labels


def batch_sum_logprob(
    model,
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    batch_size: int,
    max_prompt_length: int,
    max_length: int,
    add_eos_token: bool = True,
) -> np.ndarray:
    """Return sum log p(response | prompt) for each pair.

    We sum token log-probabilities over response tokens only, matching the sequence-level
    log-probability convention used by DPO-style objectives.
    """
    all_scores: List[float] = []
    device = get_model_device(model)
    pad_id = tokenizer.pad_token_id

    for start in tqdm(range(0, len(prompts), batch_size), desc="scoring logp"):
        end = min(start + batch_size, len(prompts))
        encoded = [
            encode_prompt_response(
                tokenizer,
                prompts[i],
                responses[i],
                max_prompt_length=max_prompt_length,
                max_length=max_length,
                add_eos_token=add_eos_token,
            )
            for i in range(start, end)
        ]
        max_len = max(len(x[0]) for x in encoded)
        input_ids, attention_mask, labels = [], [], []
        for ids, mask, lab in encoded:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(lab + [-100] * pad_len)

        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long, device=device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(input_ids=input_ids_t, attention_mask=attention_mask_t)
            logits = out.logits
            shift_logits = logits[:, :-1, :].float()
            shift_labels = labels_t[:, 1:]
            mask_t = shift_labels.ne(-100)
            safe_labels = shift_labels.masked_fill(~mask_t, 0)
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_logp = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
            seq_logp = (token_logp * mask_t.float()).sum(dim=-1)
        all_scores.extend(seq_logp.detach().cpu().tolist())

        del input_ids_t, attention_mask_t, labels_t, logits, shift_logits, shift_labels, mask_t, safe_labels, log_probs, token_logp, seq_logp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.asarray(all_scores, dtype=np.float64)


def compute_model_labels_for_props(
    ds_raw: Dataset,
    question_col: str,
    chosen_col: str,
    rejected_col: str,
    base_model_name_or_path: str,
    stage1_peft_path: str,
    ref_peft_path: str,
    batch_size: int,
    max_prompt_length: int,
    max_length: int,
    torch_dtype: str,
    load_in_4bit: bool,
    device_map: str,
    trust_remote_code: bool,
    add_eos_token: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Use the Stage-1 DPO model as a preference labeler.

    The ranking score is the DPO implicit reward difference:
      [log pi_M1(y_j|x) - log pi_ref(y_j|x)] - [log pi_M1(y_k|x) - log pi_ref(y_k|x)].
    z_m=+1 means original response_j1/response_j is preferred; z_m=-1 means response_k is preferred.
    """
    prompts = [(ex.get(question_col) or "").strip() for ex in ds_raw]
    yj = [(ex.get(chosen_col) or "").strip() for ex in ds_raw]
    yk = [(ex.get(rejected_col) or "").strip() for ex in ds_raw]

    tokenizer = prepare_tokenizer(base_model_name_or_path, trust_remote_code=trust_remote_code)

    print(f"[SCORE] Loading Stage-1 labeler adapter: {stage1_peft_path}", flush=True)
    m1 = load_causal_lm_with_optional_peft(
        base_model_name_or_path=base_model_name_or_path,
        peft_path=stage1_peft_path,
        torch_dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    logp_m1_j = batch_sum_logprob(m1, tokenizer, prompts, yj, batch_size, max_prompt_length, max_length, add_eos_token)
    logp_m1_k = batch_sum_logprob(m1, tokenizer, prompts, yk, batch_size, max_prompt_length, max_length, add_eos_token)
    del m1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[SCORE] Loading reference/SFT adapter: {ref_peft_path}", flush=True)
    ref = load_causal_lm_with_optional_peft(
        base_model_name_or_path=base_model_name_or_path,
        peft_path=ref_peft_path,
        torch_dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    logp_ref_j = batch_sum_logprob(ref, tokenizer, prompts, yj, batch_size, max_prompt_length, max_length, add_eos_token)
    logp_ref_k = batch_sum_logprob(ref, tokenizer, prompts, yk, batch_size, max_prompt_length, max_length, add_eos_token)
    del ref
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    score = (logp_m1_j - logp_ref_j) - (logp_m1_k - logp_ref_k)
    z_m = np.where(score >= 0.0, 1, -1).astype(np.int8)
    return z_m, score


def estimate_model_error_from_disagreement(z_rr: np.ndarray, z_m: np.ndarray, gamma_rr: float) -> Tuple[float, float]:
    """Estimate eta=P(model label != true label) from disagreement with RR.

    Let RR flip rate be gamma and model error be eta. Assuming independent errors:
      P(z_rr != z_m) = gamma * (1-eta) + (1-gamma) * eta
                     = gamma + eta - 2 gamma eta.
    Therefore eta = (d - gamma) / (1 - 2 gamma).
    """
    d = float(np.mean(z_rr != z_m))
    denom = 1.0 - 2.0 * gamma_rr
    if abs(denom) < 1e-8:
        raise ValueError("Cannot estimate model error when gamma_rr is too close to 0.5. Use epsilon > 0.")
    eta = (d - gamma_rr) / denom
    eta = float(np.clip(eta, 1e-6, 1.0 - 1e-6))
    return eta, d


def mle_fuse_labels(z_rr: np.ndarray, z_m: np.ndarray, gamma_rr: float, eta_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """MLE fusion of two noisy labels in {-1,+1}.

    LLR = z_rr * log((1-gamma)/gamma) + z_m * log((1-eta)/eta).
    Choose +1 if LLR >= 0, otherwise -1.
    """
    w_rr = math.log((1.0 - gamma_rr) / gamma_rr)
    w_m = math.log((1.0 - eta_m) / eta_m)
    llr = z_rr.astype(np.float64) * w_rr + z_m.astype(np.float64) * w_m
    z_hat = np.where(llr >= 0.0, 1, -1).astype(np.int8)
    return z_hat, llr


def make_stage2_mle_dataset(
    ds_raw: Dataset,
    question_col: str,
    chosen_col: str,
    rejected_col: str,
    z_rr: np.ndarray,
    z_m: np.ndarray,
    z_m_score: np.ndarray,
    z_mle: np.ndarray,
    llr: np.ndarray,
    epsilon: float,
    gamma_rr: float,
    eta_m: float,
    disagree_rate: float,
) -> Dataset:
    out: Dict[str, List] = {
        "question": [],
        "response_j1": [],
        "response_k": [],
        "orig_response_j1": [],
        "orig_response_k": [],
        "props_stage": [],
        "props_epsilon": [],
        "props_gamma_rr": [],
        "props_eta_model": [],
        "props_rr_model_disagree_rate": [],
        "props_rr_label": [],
        "props_model_label": [],
        "props_mle_label": [],
        "props_model_score": [],
        "props_mle_llr": [],
    }
    for i, ex in enumerate(tqdm(ds_raw, desc="[stage 2] writing MLE labels")):
        q = (ex.get(question_col) or "").strip()
        yj = (ex.get(chosen_col) or "").strip()
        yk = (ex.get(rejected_col) or "").strip()
        if z_mle[i] == 1:
            chosen, rejected = yj, yk
        else:
            chosen, rejected = yk, yj
        out["question"].append(q)
        out["response_j1"].append(chosen)
        out["response_k"].append(rejected)
        out["orig_response_j1"].append(yj)
        out["orig_response_k"].append(yk)
        out["props_stage"].append(2)
        out["props_epsilon"].append(float(epsilon))
        out["props_gamma_rr"].append(float(gamma_rr))
        out["props_eta_model"].append(float(eta_m))
        out["props_rr_model_disagree_rate"].append(float(disagree_rate))
        out["props_rr_label"].append(int(z_rr[i]))
        out["props_model_label"].append(int(z_m[i]))
        out["props_mle_label"].append(int(z_mle[i]))
        out["props_model_score"].append(float(z_m_score[i]))
        out["props_mle_llr"].append(float(llr[i]))
    return Dataset.from_dict(out)


def save_dataset_layout(train_ds: Dataset, eval_ds: Dataset, output_dir: str, train_filename: str, metadata: Dict):
    rl_dir = os.path.join(output_dir, "rl")
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(rl_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    train_path = os.path.join(rl_dir, train_filename)
    eval_path = os.path.join(eval_dir, "train_props_eval.parquet")
    train_ds.to_parquet(train_path)
    eval_ds.to_parquet(eval_path)

    metadata_path = os.path.join(output_dir, "props_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[SAVE] train -> {train_path}")
    print(f"[SAVE] eval  -> {eval_path}")
    print(f"[SAVE] meta  -> {metadata_path}")


# -----------------------------
# Main
# -----------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Build PROPS private preference-label datasets for StackExchange paired data.")

    # Dataset/data split
    p.add_argument("--stage", type=int, choices=[1, 2], required=True, help="PROPS stage to build: 1=RR(D1), 2=MLE(D2).")
    p.add_argument("--num_stages", type=int, default=2, help="Dataset is split into this many disjoint parts. Default: 2.")
    p.add_argument("--dataset_path", type=str, required=True, help="Path to stack_rl_eval/data or its rl/evaluation subdir.")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory with rl/ and evaluation/ subfolders.")
    p.add_argument("--rl_filename", type=str, default="train-*.parquet", help="RL parquet filename/glob under data/rl.")
    p.add_argument("--eval_filename", type=str, default="train-*.parquet", help="Eval parquet filename/glob under data/evaluation.")
    p.add_argument("--train_sample_size", type=int, default=None, help="Shuffle/cap train before splitting, e.g. 80000. Omit or <=0 for full.")
    p.add_argument("--eval_sample_size", type=int, default=None, help="Optional eval cap.")
    p.add_argument("--sample_seed", type=int, default=42, help="Seed for train shuffle/split.")
    p.add_argument("--rr_seed", type=int, default=1234, help="Seed for randomized response flips.")

    # Columns
    p.add_argument("--question_col", type=str, default="question")
    p.add_argument("--chosen_col", type=str, default="auto", help="Original preferred response column. auto tries response_j1,response_j,response_J,chosen.")
    p.add_argument("--rejected_col", type=str, default="auto", help="Original rejected response column. auto tries response_k,response_k1,response_K,rejected.")

    # Privacy
    p.add_argument("--epsilon", type=float, required=True, help="Preference-level DP epsilon for RR: gamma=1/(1+exp(epsilon)).")

    # Stage 2 model scoring
    p.add_argument("--base_model_name_or_path", type=str, default=None, help="Base model path/name used for scoring stage 2.")
    p.add_argument("--stage1_peft_path", type=str, default=None, help="Stage-1 DPO LoRA adapter path used as labeler for stage 2.")
    p.add_argument("--ref_peft_path", type=str, default=None, help="SFT LoRA adapter path used as reference for DPO implicit reward scoring.")
    p.add_argument("--score_batch_size", type=int, default=2)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--torch_dtype", type=str, default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--no_add_eos_token", action="store_true", help="Do not append EOS to scored responses.")

    return p.parse_args()


def main():
    args = parse_args()
    gamma = rr_flip_probability(args.epsilon)
    print(f"[PROPS] stage={args.stage} epsilon={args.epsilon} gamma_rr={gamma:.6f}")

    rl_glob, eval_glob = build_se_paths(args.dataset_path, args.rl_filename, args.eval_filename)
    print(f"[LOAD] rl_glob={rl_glob}")
    print(f"[LOAD] eval_glob={eval_glob}")

    ds_train = load_dataset("parquet", data_files={"train": rl_glob})["train"]
    ds_eval = load_dataset("parquet", data_files={"train": eval_glob})["train"]

    if args.question_col not in ds_train.column_names:
        raise ValueError(f"question_col={args.question_col!r} not in train columns: {ds_train.column_names}")
    chosen_col = choose_column(
        ds_train.column_names,
        args.chosen_col,
        candidates=["response_j1", "response_j", "response_J", "chosen"],
        role="chosen",
    )
    rejected_col = choose_column(
        ds_train.column_names,
        args.rejected_col,
        candidates=["response_k", "response_k1", "response_K", "rejected"],
        role="rejected",
    )
    print(f"[COLS] question={args.question_col} chosen={chosen_col} rejected={rejected_col}")

    # Normalize/cap eval before saving. Evaluation remains clean, not RR-flipped.
    if args.eval_sample_size is not None and args.eval_sample_size > 0 and args.eval_sample_size < len(ds_eval):
        ds_eval = ds_eval.shuffle(seed=args.sample_seed).select(range(args.eval_sample_size))
    eval_chosen_col = chosen_col if chosen_col in ds_eval.column_names else choose_column(ds_eval.column_names, "auto", ["response_j1", "response_j", "response_J", "chosen"], "eval_chosen")
    eval_rejected_col = rejected_col if rejected_col in ds_eval.column_names else choose_column(ds_eval.column_names, "auto", ["response_k", "response_k1", "response_K", "rejected"], "eval_rejected")
    ds_eval_norm = normalize_eval_dataset(ds_eval, args.question_col, eval_chosen_col, eval_rejected_col)

    # Shuffle/cap train once, then split into disjoint stages.
    ds_train = shuffle_and_cap(ds_train, args.train_sample_size, args.sample_seed)
    ds_stage = partition_dataset(ds_train, stage=args.stage, num_stages=args.num_stages)
    print(f"[SPLIT] full_after_cap={len(ds_train)} | stage_{args.stage}_size={len(ds_stage)} / num_stages={args.num_stages}")

    if args.stage == 1:
        ds_rr, z_rr, flips = make_rr_dataset(
            ds_stage,
            question_col=args.question_col,
            chosen_col=chosen_col,
            rejected_col=rejected_col,
            epsilon=args.epsilon,
            rr_seed=args.rr_seed + 1000 * args.stage,
            stage=1,
        )
        meta = {
            "method": "PROPS-stage1-RR",
            "epsilon": args.epsilon,
            "gamma_rr": gamma,
            "stage": 1,
            "num_stages": args.num_stages,
            "train_size_after_cap": len(ds_train),
            "stage_train_size": len(ds_stage),
            "rr_flip_count": int(flips.sum()),
            "rr_flip_rate_empirical": float(flips.mean()),
            "question_col": args.question_col,
            "chosen_col": chosen_col,
            "rejected_col": rejected_col,
            "sample_seed": args.sample_seed,
            "rr_seed": args.rr_seed + 1000 * args.stage,
        }
        save_dataset_layout(ds_rr, ds_eval_norm, args.output_dir, "train_props_stage1_rr.parquet", meta)
        return

    # Stage 2
    if not args.base_model_name_or_path:
        raise ValueError("--base_model_name_or_path is required for --stage 2")
    if not args.stage1_peft_path:
        raise ValueError("--stage1_peft_path is required for --stage 2")
    if not args.ref_peft_path:
        raise ValueError("--ref_peft_path is required for --stage 2")

    # RR labels on D2. We use the raw D2 rows for model scoring, and z_rr for fusion.
    _, z_rr, flips = make_rr_dataset(
        ds_stage,
        question_col=args.question_col,
        chosen_col=chosen_col,
        rejected_col=rejected_col,
        epsilon=args.epsilon,
        rr_seed=args.rr_seed + 1000 * args.stage,
        stage=2,
        keep_metadata=False,
    )

    z_m, z_m_score = compute_model_labels_for_props(
        ds_raw=ds_stage,
        question_col=args.question_col,
        chosen_col=chosen_col,
        rejected_col=rejected_col,
        base_model_name_or_path=args.base_model_name_or_path,
        stage1_peft_path=args.stage1_peft_path,
        ref_peft_path=args.ref_peft_path,
        batch_size=args.score_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        add_eos_token=not args.no_add_eos_token,
    )

    eta_m, disagree_rate = estimate_model_error_from_disagreement(z_rr, z_m, gamma)
    z_mle, llr = mle_fuse_labels(z_rr, z_m, gamma, eta_m)

    print(f"[STAGE2] RR empirical flip rate       = {float(flips.mean()):.6f}")
    print(f"[STAGE2] RR/model disagreement rate  = {disagree_rate:.6f}")
    print(f"[STAGE2] estimated eta_model         = {eta_m:.6f}")
    print(f"[STAGE2] MLE uses model label rate   = {float(np.mean(z_mle == z_m)):.6f}")
    print(f"[STAGE2] MLE uses RR label rate      = {float(np.mean(z_mle == z_rr)):.6f}")

    ds_mle = make_stage2_mle_dataset(
        ds_raw=ds_stage,
        question_col=args.question_col,
        chosen_col=chosen_col,
        rejected_col=rejected_col,
        z_rr=z_rr,
        z_m=z_m,
        z_m_score=z_m_score,
        z_mle=z_mle,
        llr=llr,
        epsilon=args.epsilon,
        gamma_rr=gamma,
        eta_m=eta_m,
        disagree_rate=disagree_rate,
    )

    meta = {
        "method": "PROPS-stage2-MLE",
        "epsilon": args.epsilon,
        "gamma_rr": gamma,
        "eta_model_estimated": eta_m,
        "rr_model_disagree_rate": disagree_rate,
        "stage": 2,
        "num_stages": args.num_stages,
        "train_size_after_cap": len(ds_train),
        "stage_train_size": len(ds_stage),
        "rr_flip_count": int(flips.sum()),
        "rr_flip_rate_empirical": float(flips.mean()),
        "mle_equals_model_rate": float(np.mean(z_mle == z_m)),
        "mle_equals_rr_rate": float(np.mean(z_mle == z_rr)),
        "question_col": args.question_col,
        "chosen_col": chosen_col,
        "rejected_col": rejected_col,
        "sample_seed": args.sample_seed,
        "rr_seed": args.rr_seed + 1000 * args.stage,
        "base_model_name_or_path": args.base_model_name_or_path,
        "stage1_peft_path": args.stage1_peft_path,
        "ref_peft_path": args.ref_peft_path,
        "score_batch_size": args.score_batch_size,
        "max_prompt_length": args.max_prompt_length,
        "max_length": args.max_length,
        "torch_dtype": args.torch_dtype,
        "load_in_4bit": bool(args.load_in_4bit),
    }
    save_dataset_layout(ds_mle, ds_eval_norm, args.output_dir, "train_props_stage2_mle.parquet", meta)


if __name__ == "__main__":
    main()
