#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Strict SNO generating response_k1 with:
  (A) token-level similarity constraint
  (B) sentence-level keep constraint
  (C) non-decreasing sim(j, k1) guarantee

Goal
-----
For each sample (response_j, response_k), generate response_k1 such that:
  (1) sim(k, k1) >= tau_keep
  (2) sim(j, k1) >= sim(j, k)
  (3) for each replaced token: token_sim(src, cand) >= tau_token

If no candidate satisfies all constraints within M trials, fallback to original k.

New output columns appended:
  - response_k1
  - sim_j_k
  - sim_k_k1
  - sim_j_k1
  - ok_nondecrease
  - ok_keep
  - fallback
  - n_changed
"""

import os
import re
import math
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import torch
from tqdm import tqdm
from datasets import load_dataset

import pyarrow as pa
import pyarrow.parquet as pq

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util


# ----------------------------
# 1) Token heuristics / filters
# ----------------------------
STOPWORDS = set("""
a an the and or but if then else for of to in on at by with from as is are was were be been being
this that these those it its i'm you're he's she's we're they're do does did done doing
can could would should may might must will shall not no yes
very more most less least much many some any each every
""".split())

PUNCT_RE = re.compile(r"^[\.\,\!\?\:\;\-\—\(\)\[\]\{\}\"\'\/\\]+$")
NUM_RE   = re.compile(r"^\d+([\,\.]\d+)*$")


def spm_clean(tok: str) -> str:
    return tok.replace("▁", "").strip()


def is_low_semantic_token(tok: str) -> bool:
    s = spm_clean(tok).strip()
    if not s:
        return False
    s_low = s.lower()
    if s_low in STOPWORDS:
        return True
    if PUNCT_RE.match(s):
        return True
    return False


def contains_sensitive_pattern(tok: str) -> bool:
    s = spm_clean(tok).strip()
    if not s:
        return False
    if NUM_RE.match(s):
        return True
    if s.startswith("@") or s.startswith("#"):
        return True
    if "http" in s.lower() or "www." in s.lower():
        return True
    if "@" in s and "." in s:
        return True
    return False


def same_coarse_type(a: str, b: str) -> bool:
    """
    Keep coarse type stable:
      - punctuation <-> punctuation
      - alphabetic-ish <-> alphabetic-ish
    """
    a0, b0 = spm_clean(a), spm_clean(b)
    a_p = bool(PUNCT_RE.match(a0))
    b_p = bool(PUNCT_RE.match(b0))
    if a_p or b_p:
        return a_p == b_p
    a_alpha = any(ch.isalpha() for ch in a0)
    b_alpha = any(ch.isalpha() for ch in b0)
    return a_alpha == b_alpha


# ----------------------------
# 2) Config
# ----------------------------
@dataclass
class SNOConfig:
    replace_ratio: float = 0.05
    sigma: float = 0.06
    topk: int = 15
    temperature: float = 0.4
    max_attempts_per_token: int = 20
    only_low_semantic: bool = False
    allow_same_token: bool = False
    max_replace_tokens: int = -1
    tau_token: float = 0.80  # NEW: token-level cosine threshold


# ----------------------------
# 3) Token-level embedding perturbation
# ----------------------------
class SNOEmbeddingPerturber:
    """
    Token-level embedding neighborhood obfuscation with token-level similarity filtering:
      - pick eligible tokens
      - add Gaussian noise in normalized embedding space
      - nearest neighbors in vocab embedding space (cosine)
      - sample replacement token id
      - require token cosine(src, cand) >= tau_token
    Returns: (perturbed_text, n_changed_tokens)
    """

    def __init__(self, model_path: str, device: Optional[str] = None, dtype=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype if dtype is not None else (
                torch.float16 if self.device.startswith("cuda") else torch.float32
            ),
            device_map=None,
        ).to(self.device).eval()

        emb = self.model.get_input_embeddings().weight.detach().to(self.device)  # (V, D)
        self.emb_norm = torch.nn.functional.normalize(emb, dim=1).contiguous()   # (V, D)
        self.vocab_size = self.emb_norm.shape[0]
        self.special_ids = set(self.tokenizer.all_special_ids)

    @torch.no_grad()
    def _topk(self, vecs: torch.Tensor, k: int):
        scores = vecs @ self.emb_norm.T  # (N, V), cosine because emb_norm normalized
        topk_scores, topk_ids = torch.topk(scores, k=min(k, self.vocab_size), dim=1, largest=True)
        return topk_ids, topk_scores

    def _eligible(self, tid: int, tok: str, cfg: SNOConfig) -> bool:
        if tid in self.special_ids:
            return False
        if contains_sensitive_pattern(tok):
            return False
        if cfg.only_low_semantic and (not is_low_semantic_token(tok)):
            return False
        if not spm_clean(tok):
            return False
        return True

    @torch.no_grad()
    def perturb_once(self, text: str, cfg: SNOConfig, seed: int) -> Tuple[str, int]:
        rng = random.Random(seed)

        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"][0].to(self.device)  # (L,)
        ids_list = input_ids.tolist()
        toks = self.tokenizer.convert_ids_to_tokens(ids_list)

        eligible_pos = [i for i, (tid, tok) in enumerate(zip(ids_list, toks)) if self._eligible(tid, tok, cfg)]
        if not eligible_pos:
            return text.strip(), 0

        n_replace = max(1, int(math.ceil(len(ids_list) * cfg.replace_ratio)))
        n_replace = min(n_replace, len(eligible_pos))
        if cfg.max_replace_tokens is not None and cfg.max_replace_tokens > 0:
            n_replace = min(n_replace, cfg.max_replace_tokens)

        replace_pos = rng.sample(eligible_pos, n_replace)

        src_ids = torch.tensor([ids_list[p] for p in replace_pos], device=self.device, dtype=torch.long)  # (N,)
        src_vec = self.emb_norm[src_ids]  # (N, D)
        noise = torch.randn_like(src_vec) * cfg.sigma
        pert_vec = torch.nn.functional.normalize(src_vec + noise, dim=1)  # (N, D)

        topk_ids, topk_scores = self._topk(pert_vec, cfg.topk)  # (N, K)

        new_ids = ids_list[:]
        changed = 0

        for j, pos in enumerate(replace_pos):
            src_id = new_ids[pos]
            src_tok = toks[pos]
            src_emb = self.emb_norm[src_id]  # (D,)

            chosen = None
            for _ in range(cfg.max_attempts_per_token):
                scores = topk_scores[j] / max(1e-6, cfg.temperature)
                probs = torch.softmax(scores, dim=0)
                pick_k = torch.multinomial(probs, num_samples=1).item()
                cand_id = int(topk_ids[j, pick_k].item())

                if (not cfg.allow_same_token) and (cand_id == src_id):
                    continue
                if cand_id in self.special_ids:
                    continue

                cand_tok = self.tokenizer.convert_ids_to_tokens([cand_id])[0]
                if contains_sensitive_pattern(cand_tok):
                    continue
                if not same_coarse_type(src_tok, cand_tok):
                    continue
                if cfg.only_low_semantic and (not is_low_semantic_token(cand_tok)):
                    continue

                # NEW: token-level semantic similarity constraint
                cand_emb = self.emb_norm[cand_id]
                token_sim = float(torch.dot(src_emb, cand_emb).item())
                if token_sim < cfg.tau_token:
                    continue

                chosen = cand_id
                break

            if chosen is not None and chosen != src_id:
                new_ids[pos] = chosen
                changed += 1

        out = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return out, changed


# ----------------------------
# 4) Sentence similarity model (MiniLM)
# ----------------------------
class MiniLMSim:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.m = SentenceTransformer(model_path, device=device)

    @torch.no_grad()
    def sim(self, a: str, b: str) -> float:
        ea = self.m.encode(a, convert_to_tensor=True, normalize_embeddings=True)
        eb = self.m.encode(b, convert_to_tensor=True, normalize_embeddings=True)
        return float(util.cos_sim(ea, eb).item())


# ----------------------------
# 5) Strict non-decreasing selection (generate k1)
# ----------------------------
def make_k1_nondecrease_to_j(
    j: str,
    k: str,
    idx: int,
    pert: SNOEmbeddingPerturber,
    cfg: SNOConfig,
    sim_model: MiniLMSim,
    tau_keep: float,
    M: int,
    seed_base: int,
) -> Tuple[str, Dict]:
    """
    Strictly enforce:
      sim(k, k1) >= tau_keep
      sim(j, k1) >= sim(j, k)

    Token-level similarity is enforced inside perturb_once:
      token_sim(src, cand) >= cfg.tau_token

    If not found, fallback to k.
    Among valid candidates, choose the one with max sim(j, k1).
    """
    j = (j or "").strip()
    k = (k or "").strip()

    s_jk = sim_model.sim(j, k)
    target = s_jk

    best = None
    best_s_jk1 = -1e9
    best_stats = None

    for m in range(M):
        seed = seed_base + idx * 1000 + m
        k1, n_changed = pert.perturb_once(k, cfg, seed=seed)
        k1 = (k1 or "").strip()

        if (not k1) or (k1 == k):
            continue

        s_kk1 = sim_model.sim(k, k1)
        if s_kk1 < tau_keep:
            continue

        s_jk1 = sim_model.sim(j, k1)
        if s_jk1 < target:
            continue

        if s_jk1 > best_s_jk1:
            best_s_jk1 = s_jk1
            best = k1
            best_stats = {
                "sim_j_k": s_jk,
                "sim_k_k1": s_kk1,
                "sim_j_k1": s_jk1,
                "ok_nondecrease": 1,
                "ok_keep": 1,
                "fallback": 0,
                "n_changed": int(n_changed),
            }

    if best is not None:
        return best, best_stats

    return k, {
        "sim_j_k": s_jk,
        "sim_k_k1": 1.0,
        "sim_j_k1": s_jk,
        "ok_nondecrease": 1,
        "ok_keep": 1,
        "fallback": 1,
        "n_changed": 0,
    }


# ----------------------------
# 6) Writer utils (preserve original schema)
# ----------------------------
def build_output_schema(in_path: str) -> pa.Schema:
    orig_schema = pq.read_schema(in_path)
    orig_names = set(orig_schema.names)

    new_fields = []

    def add(name: str, field: pa.Field):
        if name not in orig_names:
            new_fields.append(field)

    add("response_k1",     pa.field("response_k1", pa.string()))
    add("sim_j_k",         pa.field("sim_j_k", pa.float32()))
    add("sim_k_k1",        pa.field("sim_k_k1", pa.float32()))
    add("sim_j_k1",        pa.field("sim_j_k1", pa.float32()))
    add("ok_nondecrease",  pa.field("ok_nondecrease", pa.int32()))
    add("ok_keep",         pa.field("ok_keep", pa.int32()))
    add("fallback",        pa.field("fallback", pa.int32()))
    add("n_changed",       pa.field("n_changed", pa.int32()))

    return pa.schema(list(orig_schema) + new_fields)


def write_rows(writer: pq.ParquetWriter, rows: List[Dict], schema: pa.Schema):
    if not rows:
        return
    table = pa.Table.from_pylist(rows, schema=schema)
    writer.write_table(table)


# ----------------------------
# 7) Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="rl", choices=["rl", "evaluation"])

    ap.add_argument("--embed_model", type=str, required=True)
    ap.add_argument("--sim_model", type=str, required=True)

    ap.add_argument("--out_path", type=str, required=True)

    # perturb cfg
    ap.add_argument("--replace_ratio", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=0.06)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max_attempts_per_token", type=int, default=20)
    ap.add_argument("--only_low_semantic", action="store_true")
    ap.add_argument("--max_replace_tokens", type=int, default=-1)
    ap.add_argument("--tau_token", type=float, default=0.80)  # NEW

    # gating params
    ap.add_argument("--tau_keep", type=float, default=0.90)
    ap.add_argument("--M", type=int, default=24)
    ap.add_argument("--seed_base", type=int, default=100000)

    # processing
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--save_every", type=int, default=500)

    ap.add_argument("--debug_jsonl", type=str, default=None)

    args = ap.parse_args()

    in_path = os.path.join(
        args.data_root,
        "rl" if args.split == "rl" else "evaluation",
        "train-00000-of-00001.parquet"
    )

    ds = load_dataset("parquet", data_files=in_path, split="train")

    n = len(ds)
    s = max(0, args.start)
    e = n if args.end < 0 else min(n, args.end)
    assert s < e, f"Invalid range: start={s}, end={e}, n={n}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pert = SNOEmbeddingPerturber(args.embed_model, device=device)
    sim_model = MiniLMSim(args.sim_model, device=device)

    cfg = SNOConfig(
        replace_ratio=args.replace_ratio,
        sigma=args.sigma,
        topk=args.topk,
        temperature=args.temperature,
        max_attempts_per_token=args.max_attempts_per_token,
        only_low_semantic=args.only_low_semantic,
        allow_same_token=False,
        max_replace_tokens=args.max_replace_tokens,
        tau_token=args.tau_token,   # NEW
    )

    out_schema = build_output_schema(in_path)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    writer = pq.ParquetWriter(args.out_path, schema=out_schema, compression="zstd")

    dbg_f = None
    if args.debug_jsonl:
        os.makedirs(os.path.dirname(args.debug_jsonl), exist_ok=True)
        dbg_f = open(args.debug_jsonl, "w", encoding="utf-8")

    buf: List[Dict] = []
    t0 = time.time()

    for idx in tqdm(range(s, e), desc=f"Build response_k1 (token+sentence constrained) ({args.split}) [{s},{e})"):
        ex = ds[idx]
        j = ex.get("response_j", "")
        k = ex.get("response_k", "")

        k1, stats = make_k1_nondecrease_to_j(
            j=str(j),
            k=str(k),
            idx=idx,
            pert=pert,
            cfg=cfg,
            sim_model=sim_model,
            tau_keep=args.tau_keep,
            M=args.M,
            seed_base=args.seed_base,
        )

        row = dict(ex)
        row.update({
            "response_k1": k1,
            "sim_j_k": float(stats["sim_j_k"]),
            "sim_k_k1": float(stats["sim_k_k1"]),
            "sim_j_k1": float(stats["sim_j_k1"]),
            "ok_nondecrease": int(stats["ok_nondecrease"]),
            "ok_keep": int(stats["ok_keep"]),
            "fallback": int(stats["fallback"]),
            "n_changed": int(stats["n_changed"]),
        })
        buf.append(row)

        if dbg_f is not None:
            dbg_f.write(json.dumps({"idx": idx, **stats}, ensure_ascii=False) + "\n")

        if len(buf) >= args.save_every:
            write_rows(writer, buf, out_schema)
            buf = []

    if buf:
        write_rows(writer, buf, out_schema)

    writer.close()
    if dbg_f:
        dbg_f.close()

    dt = time.time() - t0
    print(f"Saved: {args.out_path} rows={e-s} speed={((e-s)/max(1e-6, dt)):.2f} rows/s")


if __name__ == "__main__":
    main()