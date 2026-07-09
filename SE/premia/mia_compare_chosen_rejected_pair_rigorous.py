#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, argparse
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser("PREMIA-style MIA (ρ / Δρ) with length & truncation controls")
    p.add_argument("--model_dir",  type=str, required=True, help="Reference/base or SFT model path (π_ref)")
    p.add_argument("--target_dir", type=str, required=True, help="Target/DPO/PPO model path or checkpoint (π_θ)")
    p.add_argument("--eval_json",  type=str, required=True, help="Path to mia_eval.json")
    p.add_argument("--out_dir",    type=str, required=True, help="Output directory")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_len",    type=int, default=1024)
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--dtype",      type=str, default="auto",
                   choices=["auto","float16","bfloat16","float32"])
    p.add_argument("--load_4bit",  action="store_true",
                   help="Load models in 4bit if available")
    p.add_argument("--skip_zero_answer", action="store_true",
                   help="Skip samples whose answer tokens become 0")
    p.add_argument("--filter_truncated", action="store_true",
                   help="Filter out samples truncated on either chosen or rejected")
    p.add_argument("--max_exp", type=float, default=20.0,
                   help="Clip log-ratio before exp to avoid overflow (exp(max_exp))")
    return p.parse_args()


def torch_dtype_from_str(s):
    if s == "auto":
        return None
    return dict(float16=torch.float16,
                bfloat16=torch.bfloat16,
                float32=torch.float32)[s]


# 统一极简模板，避免 ref/tgt 模板差异
def chat_pack(prompt, answer):
    return f"<user>\n{prompt}\n</user>\n<assistant>\n{answer}\n</assistant>"


@torch.no_grad()
def answer_logps(model, tok, prompt, answer, max_len=1024):
    """
    返回：
      sum_logp, mean_logp, ans_len, truncated_flag
    如果回答 token 为 0（被完全截断），返回 (None, None, 0, 1)
    """
    prompt_text = chat_pack(prompt, "")
    full_text   = chat_pack(prompt, answer)

    enc_p = tok(prompt_text, return_tensors="pt",
                truncation=True, max_length=max_len)
    enc_f = tok(full_text,   return_tensors="pt",
                truncation=True, max_length=max_len)

    input_ids = enc_f["input_ids"].to(model.device)
    attn      = enc_f["attention_mask"].to(model.device)

    start = enc_p["input_ids"].shape[1]
    seq_len = input_ids.shape[1]

    # 回答被完全截断：没有一个 answer token 可评估
    if start >= seq_len:
        return None, None, 0, 1

    labels = input_ids.clone()
    labels[:, :start] = -100  # 只评估回答区间
    ans_len = (labels != -100).sum(dim=1).float()
    if ans_len.item() == 0:
        return None, None, 0, int(seq_len >= max_len)

    out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    nll_mean = out.loss               # 平均 NLL（仅回答 token）
    nll_sum  = nll_mean * ans_len     # 总 NLL（回答 token 之和）

    mean_logp = -float(nll_mean.detach().cpu())
    sum_logp  = -float(nll_sum.detach().cpu().numpy()[0])
    truncated = int(seq_len >= max_len)
    return sum_logp, mean_logp, int(ans_len.item()), truncated


def batched(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def plot_roc(y, score, title, fname):
    mask = np.isfinite(score)
    y = y[mask]; score = score[mask]
    if len(np.unique(y)) < 2:
        print(f"[Warn] ROC skipped for {title}: labels not binary after mask.")
        return
    fpr, tpr, _ = roc_curve(y, score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"{title} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dtype = torch_dtype_from_str(args.dtype)
    load_kwargs = dict(device_map=args.device_map, torch_dtype=dtype)
    if args.load_4bit:
        load_kwargs.update(dict(load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16))

    print(f"[Load] REF  from {args.model_dir} (4bit={args.load_4bit}, dtype={args.dtype})")
    tok_ref = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True,
                                            trust_remote_code=True)
    ref     = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, **load_kwargs
    ).eval()

    print(f"[Load] TGT  from {args.target_dir} (4bit={args.load_4bit}, dtype={args.dtype})")
    tok_tgt = AutoTokenizer.from_pretrained(args.target_dir, use_fast=True,
                                            trust_remote_code=True)
    tgt     = AutoModelForCausalLM.from_pretrained(
        args.target_dir, trust_remote_code=True, **load_kwargs
    ).eval()

    print(f"[Data] {args.eval_json}")
    with open(args.eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Total eval samples:", len(data))

    # 这里的 “sum” / “mean” 指的是使用整句 logp 还是 per-token logp
    # score 本身是 ρ 或 Δρ（ratio 空间），对齐论文 Eq.(15)(17)。
    S = {
        "chosen_sum":[], "rejected_sum":[], "pair_sum":[],
        "chosen_mean":[], "rejected_mean":[], "pair_mean":[],
        "label":[],
        "ans_len_ch":[], "ans_len_rj":[],
        "trunc_ch":[], "trunc_rj":[]
    }

    skipped_zero_answer = 0

    def to_rho(delta, max_exp):
        """delta = log ρ；返回 ρ = exp(clipped(delta))"""
        if delta is None:
            return float("nan")
        if isinstance(delta, float) and (math.isnan(delta) or math.isinf(delta)):
            return float("nan")
        d = max(min(delta, max_exp), -max_exp)
        return math.exp(d)

    for batch in tqdm(batched(data, args.batch_size),
                      total=math.ceil(len(data)/args.batch_size)):
        for ex in batch:
            prompt   = ex.get("prompt","").strip()
            chosen   = ex.get("chosen","").strip()
            rejected = ex.get("rejected","").strip()

            c_ref_sum, c_ref_mean, Lc, Tc = answer_logps(
                ref, tok_ref, prompt, chosen,   args.max_len
            )
            r_ref_sum, r_ref_mean, Lr, Tr = answer_logps(
                ref, tok_ref, prompt, rejected, args.max_len
            )
            c_tgt_sum, c_tgt_mean, Lc2, Tc2 = answer_logps(
                tgt, tok_tgt, prompt, chosen,   args.max_len
            )
            r_tgt_sum, r_tgt_mean, Lr2, Tr2 = answer_logps(
                tgt, tok_tgt, prompt, rejected, args.max_len
            )

            # 任一边 0-token：可选跳过
            if args.skip_zero_answer and any(
                v is None for v in [c_ref_sum, r_ref_sum, c_tgt_sum, r_tgt_sum]
            ):
                skipped_zero_answer += 1
                continue

            # 将 None 替换为 NaN（如果不跳过）
            def nz(x): return float("nan") if x is None else x
            c_ref_sum, r_ref_sum, c_tgt_sum, r_tgt_sum = map(
                nz, [c_ref_sum, r_ref_sum, c_tgt_sum, r_tgt_sum]
            )
            c_ref_mean, r_ref_mean, c_tgt_mean, r_tgt_mean = map(
                nz, [c_ref_mean, r_ref_mean, c_tgt_mean, r_tgt_mean]
            )

            # ---- ρ 与 Δρ（PREMIA 定义）----
            # sum 口径：使用整句 logprob
            delta_ch_sum = c_tgt_sum - c_ref_sum   # log ρ_w
            delta_rj_sum = r_tgt_sum - r_ref_sum   # log ρ_l

            rho_ch_sum = to_rho(delta_ch_sum, args.max_exp)  # ρ_w
            rho_rj_sum = to_rho(delta_rj_sum, args.max_exp)  # ρ_l
            pair_sum   = rho_ch_sum - rho_rj_sum             # Δρ = ρ_w - ρ_l

            # mean 口径：per-token logprob 的 ratio（辅助分析，不是论文主口径）
            delta_ch_mean = c_tgt_mean - c_ref_mean
            delta_rj_mean = r_tgt_mean - r_ref_mean

            rho_ch_mean = to_rho(delta_ch_mean, args.max_exp)
            rho_rj_mean = to_rho(delta_rj_mean, args.max_exp)
            pair_mean   = rho_ch_mean - rho_rj_mean

            S["chosen_sum"].append(rho_ch_sum)
            S["rejected_sum"].append(rho_rj_sum)
            S["pair_sum"].append(pair_sum)

            S["chosen_mean"].append(rho_ch_mean)
            S["rejected_mean"].append(rho_rj_mean)
            S["pair_mean"].append(pair_mean)

            S["label"].append(int(ex.get("member",0)))

            # 诊断信息：回答长度与截断
            S["ans_len_ch"].append(Lc)
            S["ans_len_rj"].append(Lr)
            S["trunc_ch"].append(int(Tc or Tc2))
            S["trunc_rj"].append(int(Tr or Tr2))

    # 过滤截断样本（可选）
    if args.filter_truncated:
        mask = (np.array(S["trunc_ch"]) == 0) & (np.array(S["trunc_rj"]) == 0)
        for k in ["chosen_sum","rejected_sum","pair_sum",
                  "chosen_mean","rejected_mean","pair_mean",
                  "label","ans_len_ch","ans_len_rj","trunc_ch","trunc_rj"]:
            arr = np.array(S[k])
            S[k] = list(arr[mask])

    # 去除 NaN 的保险
    def finite_mask(*arrs):
        m = np.ones(len(arrs[0]), dtype=bool)
        for a in arrs:
            a = np.array(a)
            m &= np.isfinite(a)
        return m

    # sum 口径：整句概率，对齐论文 ρ_y / Δρ 主定义
    M_sum = finite_mask(S["chosen_sum"], S["rejected_sum"], S["pair_sum"])
    y_sum = np.array(S["label"])[M_sum]
    ch_sum = np.array(S["chosen_sum"])[M_sum]
    rj_sum = np.array(S["rejected_sum"])[M_sum]
    pr_sum = np.array(S["pair_sum"])[M_sum]

    # mean 口径：per-token 归一版本（仅用于敏感性分析）
    M_mean = finite_mask(S["chosen_mean"], S["rejected_mean"], S["pair_mean"])
    y_mean = np.array(S["label"])[M_mean]
    ch_mean = np.array(S["chosen_mean"])[M_mean]
    rj_mean = np.array(S["rejected_mean"])[M_mean]
    pr_mean = np.array(S["pair_mean"])[M_mean]

    def safe_auc(y, s):
        try:
            return roc_auc_score(y, s)
        except Exception:
            return float("nan")

    aucs = dict(
        # sum：论文主指标（推荐看这几个来对 Table 1 / 2）
        AUROC_chosen_sum    = safe_auc(y_sum,  ch_sum),
        AUROC_rejected_sum  = safe_auc(y_sum,  rj_sum),
        AUROC_pair_sum      = safe_auc(y_sum,  pr_sum),
        # mean：辅助
        AUROC_chosen_mean   = safe_auc(y_mean, ch_mean),
        AUROC_rejected_mean = safe_auc(y_mean, rj_mean),
        AUROC_pair_mean     = safe_auc(y_mean, pr_mean),
        num_samples_raw     = len(S["label"]),
        num_samples_sum     = int(y_sum.shape[0]),
        num_samples_mean    = int(y_mean.shape[0]),
        trunc_rate_ch       = float(np.mean(np.array(S["trunc_ch"])) if len(S["trunc_ch"]) else 0.0),
        trunc_rate_rj       = float(np.mean(np.array(S["trunc_rj"])) if len(S["trunc_rj"]) else 0.0),
        avg_len_ch          = float(np.mean(np.array(S["ans_len_ch"])) if len(S["ans_len_ch"]) else 0.0),
        avg_len_rj          = float(np.mean(np.array(S["ans_len_rj"])) if len(S["ans_len_rj"]) else 0.0),
        skipped_zero_answer = skipped_zero_answer,
    )

    # 保存明细 CSV
    df = pd.DataFrame(S)
    df.to_csv(os.path.join(args.out_dir, "rigorous_scores.csv"), index=False)

    # 保存总结
    pd.Series(aucs).to_csv(os.path.join(args.out_dir, "rigorous_summary.csv"))

    print("[Summary]")
    for k,v in aucs.items():
        print(f"  {k}: {v}")

    # 画 ROC：标题里标明是 ρ / Δρ
    # mean（辅助观察）
    plot_roc(y_mean, ch_mean, "MIA ROC (mean) - Chosen (ρ_mean)",
             os.path.join(args.out_dir, "rigorous_roc_chosen_mean.png"))
    plot_roc(y_mean, rj_mean, "MIA ROC (mean) - Rejected (ρ_mean)",
             os.path.join(args.out_dir, "rigorous_roc_rejected_mean.png"))
    plot_roc(y_mean, pr_mean, "MIA ROC (mean) - Pair Δρ_mean",
             os.path.join(args.out_dir, "rigorous_roc_pair_mean.png"))

    # sum（主观察）
    plot_roc(y_sum, ch_sum, "MIA ROC (sum) - Chosen (ρ)",
             os.path.join(args.out_dir, "rigorous_roc_chosen_sum.png"))
    plot_roc(y_sum, rj_sum, "MIA ROC (sum) - Rejected (ρ)",
             os.path.join(args.out_dir, "rigorous_roc_rejected_sum.png"))
    plot_roc(y_sum, pr_sum, "MIA ROC (sum) - Pair Δρ",
             os.path.join(args.out_dir, "rigorous_roc_pair_sum.png"))


if __name__ == "__main__":
    main()
