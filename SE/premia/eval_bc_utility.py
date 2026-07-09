import os
import json
import math
import argparse
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# ---------------------------
# utils
# ---------------------------

def load_parquet(pattern, max_eval):
    files = glob.glob(pattern)
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs)
    df = df.sample(frac=1, random_state=42).head(max_eval)
    return df

def generate(model, tokenizer, prompts, max_new_tokens):
    outputs = []
    for p in tqdm(prompts):
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        outputs.append(text[len(p):].strip())
    return outputs

def compute_nll(model, tokenizer, prompts, responses):
    losses = []
    for p, r in zip(prompts, responses):
        text = p + r
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, labels=inputs["input_ids"])
        losses.append(out.loss.item())
    return np.mean(losses)

def distinct_k(texts, k):
    total = 0
    uniq = set()
    for t in texts:
        tokens = t.split()
        ngrams = zip(*[tokens[i:] for i in range(k)])
        ngrams = list(ngrams)
        total += len(ngrams)
        uniq.update(ngrams)
    return len(uniq) / max(total, 1)

def unique_k(texts, k):
    uniq = set()
    for t in texts:
        tokens = t.split()
        ngrams = zip(*[tokens[i:] for i in range(k)])
        uniq.update(ngrams)
    return len(uniq)

def repetition_k(texts, k):
    ratios = []
    for t in texts:
        tokens = t.split()
        ngrams = list(zip(*[tokens[i:] for i in range(k)]))
        if len(ngrams) == 0:
            continue
        c = Counter(ngrams)
        rep = sum(v for v in c.values() if v > 1)
        ratios.append(rep / len(ngrams))
    return np.mean(ratios) if ratios else 0

# ---------------------------
# main
# ---------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_parquet(args.eval_pattern, args.max_eval)

    questions = df["question"].tolist()
    refs = df["response_j"].tolist()

    models = json.loads(args.models_json)
    results = {}

    for name, path in models.items():
        print(f"\n=== Evaluating {name} ===")

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path).cuda().eval()

        preds = generate(model, tokenizer, questions, args.max_new_tokens)

        # ---------- B ----------
        nll = compute_nll(model, tokenizer, questions, refs)
        ppl = math.exp(nll)

        self_nll = compute_nll(model, tokenizer, questions, preds)
        self_ppl = math.exp(self_nll)

        P, R, F1 = bert_score(preds, refs, model_type=args.bert_model_type, device="cuda")
        bert_f1 = float(F1.mean())

        rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l = np.mean([rouge.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds)])

        bleu = np.mean([sentence_bleu([r.split()], p.split()) for r, p in zip(refs, preds)])
        meteor = np.mean([meteor_score([r], p) for r, p in zip(refs, preds)])

        avg_len = np.mean([len(p.split()) for p in preds])

        # ---------- C ----------
        d1 = distinct_k(preds, 1)
        d2 = distinct_k(preds, 2)
        u1 = unique_k(preds, 1)
        u2 = unique_k(preds, 2)
        rep2 = repetition_k(preds, 2)
        rep3 = repetition_k(preds, 3)

        results[name] = {
            "PPL": ppl,
            "Self_PPL": self_ppl,
            "BERTScore": bert_f1,
            "ROUGE_L": rouge_l,
            "BLEU": bleu,
            "METEOR": meteor,
            "avg_len": avg_len,
            "distinct_1": d1,
            "distinct_2": d2,
            "unique_1": u1,
            "unique_2": u2,
            "repetition_2": rep2,
            "repetition_3": rep3
        }

    with open(os.path.join(args.output_dir, "bc_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_pattern", type=str)
    parser.add_argument("--models_json", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--bert_model_type", type=str)
    parser.add_argument("--max_eval", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()
    main(args)