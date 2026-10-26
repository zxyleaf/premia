# Blinded LLM-as-Judge Evaluation Package

This directory contains the evaluation inputs needed for two concerns raised in the AAAI reviews. It contains **no observed judge results**.

## What is evaluated

### 1. Held-out generation quality

`inputs/generation_quality_blind.jsonl` contains 4,000 pairwise comparisons: 500 prompts for each of the paper's eight backbones. Each item presents one archived DPO generation and one archived PRP-R generation as anonymous Candidate A/B. The prompt and reference match exactly within every pair.

This audit addresses the request for an independent generation preference assessment beyond reward accuracy, reward margin, perplexity, and reference-overlap metrics. It does not measure privacy and cannot show that PRP-R caused a utility difference.

Three archived pairs have known configuration limitations that an LLM judge cannot repair:

- Open-LLaMA-7B: different optimizer-update counts;
- Qwen3-1.7B: different saved EOS IDs;
- Gemma-2-2B: two versus three reported training epochs.

Open-LLaMA-3B and Gemma-3 also used different PyTorch versions between DPO and PRP-R. Qwen3.5's archived DPO generation target differs from the target used in its reward-table row. Results for these files must therefore be described as paired comparisons of the archived outputs, not uniformly controlled method-only effects.

### 2. Sanitized preference-label validity

`inputs/preference_label_validity_blind.jsonl` contains 240 comparisons from 120 source training tuples. Every source tuple contributes two independently shuffled questions:

- chosen versus original rejected;
- chosen versus sanitized rejected.

The sample is stratified by fallback and realized edit count. It intentionally oversamples uncommon strata; aggregate estimates must use the population weights implemented in `analyze_judgments.py`. This audit tests whether the chosen response is still preferred after sanitization. It does not establish a formal utility guarantee, and an LLM judge is not a substitute for the proposed two-annotator human audit.

## Blinding and frozen evidence

Public inputs contain only random A/B positions. Method labels, original/transformed conditions, chosen-side positions, and source-row pairings are stored under `private/`, which is ignored by Git. Do not give those files to the judge. Freeze the raw result JSONL before revealing the key.

`manifest.json` records the exact source-file and public-input SHA-256 hashes. `prepare_inputs.py` deterministically rebuilds the inputs and local keys when the original experiment files are present.

## Running the judge

The judge uses temperature 0 and evaluates every item twice with A/B positions swapped. Position-inconsistent cases are reported separately rather than silently forced into a winner.

On the machine that can access the Azure endpoint, clone/update the repository
and create a small evaluation environment:

```bash
git clone https://github.com/zxyleaf/premia.git
cd premia
git pull origin main

conda create -n premia-judge python=3.10 -y
conda activate premia-judge
python -m pip install -r llm_judge_evaluation/requirements.txt
```

Set the Azure credentials in environment variables; never put an API key in
this repository, a shell script, or a committed `.env` file:

```bash
export AZURE_OPENAI_ENDPOINT='YOUR_AZURE_ENDPOINT'
export AZURE_OPENAI_DEPLOYMENT='YOUR_DEPLOYMENT_NAME'
export AZURE_OPENAI_API_VERSION='2024-03-01-preview'
read -s AZURE_OPENAI_API_KEY
export AZURE_OPENAI_API_KEY
```

First run a two-item smoke test into disposable output paths:

```bash
python llm_judge_evaluation/run_blind_judge.py \
  --input-jsonl llm_judge_evaluation/inputs/generation_quality_blind.jsonl \
  --output-jsonl /tmp/premia_judge_smoke.jsonl \
  --summary-json /tmp/premia_judge_smoke_summary.json \
  --limit 2
```

Then run the complete experiment. The launcher is resumable, so rerunning the
same command after an API/network failure continues from completed item IDs:

```bash
bash llm_judge_evaluation/run_all.sh all
```

You can also run the two tasks separately:

```bash
bash llm_judge_evaluation/run_all.sh generation
bash llm_judge_evaluation/run_all.sh preference
```

The complete run requires 8,480 model calls: two positions for each of 4,000 generation items and 240 label-validity items. `--resume` safely skips completed item IDs.

The blind summaries contain only A/B counts and therefore do not reveal which
method won. Check completion with:

```bash
wc -l llm_judge_evaluation/results/generation_quality_raw.jsonl
wc -l llm_judge_evaluation/results/preference_label_raw.jsonl
python -m json.tool llm_judge_evaluation/results/generation_quality_blind_summary.json
python -m json.tool llm_judge_evaluation/results/preference_label_blind_summary.json
```

A complete run has exactly 4,000 and 240 raw-result lines, respectively.

After the raw outputs are frozen, copy the two raw JSONL files back to the
original `/data1/zxy/Premia` checkout, where the ignored private keys remain.
Do not commit either private key. Then reveal and summarize locally:

```bash
conda run -n premia python llm_judge_evaluation/analyze_judgments.py \
  --task generation \
  --results llm_judge_evaluation/results/generation_quality_raw.jsonl \
  --private-key llm_judge_evaluation/private/generation_quality_PRIVATE_key.jsonl \
  --output llm_judge_evaluation/results/generation_quality_revealed_summary.json

conda run -n premia python llm_judge_evaluation/analyze_judgments.py \
  --task preference \
  --results llm_judge_evaluation/results/preference_label_raw.jsonl \
  --private-key llm_judge_evaluation/private/preference_label_validity_PRIVATE_key.jsonl \
  --output llm_judge_evaluation/results/preference_label_revealed_summary.json
```

Report raw wins/ties/losses, invalid and position-inconsistent counts, tie-adjusted scores, and 95% prompt-bootstrap intervals. For label validity, report the population-weighted original and transformed chosen scores plus their paired stratified-bootstrap difference interval.

The revealed generation summary is organized by backbone and reports PRP-R
wins, DPO wins, ties, excluded position-inconsistent/invalid items, the
tie-adjusted PRP-R score, and its 95% prompt-bootstrap interval. A score of 0.5
means parity; intervals should be interpreted directly rather than converting
a near-0.5 point estimate into a win claim.

## Scope and data notice

These files do not answer the adaptive-attack, multiple-training-seed, DP-accounting, or low-FPR calibration concerns. The text originates from the StackExchange preference dataset and remains subject to the upstream dataset/content licensing and attribution requirements; verify those requirements before making the data package public.
