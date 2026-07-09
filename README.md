# PREMIA-Style Privacy Experiments for Preference Alignment

This repository contains experimental scripts for studying privacy leakage and privacy defenses in preference-based large language model (LLM) alignment. The code is organized around StackExchange-style preference tuples with the schema

```text
(question, response_j, response_k)
```

where `response_j` is the chosen/preferred response and `response_k` is the rejected response. The repository supports standard SFT + DPO training, rejected-side preference tuple sanitization, DP-DPO baselines, and PROPS-style preference-label privacy experiments.

> This is a research codebase. Paths, model names, and hyperparameters in the examples below should be adapted to your local environment.

## Repository Structure

```text
premia/
├── SE/
│   ├── sft.py                         # Supervised fine-tuning on question + response_j
│   ├── sft_qwen35.py                   # Qwen-specific SFT variant
│   ├── train_dpo_withsft_k1.py          # DPO from an SFT LoRA adapter; supports response_k1
│   ├── dp_dpo_manual.py                 # Manual DP-DPO baseline with per-example clipping/noise
│   └── train_props_dpo_withsft.py        # PROPS-compatible DPO with separate init/ref adapters
│
└── dataset/
    ├── make_sno_paraphrase_parquet_reject_less.py  # Builds sanitized response_k1
    └── stack_rl_eval/data_seed42_80k/               # Example StackExchange-style data layout
```

## Main Components

### 1. SFT

`SE/sft.py` performs supervised fine-tuning by mapping preference-style columns to a single text field:

```text
text = question + "\n\n" + response_j
```

This produces an SFT adapter that is later used as both the DPO initialization and the DPO reference model.

### 2. DPO with Rejected-Side Sanitization

`SE/train_dpo_withsft_k1.py` trains a DPO model initialized from an SFT LoRA adapter. It maps each row to the DPO format:

```text
prompt   = question
chosen   = response_j
rejected = response_k1 if available, otherwise response_k / response_l
```

This is the main training entry for PTS-R-style experiments, where the rejected response is replaced by a sanitized version `response_k1`.

### 3. Rejected-Side Sanitization

`dataset/make_sno_paraphrase_parquet_reject_less.py` generates `response_k1` from `(response_j, response_k)`. The script enforces three constraints:

```text
1. sim(response_k, response_k1) >= tau_keep
2. sim(response_j, response_k1) >= sim(response_j, response_k)
3. token_sim(original_token, candidate_token) >= tau_token for replaced tokens
```

If the constraints cannot be satisfied within the configured number of trials, the script falls back to the original rejected response. It appends diagnostic columns such as:

```text
response_k1
sim_j_k
sim_k_k1
sim_j_k1
ok_nondecrease
ok_keep
fallback
n_changed
```

### 4. DP-DPO Baseline

`SE/dp_dpo_manual.py` implements a manual DP-DPO baseline for LoRA/QLoRA models. It performs one backward pass per sample, clips per-example gradients, adds Gaussian noise, and logs an approximate privacy budget if Opacus is installed.

This implementation is exact but slow, and is intended mainly as a baseline for privacy-utility comparison.

### 5. PROPS-Compatible DPO

`SE/train_props_dpo_withsft.py` supports PROPS-style staged training by separating the trainable initialization adapter and the frozen reference adapter:

```text
Stage 1: init_peft_path = SFT adapter,      ref_peft_path = SFT adapter
Stage 2: init_peft_path = Stage-1 adapter,  ref_peft_path = SFT adapter
```

This separation is necessary because the second stage should continue from the Stage-1 model while still using the original SFT model as the DPO reference.

## Environment Setup

A typical environment can be created as follows:

```bash
conda create -n premia python=3.10 -y
conda activate premia

pip install torch transformers datasets accelerate trl peft bitsandbytes \
  sentence-transformers pyarrow tqdm wandb opacus
```

Depending on your CUDA version and cluster environment, install the matching PyTorch build from the official PyTorch instructions.

## Data Layout

Most scripts expect a StackExchange-style local data root with the following layout:

```text
/path/to/data_seed42_80k/
├── rl/
│   └── train-00000-of-00001.parquet
└── evaluation/
    └── train-00000-of-00001.parquet
```

The original preference data should contain at least:

```text
question
response_j
response_k
```

After rejected-side sanitization, the training file should additionally contain:

```text
response_k1
```

For PROPS-style experiments, transformed files may also contain columns such as `response_j1`.

## Quick Start

### Step 1: Run SFT

```bash
accelerate launch SE/sft.py \
  --model_name_or_path /path/to/base_model \
  --dataset_name /path/to/data_seed42_80k \
  --output_dir outputs/sft_lora \
  --num_train_epochs 2 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 8e-5 \
  --bf16 \
  --report_to none
```

The output directory should contain a LoRA adapter that can be used by DPO scripts.

### Step 2: Build Rejected-Side Sanitized Data

```bash
python dataset/make_sno_paraphrase_parquet_reject_less.py \
  --data_root /path/to/data_seed42_80k \
  --split rl \
  --embed_model /path/to/base_or_lm_for_token_candidates \
  --sim_model sentence-transformers/all-MiniLM-L6-v2 \
  --out_path /path/to/data_seed42_80k/rl/train_with_response_k1.parquet \
  --replace_ratio 0.05 \
  --topk 10 \
  --temperature 0.5 \
  --tau_token 0.80 \
  --tau_keep 0.90 \
  --M 24 \
  --save_every 500
```

This produces a new parquet file with `response_k1` and diagnostic fields.

### Step 3: Train DPO on Sanitized Tuples

```bash
accelerate launch SE/train_dpo_withsft_k1.py \
  --model_name_or_path /path/to/base_model \
  --base_model_name_or_path /path/to/base_model \
  --sft_peft_path outputs/sft_lora \
  --dataset_name /path/to/data_seed42_80k \
  --rl_filename train_with_response_k1.parquet \
  --output_dir outputs/dpo_ptsr \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --beta 0.4 \
  --bf16 \
  --report_to none
```

If `response_k1` is not present, the script falls back to `response_k` or `response_l` as the rejected response.

### Step 4: Run DP-DPO Baseline

```bash
accelerate launch --num_processes 1 SE/dp_dpo_manual.py \
  --base_model_name_or_path /path/to/base_model \
  --sft_peft_path outputs/sft_lora \
  --dataset_name /path/to/data_seed42_80k \
  --rl_filename train-00000-of-00001.parquet \
  --output_dir outputs/dp_dpo_sigma1_c01 \
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
```

Notes:

- This script is slow because it computes one backward pass per sample.
- DP is applied to trainable parameters, usually the LoRA adapter parameters.
- The logged epsilon is an approximate reporting aid, not a formal privacy audit.

### Step 5: Run PROPS-Compatible DPO

Stage 1:

```bash
accelerate launch SE/train_props_dpo_withsft.py \
  --model_name_or_path /path/to/base_model \
  --base_model_name_or_path /path/to/base_model \
  --init_peft_path outputs/sft_lora \
  --ref_peft_path outputs/sft_lora \
  --dataset_name /path/to/props_stage1_data \
  --rl_filename train-00000-of-00001.parquet \
  --output_dir outputs/props_stage1 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --beta 0.4 \
  --bf16 \
  --report_to none
```

Stage 2:

```bash
accelerate launch SE/train_props_dpo_withsft.py \
  --model_name_or_path /path/to/base_model \
  --base_model_name_or_path /path/to/base_model \
  --init_peft_path outputs/props_stage1 \
  --ref_peft_path outputs/sft_lora \
  --dataset_name /path/to/props_stage2_data \
  --rl_filename train-00000-of-00001.parquet \
  --output_dir outputs/props_stage2 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --beta 0.4 \
  --bf16 \
  --report_to none
```

## Metrics Used in Experiments

The project is intended for privacy-utility evaluation of preference alignment. Typical metrics include:

### Privacy

- AUROC on chosen-response PREMIA-style score
- AUROC on rejected-response PREMIA-style score
- AUROC on pairwise likelihood-gap score

Lower AUROC is better; 0.5 indicates random guessing.

### Utility

- DPO reward accuracy
- DPO reward margin
- Perplexity on held-out chosen responses
- BERTScore
- ROUGE-L
- BLEU
- METEOR

## Recommended Experiment Naming

For reproducibility, include key settings in output names, for example:

```text
openllama3b_sft_lora_ba4_ga4_lr8e-5
openllama3b_dpo_original_ba8_ga2_lr5e-4_beta0.4
openllama3b_ptsr_k10_t05_tau08
openllama3b_dp_dpo_sigma1_c01_ba8_lr5e-5
openllama3b_props_eps05_stage2_ba8_ga2
```

## Common Pitfalls

1. **DPO reference model must match the SFT model.**  
   For ordinary DPO, both the trainable model and the reference model should start from the same SFT adapter. For PROPS Stage 2, the trainable model should initialize from Stage 1, but the reference should remain the SFT adapter.

2. **Check whether `response_k1` exists.**  
   If `response_k1` is missing, `train_dpo_withsft_k1.py` falls back to the original rejected response, so the run is no longer a rejected-side sanitization run.

3. **DP-DPO is computationally expensive.**  
   The manual implementation is intentionally simple and exact, but it is much slower than standard DPO because it performs one backward pass per sample.

4. **Keep training and evaluation files separate.**  
   Most scripts assume `rl/` for training and `evaluation/` for held-out evaluation.

## Citation

If you use this repository, please cite the relevant papers for the methods you run, such as DPO, PREMIA-style membership inference, DP-SGD/DP-DPO, and PROPS. A project-specific citation can be added here once the associated report or paper is finalized.

```bibtex
@misc{premia_privacy_alignment_repo,
  title        = {PREMIA-Style Privacy Experiments for Preference Alignment},
  author       = {Zhao, Xinyue},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/zxyleaf/premia}
}
```

## License

No license file is currently specified in this repository. Please add a `LICENSE` file before public reuse or release.
