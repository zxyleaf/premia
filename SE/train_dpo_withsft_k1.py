# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

import argparse
import os
from typing import Optional

import torch
from accelerate import logging
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM

from trl import (
    DatasetMixtureConfig,
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,            # 保留以兼容纯 LoRA 初始化
    get_quantization_config,
)

# MOD: 额外导入 PeftModel 用来把你 SFT 阶段的 LoRA adapter 接到基座模型上
from peft import PeftModel

logger = logging.get_logger(__name__)

# 让 HF Spaces 和本地都能看到日志（无害）
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

# ------------------------------
# 辅助函数
# ------------------------------
def _sample_dataset(ds, n: Optional[int], seed: int):
    try:
        if n is None or int(n) <= 0 or int(n) >= len(ds):
            return ds
        n = int(n)
        logger.info(f"[SAMPLE] Shuffling and selecting first {n} / {len(ds)} examples (seed={seed})…")
        return ds.shuffle(seed=seed).select(range(n))
    except Exception as e:
        logger.warning(f"[SAMPLE] Sampling skipped due to error: {e}")
        return ds

def _build_se_paths(base_path: str, rl_filename: str):
    """
    支持通过 --rl_filename 指定 RL 训练文件名。
    例如：
      --dataset_name /path/to/data_seed42_80k
      --rl_filename train_with_response_k1.parquet

    则会拼成：
      /path/to/data_seed42_80k/rl/train_with_response_k1.parquet
    """
    base = os.path.abspath(base_path)

    if base.endswith("/rl"):
        rl_glob = os.path.join(base, rl_filename)
        eval_glob = os.path.join(os.path.dirname(base), "evaluation", "train-*.parquet")
    elif base.endswith("/evaluation"):
        eval_glob = os.path.join(base, "train-*.parquet")
        rl_glob = os.path.join(os.path.dirname(base), "rl", rl_filename)
    else:
        rl_glob = os.path.join(base, "rl", rl_filename)
        eval_glob = os.path.join(base, "evaluation", "train-*.parquet")

    return rl_glob, eval_glob

def _to_dpo_triplet(ex):
    """
    Mirror to k1:
      prompt   = question
      chosen   = response_j (preferred)
      rejected = response_k1 (obfuscated / rewritten rejected), fallback to response_k/response_l
    """
    rejected = (ex.get("response_k1") or ex.get("response_k") or ex.get("response_l") or "")
    return {
        "prompt": (ex.get("question") or "").strip(),
        "chosen": (ex.get("response_j") or "").strip(),   # preferred
        "rejected": rejected.strip(),
    }

# ------------------------------
# 主流程
# ------------------------------
def main(script_args, training_args, model_args, dataset_args, extra_opts):
    # ========== 模型加载 ==========
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    base_model_path = getattr(extra_opts, "base_model_name_or_path", None) or model_args.model_name_or_path

    logger.info(f"[MODEL] base={base_model_path}")
    logger.info(f"[MODEL] quantization_config={bool(quantization_config)} device_map in k-bit mode")

    # 先加载“可训练模型”
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    # MOD: 如果给了 SFT LoRA 的路径，则把 LoRA 适配器加载到 model 上（承接 SFT 权重）
    sft_peft_path = getattr(extra_opts, "sft_peft_path", None)
    if sft_peft_path:
        logger.info(f"[PEFT] Loading SFT LoRA adapter from: {sft_peft_path}")
        model = PeftModel.from_pretrained(model, sft_peft_path, is_trainable=True)
    else:
        logger.warning("[WARN] --sft_peft_path 未提供，将直接从 base_model 进行 DPO（不承接 SFT LoRA）。")

    # ref model：**冻结**一份作为参考
    logger.info("[REF] Building frozen reference model ...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    if sft_peft_path:
        ref_model = PeftModel.from_pretrained(ref_model, sft_peft_path, is_trainable=False)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    # ========== 数据 ==========
    rl_filename = getattr(extra_opts, "rl_filename", "train_with_response_k1.parquet")
    rl_glob, eval_glob = _build_se_paths(script_args.dataset_name, rl_filename)
    logger.info(f"[SE] rl_glob={rl_glob}")
    logger.info(f"[SE] eval_glob={eval_glob}")

    print("[SE] rl_glob", rl_glob)
    print("[SE] eval_glob", eval_glob)

    ds_train = load_dataset("parquet", data_files={"train": rl_glob})["train"]
    ds_eval  = load_dataset("parquet", data_files={"train": eval_glob})["train"]
    dataset  = DatasetDict({"train": ds_train, "test": ds_eval})

    # ========= （可选）子采样：先采样再 map，避免对超大 eval 全量 map =========
    train_n = getattr(extra_opts, "train_sample_size", None)
    eval_n  = getattr(extra_opts, "eval_sample_size", None)
    seed    = getattr(extra_opts, "sample_seed", 42)

    if train_n is not None and int(train_n) > 0:
        dataset["train"] = _sample_dataset(dataset["train"], int(train_n), seed)

    if training_args.eval_strategy != "no" and "test" in dataset and eval_n is not None and int(eval_n) > 0:
        dataset["test"] = _sample_dataset(dataset["test"], int(eval_n), seed)

    logger.info(f"[DATA] (pre-map) sizes -> train={len(dataset['train'])} | "
                f"test={len(dataset['test']) if 'test' in dataset else 0}")

    # ========= map -> DPO triplet：分别对 train/test 处理，remove_columns 各自独立 =========
    def _ensure_cols(ds, cols, split_name):
        missing = [c for c in cols if c not in ds.column_names]
        if missing:
            raise ValueError(f"[DATA][{split_name}] Missing columns: {missing}. "
                             f"Available: {ds.column_names}")

    # MOD: mirror to k1 -> chosen uses response_j, rejected prefers response_k1
    _ensure_cols(dataset["train"], ["question", "response_j"], "train")
    if not (("response_k1" in dataset["train"].column_names) or
            ("response_k" in dataset["train"].column_names) or
            ("response_l" in dataset["train"].column_names)):
        raise ValueError(f"[DATA][train] Need 'response_k1' (preferred) or 'response_k' or 'response_l' for rejected. "
                         f"Available: {dataset['train'].column_names}")

    train_remove_cols = list(dataset["train"].column_names)
    dataset["train"] = dataset["train"].map(
        _to_dpo_triplet,
        remove_columns=train_remove_cols,
        desc="[MAP][train] to DPO triplet",
        num_proc=max(1, (os.cpu_count() or 4) // 2),
    )

    if "test" in dataset:
        _ensure_cols(dataset["test"], ["question", "response_j"], "test")
        if not (("response_k1" in dataset["test"].column_names) or
                ("response_k" in dataset["test"].column_names) or
                ("response_l" in dataset["test"].column_names)):
            raise ValueError(f"[DATA][test] Need 'response_k1' (preferred) or 'response_k' or 'response_l' for rejected. "
                             f"Available: {dataset['test'].column_names}")

        test_remove_cols = list(dataset["test"].column_names)
        dataset["test"] = dataset["test"].map(
            _to_dpo_triplet,
            remove_columns=test_remove_cols,
            desc="[MAP][test] to DPO triplet",
            num_proc=max(1, (os.cpu_count() or 4) // 2),
        )

    logger.info(f"[DATA] (post-map) sizes -> train={len(dataset['train'])} | "
                f"test={len(dataset['test']) if 'test' in dataset else 0}")

    # ========== DPO 配置 ==========
    dpo_beta = getattr(training_args, "beta", None)
    if dpo_beta is None:
        dpo_beta = getattr(extra_opts, "beta", 0.2)
    logger.info(f"[DPO] beta={dpo_beta}")

    # ========== Trainer ==========
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=None,  # 已显式承接 SFT LoRA
    )

    # 训练
    resume_ckpt = getattr(extra_opts, "resume_from_checkpoint", None)
    if resume_ckpt:
        logger.info(f"[RESUME] Resuming from checkpoint: {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()
    trainer.accelerator.print("✅ Training completed.")

    # 评估
    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 保存
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    dataclass_types = (ScriptArguments, DPOConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("dpo", help="Run the DPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, dataset_args, extra = parser.parse_args_and_config(
        return_remaining_strings=True
    )

    # 这里用“额外参数解析器”接住我们新增的 flags
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--train_sample_size", type=int, default=None)
    extra_parser.add_argument("--eval_sample_size", type=int, default=None)
    extra_parser.add_argument("--sample_seed", type=int, default=42)

    # MOD: 新增两个关键 flag
    extra_parser.add_argument("--base_model_name_or_path", type=str, default=None,
                              help="若与 --model_name_or_path 不同，可指定基座模型路径")
    extra_parser.add_argument("--sft_peft_path", type=str, default=None,
                              help="承接的 SFT LoRA 目录（包含 adapter_config.json / adapter_model.safetensors）")

    extra_parser.add_argument(
        "--rl_filename",
        type=str,
        default="train_with_response_k1.parquet",
        help="RL 训练集文件名，例如 train_with_response_k1.parquet 或 train_with_response_j1.parquet",
    )

    # MOD: 兜底 beta（如果 DPOConfig 未从 CLI 注入成功）
    extra_parser.add_argument("--beta", type=float, default=0.2)
    extra_parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="checkpoint 路径；例如 output_dir/checkpoint-1000",
    )

    extra_opts, _ = extra_parser.parse_known_args(extra)

    # 环境变量兜底
    if extra_opts.train_sample_size is None:
        env_train = os.getenv("TRAIN_SAMPLE_SIZE")
        if env_train:
            extra_opts.train_sample_size = int(env_train)
    if extra_opts.eval_sample_size is None:
        env_eval = os.getenv("EVAL_SAMPLE_SIZE")
        if env_eval:
            extra_opts.eval_sample_size = int(env_eval)
    if "SAMPLE_SEED" in os.environ:
        extra_opts.sample_seed = int(os.getenv("SAMPLE_SEED", extra_opts.sample_seed))

    # 打印一下关键路径，方便你在日志里确认
    logger.info(f"[ENV] RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
                f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"[ARGS] base_model_name_or_path={getattr(extra_opts, 'base_model_name_or_path', None)} "
                f"sft_peft_path={getattr(extra_opts, 'sft_peft_path', None)}")

    main(script_args, training_args, model_args, dataset_args, extra_opts)