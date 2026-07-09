#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPO trainer for PROPS experiments, adapted from the user's train_dpo_withsft.py.

Key change vs the uploaded script:
  - separate --init_peft_path and --ref_peft_path.

Why this matters for PROPS:
  Stage 1: init_peft_path = SFT adapter,    ref_peft_path = SFT adapter.
  Stage 2: init_peft_path = Stage-1 adapter, ref_peft_path = SFT adapter.

The original train_dpo_withsft.py uses one --sft_peft_path for both the trainable model
and reference model. That is fine for ordinary DPO or PROPS stage 1, but it is not the
right stage-2 setup if you want M2 initialized from M1 while keeping the DPO reference
as the SFT model.
"""

import argparse
import os
from typing import Optional, Tuple

import torch
from accelerate import logging
from datasets import DatasetDict, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM

from trl import (
    DatasetMixtureConfig,
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_quantization_config,
)

logger = logging.get_logger(__name__)
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


def _sample_dataset(ds, n: Optional[int], seed: int):
    try:
        if n is None or int(n) <= 0 or int(n) >= len(ds):
            return ds
        n = int(n)
        logger.info(f"[SAMPLE] Shuffling and selecting first {n} / {len(ds)} examples (seed={seed})")
        return ds.shuffle(seed=seed).select(range(n))
    except Exception as e:
        logger.warning(f"[SAMPLE] Sampling skipped due to error: {e}")
        return ds


def _build_se_paths(base_path: str, rl_filename: Optional[str] = None, eval_filename: Optional[str] = None) -> Tuple[str, str]:
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
    return rl_glob, eval_glob


def _first_existing(ex, keys):
    for k in keys:
        v = ex.get(k)
        if v is not None:
            return v
    return ""


def _to_dpo_triplet(ex):
    """
    Output schema required by TRL DPOTrainer:
      prompt, chosen, rejected

    PROPS data builder writes response_j1/response_k.
    This mapper also tolerates response_j/response_J and response_k1/response_K.
    """
    prompt = _first_existing(ex, ["question", "prompt"])
    chosen = _first_existing(ex, ["response_j1", "response_j", "response_J", "chosen"])
    rejected = _first_existing(ex, ["response_k", "response_k1", "response_K", "rejected"])
    return {
        "prompt": (prompt or "").strip(),
        "chosen": (chosen or "").strip(),
        "rejected": (rejected or "").strip(),
    }


def _ensure_preference_columns(ds, split_name: str):
    cols = set(ds.column_names)
    has_prompt = bool({"question", "prompt"} & cols)
    has_chosen = bool({"response_j1", "response_j", "response_J", "chosen"} & cols)
    has_rejected = bool({"response_k", "response_k1", "response_K", "rejected"} & cols)
    if not has_prompt or not has_chosen or not has_rejected:
        raise ValueError(
            f"[DATA][{split_name}] Need prompt/question, chosen, rejected columns. "
            f"Available columns: {ds.column_names}"
        )


def _torch_dtype_from_model_args(model_args):
    dtype = model_args.dtype
    if dtype in [None, "auto"]:
        return dtype
    if isinstance(dtype, str) and hasattr(torch, dtype):
        return getattr(torch, dtype)
    return dtype


def _load_base_plus_adapter(base_model_path: str, adapter_path: Optional[str], model_args, model_kwargs, is_trainable: bool):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    if adapter_path:
        logger.info(f"[PEFT] Loading adapter={adapter_path} | is_trainable={is_trainable}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable)
    else:
        logger.warning("[PEFT] No adapter path provided; using the base model directly.")
    return model


def main(script_args, training_args, model_args, dataset_args, extra_opts):
    # ----------------
    # Model loading
    # ----------------
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=_torch_dtype_from_model_args(model_args),
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    base_model_path = extra_opts.base_model_name_or_path or model_args.model_name_or_path
    init_peft_path = extra_opts.init_peft_path or extra_opts.sft_peft_path
    ref_peft_path = extra_opts.ref_peft_path or extra_opts.sft_peft_path or init_peft_path

    logger.info(f"[MODEL] base_model_path={base_model_path}")
    logger.info(f"[MODEL] init_peft_path={init_peft_path}")
    logger.info(f"[MODEL] ref_peft_path={ref_peft_path}")

    model = _load_base_plus_adapter(base_model_path, init_peft_path, model_args, model_kwargs, is_trainable=True)
    ref_model = _load_base_plus_adapter(base_model_path, ref_peft_path, model_args, model_kwargs, is_trainable=False)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    # ----------------
    # Dataset loading
    # ----------------
    rl_glob, eval_glob = _build_se_paths(
        script_args.dataset_name,
        rl_filename=extra_opts.rl_filename,
        eval_filename=extra_opts.eval_filename,
    )
    logger.info(f"[SE] rl_glob={rl_glob}")
    logger.info(f"[SE] eval_glob={eval_glob}")
    print("[SE] rl_glob", rl_glob, flush=True)
    print("[SE] eval_glob", eval_glob, flush=True)

    ds_train = load_dataset("parquet", data_files={"train": rl_glob})["train"]
    ds_eval = load_dataset("parquet", data_files={"train": eval_glob})["train"]
    dataset = DatasetDict({"train": ds_train, "test": ds_eval})

    train_n = extra_opts.train_sample_size
    eval_n = extra_opts.eval_sample_size
    seed = extra_opts.sample_seed
    if train_n is not None and int(train_n) > 0:
        dataset["train"] = _sample_dataset(dataset["train"], int(train_n), seed)
    if training_args.eval_strategy != "no" and eval_n is not None and int(eval_n) > 0:
        dataset["test"] = _sample_dataset(dataset["test"], int(eval_n), seed)

    logger.info(f"[DATA] pre-map sizes -> train={len(dataset['train'])} | test={len(dataset['test'])}")
    _ensure_preference_columns(dataset["train"], "train")
    if training_args.eval_strategy != "no":
        _ensure_preference_columns(dataset["test"], "test")

    train_remove_cols = list(dataset["train"].column_names)
    dataset["train"] = dataset["train"].map(
        _to_dpo_triplet,
        remove_columns=train_remove_cols,
        desc="[MAP][train] to DPO triplet",
        num_proc=max(1, (os.cpu_count() or 4) // 2),
    )

    if training_args.eval_strategy != "no":
        test_remove_cols = list(dataset["test"].column_names)
        dataset["test"] = dataset["test"].map(
            _to_dpo_triplet,
            remove_columns=test_remove_cols,
            desc="[MAP][test] to DPO triplet",
            num_proc=max(1, (os.cpu_count() or 4) // 2),
        )

    logger.info(f"[DATA] post-map sizes -> train={len(dataset['train'])} | test={len(dataset['test'])}")

    dpo_beta = getattr(training_args, "beta", None)
    if dpo_beta is None:
        dpo_beta = extra_opts.beta
    logger.info(f"[DPO] beta={dpo_beta}")

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=None,  # adapter is loaded explicitly; do not create a new LoRA adapter here
    )

    if extra_opts.resume_from_checkpoint:
        logger.info(f"[RESUME] Resuming from checkpoint: {extra_opts.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=extra_opts.resume_from_checkpoint)
    else:
        trainer.train()
    trainer.accelerator.print("✅ Training completed.")

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    dataclass_types = (ScriptArguments, DPOConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("dpo", help="Run PROPS-compatible DPO training", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, dataset_args, extra = parser.parse_args_and_config(return_remaining_strings=True)

    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--train_sample_size", type=int, default=None)
    extra_parser.add_argument("--eval_sample_size", type=int, default=None)
    extra_parser.add_argument("--sample_seed", type=int, default=42)

    # Adapter/model paths
    extra_parser.add_argument("--base_model_name_or_path", type=str, default=None)
    extra_parser.add_argument("--init_peft_path", type=str, default=None,
                              help="Trainable initial adapter. Stage1=SFT, Stage2=PROPS Stage1.")
    extra_parser.add_argument("--ref_peft_path", type=str, default=None,
                              help="Frozen reference adapter. Usually SFT for both stages.")
    # Backward-compatible alias for your original script.
    extra_parser.add_argument("--sft_peft_path", type=str, default=None,
                              help="Alias: used as both init/ref if init_peft_path/ref_peft_path are not provided.")

    extra_parser.add_argument("--beta", type=float, default=0.2)
    extra_parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    extra_parser.add_argument("--rl_filename", type=str, default="train-*.parquet")
    extra_parser.add_argument("--eval_filename", type=str, default="train-*.parquet")

    extra_opts, _ = extra_parser.parse_known_args(extra)

    if extra_opts.train_sample_size is None and os.getenv("TRAIN_SAMPLE_SIZE"):
        extra_opts.train_sample_size = int(os.environ["TRAIN_SAMPLE_SIZE"])
    if extra_opts.eval_sample_size is None and os.getenv("EVAL_SAMPLE_SIZE"):
        extra_opts.eval_sample_size = int(os.environ["EVAL_SAMPLE_SIZE"])
    if "SAMPLE_SEED" in os.environ:
        extra_opts.sample_seed = int(os.getenv("SAMPLE_SEED", extra_opts.sample_seed))

    logger.info(
        f"[ENV] RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    main(script_args, training_args, model_args, dataset_args, extra_opts)
