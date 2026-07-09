# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
#     "wandb",
# ]
# ///

import argparse
import os
from typing import Optional

from accelerate import logging
from datasets import DatasetDict, load_dataset
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

try:
    import wandb  # noqa
except Exception:
    wandb = None


RANK = int(os.environ.get("RANK", "0"))


def log0(msg: str):
    if RANK == 0:
        print(msg, flush=True)


logger = logging.get_logger(__name__)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")
os.environ.setdefault("HF_DATASETS_VERBOSITY", "info")
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


def _has_cols(ds, cols):
    try:
        return all(c in ds.features for c in cols)
    except Exception:
        return all(c in ds.column_names for c in cols)


def _preview_split(ds, split_name: str, n: int = 2):
    try:
        length = len(ds)
    except TypeError:
        length = "(streaming)"

    cols = list(getattr(ds, "column_names", [])) or list(getattr(ds, "features", {}).keys())
    log0(f"[DATA] Split '{split_name}': length={length}, columns={cols}")

    try:
        head = ds.select(range(min(n, len(ds)))) if isinstance(length, int) else ds.take(n)  # type: ignore
        maxw = 120
        log0(f"[DATA] First {n} rows (truncated):")
        for i, row in enumerate(list(head)):
            short = {
                k: (str(v)[:maxw] + ("…" if len(str(v)) > maxw else ""))
                for k, v in dict(row).items()
            }
            log0(f"        #{i}: {short}")
    except Exception as e:
        log0(f"[DATA] Preview failed for split '{split_name}': {e}")


def _sample_dataset(ds, n: Optional[int], seed: int):
    try:
        if n is None or int(n) <= 0 or int(n) >= len(ds):
            return ds
        n = int(n)
        log0(f"[SAMPLE] Shuffling and selecting first {n} / {len(ds)} examples (seed={seed})…")
        return ds.shuffle(seed=seed).select(range(n))
    except Exception as e:
        log0(f"[SAMPLE] Sampling skipped due to error: {e}")
        return ds


def _maybe_setup_wandb(training_args):
    report_to = training_args.report_to
    if report_to is None:
        return

    if isinstance(report_to, str):
        backends = [r.strip() for r in report_to.split(",") if r.strip()]
    else:
        backends = list(report_to)

    if "wandb" not in backends:
        return

    if wandb is None:
        log0("[WANDB] 'wandb' not installed, but report_to includes 'wandb'. Please `pip install wandb`.")
        return

    if RANK != 0:
        os.environ.setdefault("WANDB_MODE", "offline")
        return

    default_project = os.environ.get("WANDB_PROJECT", "sft_premia")
    os.environ.setdefault("WANDB_PROJECT", default_project)

    run_name = training_args.run_name
    if not run_name:
        od = (training_args.output_dir or "").rstrip("/ ")
        run_name = os.path.basename(od) if od else "sft_run"
    os.environ.setdefault("WANDB_NAME", run_name)

    log0(f"[WANDB] Enabled: project={os.environ.get('WANDB_PROJECT')} name={os.environ.get('WANDB_NAME')}")


def _build_dataset_paths(base_path: str, rl_filename: str):
    """
    支持三种 dataset_name 形式：
    1) /path/to/data_root
       -> rl=/path/to/data_root/rl/<rl_filename>
          eval=/path/to/data_root/evaluation/train-*.parquet

    2) /path/to/data_root/rl
       -> rl=/path/to/data_root/rl/<rl_filename>
          eval=/path/to/data_root/evaluation/train-*.parquet

    3) /path/to/data_root/evaluation
       -> rl=/path/to/data_root/rl/<rl_filename>
          eval=/path/to/data_root/evaluation/train-*.parquet
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


def main(script_args, training_args, model_args, dataset_args, extra_opts):
    log0("[SFT] Preparing model kwargs…")
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    log0("[SFT] Loading config…")
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    log0("[SFT] Loading model…")
    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    log0("[SFT] Loading dataset…")
    rl_filename = getattr(extra_opts, "rl_filename", "train-00000-of-00001.parquet")
    rl_glob, eval_glob = _build_dataset_paths(script_args.dataset_name, rl_filename)

    if not os.path.exists(rl_glob):
        raise FileNotFoundError(f"[DATA] RL parquet not found: {rl_glob}")

    eval_dir = os.path.dirname(eval_glob)
    if not os.path.exists(eval_dir):
        raise FileNotFoundError(f"[DATA] Evaluation directory not found: {eval_dir}")

    log0(f"[SFT] Loading parquet: rl={rl_glob} | evaluation={eval_glob}")
    ds_train = load_dataset("parquet", data_files={"train": rl_glob})["train"]
    ds_eval = load_dataset("parquet", data_files={"train": eval_glob})["train"]
    dataset = DatasetDict({"train": ds_train, "test": ds_eval})

    train_split = "train"
    test_split = "test"

    if train_split in dataset:
        log0("[SFT] Dataset loaded. Preview BEFORE mapping:")
        _preview_split(dataset[train_split], train_split)
        if test_split in dataset:
            _preview_split(dataset[test_split], test_split)

    need_text_map = (
        train_split in dataset
        and _has_cols(dataset[train_split], ["question", "response_j"])
        and ("text" not in getattr(dataset[train_split], "column_names", []))
    )

    if need_text_map:
        log0("[SFT] Detected preference-style columns; mapping {question + response_j} -> 'text' for SFT…")

        def to_text(ex):
            q = (ex.get("question") or "").strip()
            r = (ex.get("response_j") or "").strip()
            return {"text": (q + "\n\n" + r).strip()}

        train_remove_cols = list(getattr(dataset[train_split], "column_names", []))
        dataset[train_split] = dataset[train_split].map(
            to_text,
            remove_columns=train_remove_cols,
            desc="[MAP][train] to 'text' column",
            num_proc=max(1, (os.cpu_count() or 4) // 2),
        )

        if test_split in dataset:
            test_remove_cols = list(getattr(dataset[test_split], "column_names", []))
            dataset[test_split] = dataset[test_split].map(
                to_text,
                remove_columns=test_remove_cols,
                desc="[MAP][test] to 'text' column",
                num_proc=max(1, (os.cpu_count() or 4) // 2),
            )

        log0("[SFT] Preview AFTER mapping:")
        _preview_split(dataset[train_split], train_split)
        if test_split in dataset:
            _preview_split(dataset[test_split], test_split)
    else:
        log0("[SFT] No mapping needed (either 'text' already present or schema not matched).")
        if train_split in dataset:
            _preview_split(dataset[train_split], train_split)

    train_n = getattr(extra_opts, "train_sample_size", None)
    eval_n = getattr(extra_opts, "eval_sample_size", None)
    seed = getattr(extra_opts, "sample_seed", 42)

    if train_n is None or int(train_n) <= 0:
        try:
            n = len(dataset[train_split])
        except TypeError:
            n = None
        target = 80000
        if isinstance(n, int) and n > target:
            log0(f"[SFT] (Legacy) Shuffling train split and selecting first {target} / {n} examples (seed=42)…")
            dataset[train_split] = dataset[train_split].shuffle(seed=42).select(range(target))
    else:
        dataset[train_split] = _sample_dataset(dataset[train_split], int(train_n), seed)

    if training_args.eval_strategy != "no" and test_split in dataset:
        if eval_n is not None and int(eval_n) > 0:
            dataset[test_split] = _sample_dataset(dataset[test_split], int(eval_n), seed)

    log0(
        f"[DATA] Final sizes -> train: {len(dataset[train_split])} | "
        f"test: {len(dataset[test_split]) if test_split in dataset else 0}"
    )

    _maybe_setup_wandb(training_args)

    # 关键修正：你当前 trl 版本要把 dataset_text_field 放到 args 上，而不是 SFTTrainer(...) 参数里
    training_args.dataset_text_field = "text"

    log0(f"[DEBUG] training_args.dataset_text_field = {getattr(training_args, 'dataset_text_field', None)}")
    log0(f"[DEBUG] train columns = {dataset[train_split].column_names}")

    log0("[SFT] Initializing trainer…")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[train_split],
        eval_dataset=dataset[test_split] if training_args.eval_strategy != "no" and test_split in dataset else None,
        peft_config=get_peft_config(model_args),
    )

    resume_ckpt = getattr(training_args, "resume_from_checkpoint", None)
    if resume_ckpt:
        log0(f"[SFT] Resuming from checkpoint (training_args): {resume_ckpt}")
    else:
        log0("[SFT] No resume_from_checkpoint provided in training_args; training from scratch.")

    log0("[SFT] Calling trainer.train() …")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.accelerator.print("✅ Training completed.")

    log0("[SFT] Saving model…")
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, dataset_args, extra = parser.parse_args_and_config(
        return_remaining_strings=True
    )

    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--train_sample_size", type=int, default=None)
    extra_parser.add_argument("--eval_sample_size", type=int, default=None)
    extra_parser.add_argument("--sample_seed", type=int, default=42)
    extra_parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    extra_parser.add_argument(
        "--rl_filename",
        type=str,
        default="train-00000-of-00001.parquet",
        help="RL 训练集文件名，例如 train-00000-of-00001.parquet",
    )
    extra_opts, _ = extra_parser.parse_known_args(extra)

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

    log0("[SFT] Parsed args. Starting main()…")
    log0(
        f"[ENV] RANK={os.environ.get('RANK')} "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    main(script_args, training_args, model_args, dataset_args, extra_opts)