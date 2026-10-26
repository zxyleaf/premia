#!/usr/bin/env python3
"""Build deterministic blinded inputs for the two reviewer-requested audits."""

import csv
import hashlib
import json
import random
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
PRIVATE = HERE / "private"
SEED = 20271025

GENERATION_PAIRS = [
    (
        "Open-LLaMA-3B",
        "SE/premia/utility_eval_bc/DPO_raw_generations.jsonl",
        "SE/premia/utility_eval_bc/open_llama_3b_v2_80k_k1_less_llama_32_3b_in_ba8_ga2_lora/open_llama_3b_v2_80k_k1_less_llama_32_3b_in_ba8_ga2_lora_generations.jsonl",
    ),
    (
        "Llama-3.2-1B",
        "SE/premia/utility_eval_bc/llama_32_1b/DPO_raw_generations.jsonl",
        "SE/premia/utility_eval_bc/llama_32_1b/DPO_defended_k1_llama_32_3b_in_generations.jsonl",
    ),
    (
        "Llama-3.2-3B",
        "SE/premia/utility_eval_bc/llama_32_3b/llama_32_3b_80k_ba8_ga2_lora_generations.jsonl",
        "SE/premia/utility_eval_bc/llama_32_3b/llama_32_3b_80k_k1_less_llama_32_3b_in_ba8_ga2_lora_generations.jsonl",
    ),
    (
        "Gemma-3-1B",
        "SE/premia/utility_eval_bc/rest2/gemma3_1b_80k_ba8_ga2_lora_generations.jsonl",
        "SE/premia/utility_eval_bc/open_llama_7b/gemma3_1b_80k_k1_less_llama_32_3b_in_ba8_ga2_lora_generations.jsonl",
    ),
    (
        "Qwen3.5-0.8B",
        "SE/premia/utility_eval_bc/qwen35/qwen35_08b_base80k_ba8_ga2_lora_generations.jsonl",
        "SE/premia/utility_eval_bc/qwen35more/qwen35_08b_base_80k_k1_less_llama_32_3b_in_ba8_ga2_lora_generations.jsonl",
    ),
]

GENERATION_PAIRS.extend(
    [
        (
            "Open-LLaMA-7B",
            "SE/premia/utility_eval_bc/rest2/dpo_llama_7b_dpo_se80k_batch4_2_generations.jsonl",
            "SE/premia/utility_eval_bc/open_llama_7b/llama_7b_80k_k1_less_llama_32_3b_in_ba8_ga2_lora_generations.jsonl",
        ),
        (
            "Qwen3-1.7B",
            "SE/premia/utility_eval_bc/more/qwen317b_base_80k_ba8_ga2_lora_generations.jsonl",
            "SE/premia/utility_eval_bc/more/qwen317b_base_80k_k1_less_ba8_ga2_lora_generations.jsonl",
        ),
        (
            "Gemma-2-2B",
            "SE/premia/utility_eval_bc/rest2/dpo_gemma_2b_80k_batch8_8_lora_generations.jsonl",
            "SE/premia/utility_eval_bc/gemma/gemma2_2b_80k_k1_less_llama_32_3b_in_ba8_ga2_lora_generations.jsonl",
        ),
    ]
)

PAIR_NOTES = {
    "Open-LLaMA-3B": "Historical held-out generation pair; DPO and PRP-R training used different PyTorch versions.",
    "Llama-3.2-1B": "Historical held-out generation pair with matching audited major training arguments.",
    "Llama-3.2-3B": "Historical held-out generation pair with matching audited major training arguments.",
    "Gemma-3-1B": "Historical held-out generation pair; DPO and PRP-R training used different PyTorch versions.",
    "Qwen3.5-0.8B": "Historical held-out generation pair; its original DPO generation target differs from its reward-table target.",
    "Open-LLaMA-7B": "Descriptive only: archived DPO and PRP-R checkpoints have different optimizer-update counts.",
    "Qwen3-1.7B": "Descriptive only: selected DPO and PRP-R tokenizer artifacts have different EOS IDs.",
    "Gemma-2-2B": "Descriptive only: recovered DPO and PRP-R artifacts report two versus three training epochs.",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_generation():
    public, private, sources = [], [], []
    for backbone, dpo_rel, prpr_rel in GENERATION_PAIRS:
        dpo_path, prpr_path = REPO / dpo_rel, REPO / prpr_rel
        dpo_rows, prpr_rows = load_jsonl(dpo_path), load_jsonl(prpr_path)
        if len(dpo_rows) != len(prpr_rows):
            raise ValueError(f"Count mismatch for {backbone}")
        for index, (dpo, prpr) in enumerate(zip(dpo_rows, prpr_rows)):
            if dpo["question"] != prpr["question"]:
                raise ValueError(f"Question mismatch: {backbone} row {index}")
            dpo_ref, prpr_ref = dpo.get("reference", ""), prpr.get("reference", "")
            if dpo_ref != prpr_ref:
                raise ValueError(f"Reference mismatch: {backbone} row {index}")
            item_id = hashlib.sha256(
                f"{SEED}:generation:{backbone}:{index}".encode()
            ).hexdigest()[:20]
            rng = random.Random(f"{SEED}:generation:{backbone}:{index}")
            a_source = rng.choice(("DPO", "PRP-R"))
            predictions = {"DPO": dpo["prediction"], "PRP-R": prpr["prediction"]}
            b_source = "PRP-R" if a_source == "DPO" else "DPO"
            public.append(
                {
                    "item_id": item_id,
                    "task_type": "generation_quality",
                    "backbone": backbone,
                    "prompt": dpo["question"],
                    "reference": dpo_ref,
                    "response_A": predictions[a_source],
                    "response_B": predictions[b_source],
                }
            )
            private.append(
                {
                    "item_id": item_id,
                    "backbone": backbone,
                    "source_row": index,
                    "response_A_source": a_source,
                    "response_B_source": b_source,
                }
            )
        sources.extend(
            [
                {"backbone": backbone, "method": "DPO", "path": dpo_rel, "sha256": sha256(dpo_path), "comparison_note": PAIR_NOTES[backbone]},
                {"backbone": backbone, "method": "PRP-R", "path": prpr_rel, "sha256": sha256(prpr_path), "comparison_note": PAIR_NOTES[backbone]},
            ]
        )
    rng = random.Random(f"{SEED}:generation:shuffle")
    rng.shuffle(public)
    return public, private, sources


def build_preference():
    blind_path = REPO / "paper_revision_20261003/evidence/preference_label_audit_BLIND_annotation_sheet.csv"
    key_path = REPO / "paper_revision_20261003/evidence/preference_label_audit_PRIVATE_key.csv"
    with blind_path.open(newline="", encoding="utf-8") as stream:
        visible = list(csv.DictReader(stream))
    with key_path.open(newline="", encoding="utf-8") as stream:
        hidden = list(csv.DictReader(stream))
    hidden_by_id = {row["item_id"]: row for row in hidden}
    if len(hidden_by_id) != len(hidden) or {r["item_id"] for r in visible} != set(hidden_by_id):
        raise ValueError("Preference blind sheet and private key do not align")
    public = [
        {
            "item_id": row["item_id"],
            "task_type": "preference_label_validity",
            "backbone": "sanitized-training-tuples",
            "prompt": row["prompt"],
            "reference": "",
            "response_A": row["response_A"],
            "response_B": row["response_B"],
        }
        for row in visible
    ]
    private = [hidden_by_id[row["item_id"]] for row in visible]
    return public, private, {
        "blind_sheet_path": str(blind_path.relative_to(REPO)),
        "blind_sheet_sha256": sha256(blind_path),
        "private_key_sha256": sha256(key_path),
    }


def main():
    INPUTS.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    generation, generation_key, generation_sources = build_generation()
    preference, preference_key, preference_source = build_preference()
    generation_path = INPUTS / "generation_quality_blind.jsonl"
    preference_path = INPUTS / "preference_label_validity_blind.jsonl"
    write_jsonl(generation_path, generation)
    write_jsonl(preference_path, preference)
    write_jsonl(PRIVATE / "generation_quality_PRIVATE_key.jsonl", generation_key)
    write_jsonl(PRIVATE / "preference_label_validity_PRIVATE_key.jsonl", preference_key)
    manifest = {
        "status": "inputs_only_no_llm_judgments_collected",
        "seed": SEED,
        "generation_quality": {
            "purpose": "Blind pairwise comparison of held-out DPO and PRP-R generations.",
            "n_items": len(generation),
            "n_backbones": len(GENERATION_PAIRS),
            "n_per_backbone": 500,
            "input": str(generation_path.relative_to(REPO)),
            "input_sha256": sha256(generation_path),
            "sources": generation_sources,
        },
        "preference_label_validity": {
            "purpose": "Compare chosen responses with original or sanitized rejected responses.",
            "n_items": len(preference),
            "n_source_tuples": len(preference) // 2,
            "input": str(preference_path.relative_to(REPO)),
            "input_sha256": sha256(preference_path),
            **preference_source,
        },
        "blinding": "A/B mappings and condition labels are stored under ignored private/ and are not committed.",
        "warning": "These are evaluation inputs, not observed LLM-as-judge results.",
    }
    manifest_path = HERE / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
