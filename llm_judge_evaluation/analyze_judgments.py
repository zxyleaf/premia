#!/usr/bin/env python3
"""Reveal blinded judgments locally and compute reviewer-facing statistics."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


STRATUM_POPULATION = {
    "fallback": 4413,
    "zero_edits_nonfallback": 3640,
    "one_edit": 47983,
    "two_edits": 15197,
    "three_plus_edits": 8767,
}


def load_rows(path):
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def percentile_ci(values, seed, reps=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(reps)]
    )
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def analyze_generation(results, keys, seed):
    key_by_id = {row["item_id"]: row for row in keys}
    grouped = defaultdict(list)
    excluded = defaultdict(Counter)
    for result in results:
        key = key_by_id[result["item_id"]]
        verdict = result["final_verdict"]
        if verdict in ("invalid", "inconsistent"):
            excluded[key["backbone"]][verdict] += 1
            continue
        if verdict == "tie":
            method, value = "tie", 0.5
        else:
            method = key[f"response_{verdict}_source"]
            value = 1.0 if method == "PRP-R" else 0.0
        grouped[key["backbone"]].append((method, value))
    table = {}
    for backbone, rows in sorted(grouped.items()):
        counts = Counter(method for method, _ in rows)
        values = [value for _, value in rows]
        table[backbone] = {
            "usable": len(values),
            "prpr_wins": counts["PRP-R"],
            "ties": counts["tie"],
            "dpo_wins": counts["DPO"],
            "position_inconsistent": excluded[backbone]["inconsistent"],
            "invalid": excluded[backbone]["invalid"],
            "tie_adjusted_prpr_score": float(np.mean(values)),
            "prompt_bootstrap_95pct_ci": percentile_ci(values, seed),
        }
    return {"task": "generation_quality", "by_backbone": table}


def analyze_preference(results, keys, seed):
    key_by_id = {row["item_id"]: row for row in keys}
    by_source = defaultdict(dict)
    excluded = Counter()
    for result in results:
        key = key_by_id[result["item_id"]]
        verdict = result["final_verdict"]
        if verdict in ("invalid", "inconsistent"):
            excluded[verdict] += 1
            continue
        if verdict == "tie":
            value = 0.5
        else:
            chosen_side = "A" if int(key["chosen_on_A"]) else "B"
            value = 1.0 if verdict == chosen_side else 0.0
        by_source[key["source_row"]][key["condition"]] = {
            "value": value,
            "stratum": key["stratum"],
        }
    complete = [pair for pair in by_source.values() if set(pair) == {"original", "transformed"}]
    strata = defaultdict(list)
    for pair in complete:
        strata[pair["original"]["stratum"]].append(
            (pair["original"]["value"], pair["transformed"]["value"])
        )
    total_population = sum(STRATUM_POPULATION.values())
    weights = {name: count / total_population for name, count in STRATUM_POPULATION.items()}

    def weighted_stat(sampled):
        original = sum(weights[name] * np.mean([x[0] for x in rows]) for name, rows in sampled.items())
        transformed = sum(weights[name] * np.mean([x[1] for x in rows]) for name, rows in sampled.items())
        return original, transformed, transformed - original

    point = weighted_stat(strata)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(10000):
        sampled = {}
        for name, rows in strata.items():
            indices = rng.integers(0, len(rows), len(rows))
            sampled[name] = [rows[i] for i in indices]
        bootstrap.append(weighted_stat(sampled))
    intervals = np.quantile(np.asarray(bootstrap), [0.025, 0.975], axis=0)
    return {
        "task": "preference_label_validity",
        "complete_source_pairs": len(complete),
        "position_inconsistent_items": excluded["inconsistent"],
        "invalid_items": excluded["invalid"],
        "population_weighted_original_chosen_score": point[0],
        "population_weighted_transformed_chosen_score": point[1],
        "transformed_minus_original": point[2],
        "paired_stratified_bootstrap_95pct_ci": {
            "original": intervals[:, 0].tolist(),
            "transformed": intervals[:, 1].tolist(),
            "difference": intervals[:, 2].tolist(),
        },
        "note": "Scores count chosen wins as 1, ties as 0.5, and rejected wins as 0; strata are reweighted to the 80k source distribution.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("generation", "preference"), required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20271025)
    args = parser.parse_args()
    results, keys = load_rows(args.results), load_rows(args.private_key)
    report = (
        analyze_generation(results, keys, args.seed)
        if args.task == "generation"
        else analyze_preference(results, keys, args.seed)
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
