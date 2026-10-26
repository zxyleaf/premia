#!/usr/bin/env bash
set -euo pipefail

task="${1:-all}"
python_bin="${PYTHON_BIN:-python}"
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${task}" in
  generation|preference|all) ;;
  *)
    echo "Usage: $0 [generation|preference|all]" >&2
    exit 2
    ;;
esac

for name in AZURE_OPENAI_ENDPOINT AZURE_OPENAI_API_KEY AZURE_OPENAI_DEPLOYMENT; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

mkdir -p "${root_dir}/results"

run_generation() {
  "${python_bin}" "${root_dir}/run_blind_judge.py" \
    --input-jsonl "${root_dir}/inputs/generation_quality_blind.jsonl" \
    --output-jsonl "${root_dir}/results/generation_quality_raw.jsonl" \
    --summary-json "${root_dir}/results/generation_quality_blind_summary.json" \
    --resume
}

run_preference() {
  "${python_bin}" "${root_dir}/run_blind_judge.py" \
    --input-jsonl "${root_dir}/inputs/preference_label_validity_blind.jsonl" \
    --output-jsonl "${root_dir}/results/preference_label_raw.jsonl" \
    --summary-json "${root_dir}/results/preference_label_blind_summary.json" \
    --resume
}

if [[ "${task}" == "generation" || "${task}" == "all" ]]; then
  run_generation
fi
if [[ "${task}" == "preference" || "${task}" == "all" ]]; then
  run_preference
fi
