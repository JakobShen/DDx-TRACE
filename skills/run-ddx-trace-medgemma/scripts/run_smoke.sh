#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --repo PATH --data-root PATH --benchmark-env PATH"
  echo "          [--target-model NAME] [--base-url URL] [--out-dir PATH]"
  echo "          [--limit N] [--max-output-tokens N] [--conda PATH]"
}

repo=""
data_root=""
benchmark_env=""
target_model="medgemma-27b-it"
base_url="http://127.0.0.1:8000/v1"
out_dir=""
limit="3"
max_output_tokens="8192"
conda_bin="${CONDA_EXE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) repo="$2"; shift 2 ;;
    --data-root) data_root="$2"; shift 2 ;;
    --benchmark-env) benchmark_env="$2"; shift 2 ;;
    --target-model) target_model="$2"; shift 2 ;;
    --base-url) base_url="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --max-output-tokens) max_output_tokens="$2"; shift 2 ;;
    --conda) conda_bin="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$repo" && -n "$data_root" && -n "$benchmark_env" ]] || {
  usage >&2
  exit 2
}
repo="$(cd "$repo" && pwd)"
data_root="$(cd "$data_root" && pwd)"
benchmark_env="$(cd "$benchmark_env" && pwd)"
data_path="$data_root/data/eurorad_neuro_01_release.json"
out_dir="${out_dir:-$repo/runs/medgemma27b_rule_smoke}"
mkdir -p "$out_dir"
out_dir="$(cd "$out_dir" && pwd)"
[[ -f "$repo/run_main.py" && -f "$data_path" && -d "$data_root/images" ]] || {
  echo "Code, manifest, or extracted images are missing" >&2
  exit 1
}
if [[ -z "$conda_bin" ]]; then
  conda_bin="$(command -v conda || true)"
fi
[[ -x "$conda_bin" ]] || { echo "A conda executable is required" >&2; exit 1; }

curl --fail --silent "${base_url%/v1}/health" >/dev/null || {
  echo "vLLM health check failed at ${base_url%/v1}/health" >&2
  exit 1
}

(
  cd "$repo"
  env PYTHONNOUSERSITE=1 "$conda_bin" run --no-capture-output -p "$benchmark_env" \
    python run_main.py \
    --data-path "$data_path" \
    --image-root "$data_root" \
    --out-dir "$out_dir" \
    --provider vllm \
    --base-url "$base_url" \
    --target-model "$target_model" \
    --model-label "MedGemma 27B IT" \
    --judge-modes rule \
    --max-output-tokens "$max_output_tokens" \
    --timeout 300 \
    --limit "$limit" \
    --continue-on-error
)

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$script_dir/validate_run.py" \
  "$out_dir" \
  --expected-cases "$limit" \
  --expected-provider vllm \
  --expected-model "$target_model"
