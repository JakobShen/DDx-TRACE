#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --work-dir PATH --repo-dir PATH [--conda PATH]"
}

work_dir=""
repo_dir=""
conda_bin="${CONDA_EXE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir) work_dir="$2"; shift 2 ;;
    --repo-dir) repo_dir="$2"; shift 2 ;;
    --conda) conda_bin="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$work_dir" && -n "$repo_dir" ]] || { usage >&2; exit 2; }
[[ -f "$repo_dir/requirements.txt" ]] || {
  echo "requirements.txt not found under $repo_dir" >&2
  exit 1
}
if [[ -z "$conda_bin" ]]; then
  conda_bin="$(command -v conda || true)"
fi
[[ -x "$conda_bin" ]] || { echo "A conda executable is required" >&2; exit 1; }

mkdir -p "$work_dir"/{envs,cache/conda-pkgs,cache/pip,cache/tmp,manifests}
work_dir="$(cd "$work_dir" && pwd)"
repo_dir="$(cd "$repo_dir" && pwd)"
export CONDA_PKGS_DIRS="$work_dir/cache/conda-pkgs"
export PIP_CACHE_DIR="$work_dir/cache/pip"
export TMPDIR="$work_dir/cache/tmp"
export PYTHONNOUSERSITE=1

benchmark_env="$work_dir/envs/ddx-benchmark-py310"
vllm_env="$work_dir/envs/vllm-0.26.0-py310"

if [[ ! -x "$benchmark_env/bin/python" ]]; then
  "$conda_bin" create -p "$benchmark_env" python=3.10 pip -y
fi
"$conda_bin" run -p "$benchmark_env" \
  python -m pip install --disable-pip-version-check -r "$repo_dir/requirements.txt"
"$conda_bin" run -p "$benchmark_env" \
  python -m pip freeze > "$work_dir/manifests/benchmark-pip-freeze.txt"

if [[ ! -x "$vllm_env/bin/python" ]]; then
  "$conda_bin" create -p "$vllm_env" python=3.10 pip -y
fi
"$conda_bin" run -p "$vllm_env" \
  python -m pip install --disable-pip-version-check "vllm==0.26.0"
"$conda_bin" run -p "$vllm_env" \
  python -m pip freeze > "$work_dir/manifests/vllm-pip-freeze.txt"

benchmark_python="$("$conda_bin" run -p "$benchmark_env" python -c 'import platform; print(platform.python_version())')"
vllm_python="$("$conda_bin" run -p "$vllm_env" python -c 'import platform; print(platform.python_version())')"
vllm_version="$("$conda_bin" run -p "$vllm_env" vllm --version)"

[[ "$benchmark_python" == 3.10.* && "$vllm_python" == 3.10.* ]] || {
  echo "Expected Python 3.10, got benchmark=$benchmark_python vllm=$vllm_python" >&2
  exit 1
}
[[ "$vllm_version" == *"0.26.0"* ]] || {
  echo "Expected vLLM 0.26.0, got $vllm_version" >&2
  exit 1
}

echo "Benchmark env: $benchmark_env (Python $benchmark_python)"
echo "vLLM env:      $vllm_env (Python $vllm_python, $vllm_version)"
echo "Manifests:     $work_dir/manifests"
