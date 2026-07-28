#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --env PATH --model PATH [--served-name NAME] [--host HOST] [--port N]"
  echo "          [--dtype VALUE] [--max-model-len N] [--max-num-seqs N]"
  echo "          [--gpu-memory-utilization F] [--vllm-arg TOKEN]..."
  echo "          [--tmp-dir PATH] [--cache-dir PATH] [--log PATH] [--pid-file PATH]"
  echo "          [--background] [--wait-seconds N] [--conda PATH]"
}

env_path=""
model_path=""
served_name=""
host="127.0.0.1"
port="8000"
dtype="auto"
max_model_len="16384"
max_num_seqs="1"
gpu_memory_utilization="0.90"
log_path=""
pid_file=""
background="false"
wait_seconds="0"
tmp_dir="/tmp/${USER:-ddx-user}/ddx-vllm"
cache_dir="$PWD/.vllm-cache"
conda_bin="${CONDA_EXE:-}"
extra_vllm_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) env_path="$2"; shift 2 ;;
    --model) model_path="$2"; shift 2 ;;
    --served-name) served_name="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --dtype) dtype="$2"; shift 2 ;;
    --max-model-len) max_model_len="$2"; shift 2 ;;
    --max-num-seqs) max_num_seqs="$2"; shift 2 ;;
    --gpu-memory-utilization) gpu_memory_utilization="$2"; shift 2 ;;
    --vllm-arg) extra_vllm_args+=("$2"); shift 2 ;;
    --tmp-dir) tmp_dir="$2"; shift 2 ;;
    --cache-dir) cache_dir="$2"; shift 2 ;;
    --log) log_path="$2"; shift 2 ;;
    --pid-file) pid_file="$2"; shift 2 ;;
    --background) background="true"; shift ;;
    --wait-seconds) wait_seconds="$2"; shift 2 ;;
    --conda) conda_bin="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$env_path" && -n "$model_path" ]] || { usage >&2; exit 2; }
[[ -x "$env_path/bin/python" ]] || { echo "Invalid environment: $env_path" >&2; exit 1; }
[[ -f "$model_path/config.json" ]] || { echo "Model config not found: $model_path" >&2; exit 1; }
model_path="$(cd "$model_path" && pwd)"
served_name="${served_name:-${model_path##*/}}"
command -v nvidia-smi >/dev/null || { echo "No NVIDIA GPU is visible" >&2; exit 1; }
if [[ -z "$conda_bin" ]]; then
  conda_bin="$(command -v conda || true)"
fi
[[ -x "$conda_bin" ]] || { echo "A conda executable is required" >&2; exit 1; }

log_path="${log_path:-$PWD/vllm-${served_name}.log}"
pid_file="${pid_file:-$PWD/vllm-${served_name}.pid}"
mkdir -p \
  "$(dirname "$log_path")" \
  "$(dirname "$pid_file")" \
  "$tmp_dir" \
  "$cache_dir" \
  "$cache_dir/triton" \
  "$cache_dir/torchinductor" \
  "$cache_dir/cuda"
tmp_dir="$(cd "$tmp_dir" && pwd)"
cache_dir="$(cd "$cache_dir" && pwd)"
if [[ "${#tmp_dir}" -gt 60 ]]; then
  echo "vLLM TMPDIR is too long for ZeroMQ IPC sockets: $tmp_dir" >&2
  echo "Use --tmp-dir with a short local path such as /tmp/$USER/ddx-vllm" >&2
  exit 1
fi

cmd=(
  "$conda_bin" run --no-capture-output -p "$env_path"
  env
  PYTHONNOUSERSITE=1
  TMPDIR="$tmp_dir"
  XDG_CACHE_HOME="$cache_dir"
  HF_HOME="$cache_dir/huggingface"
  FLASHINFER_WORKSPACE_BASE="$cache_dir"
  TRITON_CACHE_DIR="$cache_dir/triton"
  TORCHINDUCTOR_CACHE_DIR="$cache_dir/torchinductor"
  CUDA_CACHE_PATH="$cache_dir/cuda"
  VLLM_NO_USAGE_STATS=1
  vllm serve "$model_path"
  --host "$host"
  --port "$port"
  --served-model-name "$served_name"
  --dtype "$dtype"
  --gpu-memory-utilization "$gpu_memory_utilization"
  --max-model-len "$max_model_len"
  --max-num-seqs "$max_num_seqs"
)
if (( ${#extra_vllm_args[@]} > 0 )); then
  cmd+=("${extra_vllm_args[@]}")
fi

if [[ "$background" != "true" ]]; then
  exec "${cmd[@]}"
fi

nohup "${cmd[@]}" >"$log_path" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$pid_file"
echo "Started vLLM PID $server_pid; log=$log_path"

if [[ "$wait_seconds" -le 0 ]]; then
  exit 0
fi

deadline=$((SECONDS + wait_seconds))
while (( SECONDS < deadline )); do
  if curl --fail --silent "http://$host:$port/health" >/dev/null; then
    curl --fail --silent "http://$host:$port/v1/models"
    echo
    echo "vLLM is ready"
    exit 0
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "vLLM exited before becoming ready" >&2
    tail -n 100 "$log_path" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "Timed out waiting for vLLM after ${wait_seconds}s" >&2
tail -n 100 "$log_path" >&2 || true
exit 1
