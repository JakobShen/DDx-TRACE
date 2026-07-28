# Run open-source models with vLLM

DDx-TRACE talks to candidate models through an OpenAI-compatible Chat
Completions endpoint. The same workflow applies to any model that:

- is supported by the installed vLLM version;
- accepts combined text and image input;
- includes a usable chat template; and
- can return the benchmark's JSON response format.

Check the official
[vLLM model list](https://docs.vllm.ai/en/latest/models/supported_models/)
before selecting a checkpoint. Text-only models cannot run the official
image-based benchmark.

## Tested example: MedGemma

MedGemma is one validated example, not a special execution path.

| Component | Tested value |
|---|---|
| Python | 3.10.20 |
| vLLM | 0.26.0 |
| GPU | 1 × NVIDIA H100 96 GB |
| Model | `google/medgemma-27b-it`, bfloat16 |

The three-case integration test completed 3/3 cases with no failed cases,
JSON parse failures, or output-format warnings.

The commands below use MedGemma for concrete variable values. Replace
`MODEL_ID`, `MODEL_DIR`, `SERVED_NAME`, `MODEL_LABEL`, and model-specific
vLLM arguments to test another compatible checkpoint.

## 1. Download code and data

Install Git LFS, `zstd`, and Conda first.

```bash
export WORK="/path/to/ddx-trace-work"
mkdir -p "$WORK"

git clone https://github.com/DDx-Trace/DDx-TRACE.git "$WORK/DDx-TRACE"
cd "$WORK/DDx-TRACE"

bash skills/run-ddx-trace-vllm/scripts/prepare_data.sh \
  --work-dir "$WORK"
```

The script downloads the dataset, verifies the image archive checksum, and
extracts the images.

## 2. Create the environments

```bash
bash skills/run-ddx-trace-vllm/scripts/create_envs.sh \
  --work-dir "$WORK" \
  --repo-dir "$WORK/DDx-TRACE"
```

This creates separate benchmark and vLLM environments and records complete
dependency manifests.

## 3. Select a model

For the tested MedGemma example:

```bash
export MODEL_ID="google/medgemma-27b-it"
export MODEL_DIR="$WORK/models/google-medgemma-27b-it"
export SERVED_NAME="medgemma-27b-it"
export MODEL_LABEL="MedGemma 27B IT"
```

Accept any gated model terms before downloading. Authenticate interactively;
do not put a Hugging Face token in a script or log.

```bash
"$WORK/envs/vllm-0.26.0-py310/bin/hf" auth login
"$WORK/envs/vllm-0.26.0-py310/bin/hf" download "$MODEL_ID" \
  --local-dir "$MODEL_DIR"
```

Skip the download when a complete local checkpoint already exists.

## 4. Start vLLM

Run this on a machine with a supported NVIDIA GPU:

```bash
bash skills/run-ddx-trace-vllm/scripts/start_vllm.sh \
  --env "$WORK/envs/vllm-0.26.0-py310" \
  --model "$MODEL_DIR" \
  --served-name "$SERVED_NAME" \
  --dtype bfloat16 \
  --cache-dir "$WORK/cache/vllm" \
  --background \
  --wait-seconds 1200
```

The script waits until the health and model endpoints are ready. For another
model, adjust context length, dtype, or other serving options. Pass additional
vLLM tokens with repeated `--vllm-arg` options when needed.

## 5. Run and validate three cases

Run the client where it can reach the vLLM endpoint:

```bash
bash skills/run-ddx-trace-vllm/scripts/run_smoke.sh \
  --repo "$WORK/DDx-TRACE" \
  --data-root "$WORK/DDx-TRACE-data" \
  --benchmark-env "$WORK/envs/ddx-benchmark-py310" \
  --target-model "$SERVED_NAME" \
  --model-label "$MODEL_LABEL" \
  --limit 3
```

The command uses the deterministic rule judge and validates
`benchmark_full.json`. Do not start a full run unless it prints:

```text
Validation passed
```

## Full benchmark and LLM judge

After the smoke test passes, use `run_main.py` with `--provider vllm` and the
same endpoint and served model name. Remove `--limit 3` for the full dataset.
To add the LLM judge:

```bash
--judge-modes both \
--judge-provider vertex \
--judge-model gemini-3-flash-preview
```

Set `VERTEX_API_KEY` through the environment or a secret manager.

## Agent workflow

Agents can follow the detailed
[`run-ddx-trace-vllm` skill](../skills/run-ddx-trace-vllm/SKILL.md). It
contains model-selection checks, staged execution, recovery rules, result
validation, cleanup, and reporting requirements. The tested MedGemma settings
are retained as an optional example rather than the default workflow.
