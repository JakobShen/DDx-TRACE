# Run DDx-TRACE with MedGemma

This is the shortest tested path for running the multimodal
`google/medgemma-27b-it` model through vLLM.

Tested configuration:

| Component | Version |
|---|---|
| Python | 3.10.20 |
| vLLM | 0.26.0 |
| GPU | 1 × NVIDIA H100 96 GB |
| Model | `google/medgemma-27b-it`, bfloat16 |

The validated smoke test completed 3/3 cases with no failed cases, JSON parse
failures, or output-format warnings.

## 1. Download code and data

Install `git-lfs`, `zstd`, and Conda first.

```bash
export WORK="/path/to/ddx-trace-work"
mkdir -p "$WORK"

git clone https://github.com/DDx-Trace/DDx-TRACE.git "$WORK/DDx-TRACE"
cd "$WORK/DDx-TRACE"

bash skills/run-ddx-trace-medgemma/scripts/prepare_data.sh \
  --work-dir "$WORK"
```

The script downloads the dataset, verifies the image archive checksum, and
extracts the images.

## 2. Create the environments

```bash
bash skills/run-ddx-trace-medgemma/scripts/create_envs.sh \
  --work-dir "$WORK" \
  --repo-dir "$WORK/DDx-TRACE"
```

This creates separate benchmark and vLLM environments and records their
complete dependency manifests.

## 3. Obtain MedGemma

Accept the model terms on Hugging Face before downloading. Authenticate
interactively; do not place a token in a script or log.

```bash
"$WORK/envs/vllm-0.26.0-py310/bin/hf" auth login
"$WORK/envs/vllm-0.26.0-py310/bin/hf" download google/medgemma-27b-it \
  --local-dir "$WORK/models/google-medgemma-27b-it"
```

Skip this step when a complete local checkpoint already exists.

## 4. Start vLLM

Run this on the GPU node. On Slurm, create a `screen` session before entering
the allocation.

```bash
bash skills/run-ddx-trace-medgemma/scripts/start_vllm.sh \
  --env "$WORK/envs/vllm-0.26.0-py310" \
  --model "$WORK/models/google-medgemma-27b-it" \
  --served-name medgemma-27b-it \
  --cache-dir "$WORK/cache/vllm" \
  --background \
  --wait-seconds 1200
```

The script redirects compiler caches away from quota-limited home storage and
waits until both the server and model endpoint are ready.

## 5. Run and validate three cases

Run this on the same node as vLLM:

```bash
bash skills/run-ddx-trace-medgemma/scripts/run_smoke.sh \
  --repo "$WORK/DDx-TRACE" \
  --data-root "$WORK/DDx-TRACE-data" \
  --benchmark-env "$WORK/envs/ddx-benchmark-py310" \
  --target-model medgemma-27b-it \
  --limit 3
```

The command runs the deterministic rule judge and automatically validates
`benchmark_full.json`. Do not start a full run unless it prints:

```text
Validation passed
```

## Full benchmark and LLM judge

After the smoke test passes, use `run_main.py` with the same paths and
`--provider vllm`. Remove `--limit 3` for the full dataset. To reproduce the
LLM-judge setting, add:

```bash
--judge-modes both \
--judge-provider vertex \
--judge-model gemini-3-flash-preview
```

Set `VERTEX_API_KEY` in the environment without printing it.

## Agent workflow and troubleshooting

Agents should follow the detailed
[`run-ddx-trace-medgemma` skill](skills/run-ddx-trace-medgemma/SKILL.md).
It includes Slurm handling, checkpoints after every stage, recovery rules,
output acceptance criteria, cleanup, and reporting requirements.
