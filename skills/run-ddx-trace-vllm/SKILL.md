---
name: run-ddx-trace-vllm
description: Set up, run, validate, resume, and report DDx-TRACE with an open-source multimodal model served through vLLM's OpenAI-compatible API. Use when an agent must prepare the DDx-TRACE code and dataset, create Python 3.10 and vLLM 0.26.0 environments, select or download a compatible vision-language checkpoint, start or diagnose a local vLLM server, run a three-case integration test or full benchmark, add the LLM judge, validate outputs, or reproduce the tested MedGemma example.
---

# Run DDx-TRACE with vLLM

Execute the workflow as gated stages. Confirm each stage's success conditions
before advancing.

Treat the three-case run as an integration test. Do not report its clinical
metrics as a model-quality result.

## Use the bundled resources

Use these scripts rather than rebuilding the commands:

- `scripts/prepare_data.sh`: clone or reuse code and data, pull LFS files,
  verify the archive, and extract images.
- `scripts/create_envs.sh`: create isolated benchmark and serving environments
  and write dependency manifests.
- `scripts/start_vllm.sh`: configure caches, launch a selected model through
  vLLM, and optionally wait for readiness.
- `scripts/run_smoke.sh`: run three cases against a vLLM endpoint with the
  rule judge and invoke the validator.
- `scripts/validate_run.py`: validate a completed `benchmark_full.json`.

Read references selectively:

- Read `references/medgemma-example.md` when reproducing the tested MedGemma
  configuration or when a known-good baseline is needed.
- Read `references/troubleshooting.md` when a stage fails or stalls.

## Preserve safety and provenance

- Use absolute paths for work, model, environment, data, output, and log
  locations.
- Keep generated environments, caches, logs, and outputs outside the Git
  repository when practical.
- Keep the serving `TMPDIR` path short enough for ZeroMQ IPC sockets.
- Never print, copy, commit, or embed a Hugging Face or provider token.
- Never install into the user's base Conda environment.
- Never weaken preflight with `--allow-preflight-errors` for a reported run.
- Never reuse an output directory for a rerun.
- Preserve existing user changes. Do not reset or overwrite a dirty
  repository without explicit authorization.

## Establish paths and model parameters

Resolve these values before changing state:

```bash
WORK=/absolute/path/to/ddx-trace-work
REPO="$WORK/DDx-TRACE"
DATA="$WORK/DDx-TRACE-data"
MODEL=/absolute/path/to/model-checkpoint
MODEL_ID=organization/model-name
SERVED_NAME=model-name
MODEL_LABEL="Human-readable model name"
BENCH_ENV="$WORK/envs/ddx-benchmark-py310"
VLLM_ENV="$WORK/envs/vllm-0.26.0-py310"
SKILL_DIR=/absolute/path/to/run-ddx-trace-vllm
BASE_URL=http://127.0.0.1:8000/v1
```

Also establish:

- whether the checkpoint is already local;
- whether its repository is gated and terms have been accepted;
- whether Git LFS, `zstd`/`unzstd`, Conda, `curl`, and an NVIDIA GPU are
  available;
- any model-specific vLLM arguments;
- whether the benchmark client and vLLM server use the same machine or a
  reachable network endpoint.

Resolve `SKILL_DIR` to the directory containing this `SKILL.md`. It may be
inside `$REPO/skills/` or installed independently. Use its bundled scripts
even before `$REPO` exists.

Discover read-only information rather than asking for it. Request user input
only when a missing credential, gated-model acceptance, model choice, or
network choice blocks progress.

## Stage 1: prepare code and data

Run:

```bash
bash "$SKILL_DIR/scripts/prepare_data.sh" \
  --work-dir "$WORK"
```

The script is safe to rerun when `$REPO` and `$DATA` are Git repositories. It
refuses to replace unrelated non-Git paths.

Require:

```bash
test -d "$REPO/.git"
test -d "$DATA/.git"
test -f "$DATA/data/eurorad_neuro_01_release.json"
test -f "$DATA/data/DDx-TRACE-images.tar.zst"
test -d "$DATA/images"
find "$DATA/images" -type f | head
git -C "$REPO" rev-parse HEAD
git -C "$DATA" rev-parse HEAD
```

Record both commits. Require a successful SHA256 check and a nonzero image
count. The tested release contains 191 cases and 6,802 extracted image files;
inspect a different count rather than automatically treating it as corrupt.

Use the dataset root (`$DATA`) or `$DATA/images` for `--image-root`. Do not
point it at the archive or `data/` directory.

If the checksum fails, stop. Run `git -C "$DATA" lfs pull` again and confirm
the archive is not an LFS pointer file.

## Stage 2: create isolated environments

Run:

```bash
bash "$SKILL_DIR/scripts/create_envs.sh" \
  --work-dir "$WORK" \
  --repo-dir "$REPO"
```

Require:

- `$BENCH_ENV/bin/python` exists;
- `$VLLM_ENV/bin/python` exists;
- both Python versions start with `3.10`;
- `vllm --version` reports exactly `0.26.0`;
- `manifests/benchmark-pip-freeze.txt` exists;
- `manifests/vllm-pip-freeze.txt` exists.

Verify independently:

```bash
conda run -p "$BENCH_ENV" python --version
conda run -p "$VLLM_ENV" python --version
conda run -p "$VLLM_ENV" vllm --version
```

Keep `PYTHONNOUSERSITE=1`. If imports resolve from `~/.local`, consider the
environment invalid and reinstall with user-site packages disabled.

## Stage 3: select and verify a model

Use a checkpoint that satisfies all of the following:

- multimodal text-plus-image generation;
- instruction or chat tuning;
- a chat template usable by vLLM's Chat Completions API;
- an architecture supported by vLLM 0.26.0;
- enough available GPU memory for the chosen dtype, context, and concurrency.

Do not use a text-only checkpoint for the official image-based benchmark.
Check the model architecture against the vLLM supported-model documentation
and treat actual server startup plus a multimodal request as the final
compatibility test.

When downloading from Hugging Face:

```bash
"$VLLM_ENV/bin/hf" auth login
"$VLLM_ENV/bin/hf" download "$MODEL_ID" --local-dir "$MODEL"
```

Authenticate interactively. Do not pass the token on the command line.

Require:

```bash
test -f "$MODEL/config.json"
find "$MODEL" -name '*.safetensors' -type f | head
```

Confirm all referenced weight shards exist. Check for tokenizer files and a
chat template. Select a stable `SERVED_NAME`; the same exact value must be
used by vLLM, the benchmark client, the validator, and the report.

Record model-specific serving changes such as dtype, context length,
tensor-parallel size, trust-remote-code, or multimodal limits. Do not silently
apply settings copied from another architecture.

For the tested MedGemma configuration, read
`references/medgemma-example.md`. Keep it framed as one example of this generic
workflow.

## Stage 4: verify GPU availability

Run on the machine that will host vLLM:

```bash
nvidia-smi
```

Require:

- the selected GPU is visible;
- it has enough free memory for the selected model;
- `$MODEL` and the vLLM environment are readable.

Use the long-running process mechanism appropriate for the user's platform.
This skill assumes no scheduler, cloud provider, or institution.

## Stage 5: start vLLM

Run:

```bash
bash "$SKILL_DIR/scripts/start_vllm.sh" \
  --env "$VLLM_ENV" \
  --model "$MODEL" \
  --served-name "$SERVED_NAME" \
  --dtype auto \
  --cache-dir "$WORK/cache/vllm" \
  --tmp-dir "/tmp/${USER:-ddx-user}/ddx-vllm" \
  --log "$WORK/vllm-$SERVED_NAME.log" \
  --pid-file "$WORK/vllm-$SERVED_NAME.pid" \
  --background \
  --wait-seconds 1200
```

Adjust `--dtype`, `--max-model-len`, `--gpu-memory-utilization`, and
`--max-num-seqs` for the model and hardware. Pass additional vLLM command
tokens with repeated `--vllm-arg` options. Keep each flag and value as a
separate repeated argument; do not construct an `eval` string.

The script:

- isolates user-site packages;
- keeps ZeroMQ IPC under a short local `TMPDIR`;
- redirects vLLM and compiler caches to the selected cache directory;
- disables vLLM usage-stat writes;
- records the API-server PID when launched in the background.

Watch progress when needed:

```bash
tail -f "$WORK/vllm-$SERVED_NAME.log"
```

Require both endpoints:

```bash
curl --fail --silent "${BASE_URL%/v1}/health"
curl --fail --silent "${BASE_URL%/v1}/models"
```

Inspect `/v1/models` and require an `id` exactly equal to `$SERVED_NAME`.
Wait for application startup and a successful health check rather than only
loaded checkpoint shards.

If startup fails, preserve the complete log. Correct compatibility, cache,
IPC, or memory settings and retry from this stage without redownloading data.

## Stage 6: run three real cases

Run:

```bash
SMOKE_OUT="$REPO/runs/vllm-rule-smoke-$(date +%Y%m%d-%H%M%S)"

bash "$SKILL_DIR/scripts/run_smoke.sh" \
  --repo "$REPO" \
  --data-root "$DATA" \
  --benchmark-env "$BENCH_ENV" \
  --target-model "$SERVED_NAME" \
  --model-label "$MODEL_LABEL" \
  --base-url "$BASE_URL" \
  --out-dir "$SMOKE_OUT" \
  --limit 3
```

The script checks server health and then uses:

- `--provider vllm`;
- OpenAI-compatible multimodal chat;
- native JSON Schema in `auto` mode when supported;
- the deterministic rule judge;
- `--continue-on-error`;
- a 300-second request timeout;
- automatic final validation.

Expect one `[OK] case_id=...` line per completed case. Monitor the server log
for HTTP 200 chat-completion requests and active generation.

## Stage 7: validate the smoke result

The smoke script invokes the validator. Run it explicitly when reviewing an
existing output:

```bash
python3 "$SKILL_DIR/scripts/validate_run.py" \
  "$SMOKE_OUT" \
  --expected-cases 3 \
  --expected-provider vllm \
  --expected-model "$SERVED_NAME"
```

Require:

| Field | Required value |
|---|---|
| `run_progress.is_final` | `true` |
| `run_progress.attempted_cases` | `3` |
| `run_progress.completed_cases` | `3` |
| `run_progress.failed_cases` | `0` |
| `run_config.provider` | `vllm` |
| `run_config.target_model` | `$SERVED_NAME` |
| `run_health.has_blocking_preflight_error` | `false` |
| `target_output_health.json_parse_failed_turns` | `0` |

Also inspect parse warnings, probability normalization, structured-output
method counts, dataset preflight image counts, failures, and server-side
non-200 responses.

Treat isolated normalization warnings as evidence to report. Treat frequent
fallback, padding, trimming, or normalization as an output-contract problem
to diagnose before a full run.

Require the literal final line `Validation passed`.

## Stage 8: diagnose and resume

Read `references/troubleshooting.md` and resume from the earliest failed gate:

- Missing archive or images: Stage 1.
- Wrong Python, polluted imports, or wrong vLLM: Stage 2.
- Unsupported, incomplete, or text-only model: Stage 3.
- GPU unavailable or model path inaccessible: Stage 4.
- Server compatibility, load, cache, IPC, or OOM failure: Stage 5.
- HTTP, served-name, image, or JSON-schema failure: Stage 6.
- Missing or inconsistent benchmark metadata: Stage 7.

Keep useful environments and downloaded checkpoints when later stages fail.
Use a new server log and output directory for each retry.

If the server rejects `response_format`, retry with
`--agent-structured-output never` and report that native schema was not used.
Do not silently change this policy.

## Stage 9: run the full dataset or add the LLM judge

Advance only after the rule-only smoke test passes.

For the Vertex judge:

1. Export `VERTEX_API_KEY` without printing it.
2. Use `--judge-modes both`.
3. Use `--judge-provider vertex`.
4. Use `--judge-model gemini-3-flash-preview`.
5. Set `--judge-max-output-tokens 8192`.
6. Use a fresh output directory.

For a full run, remove `--limit 3`. Preserve `--provider vllm`, the reachable
`--base-url`, and the exact served model name.

When LLM scoring is enabled, require
`judge_transport.n_llm_success_cases` to equal completed cases. Record judge
failures separately from target-model failures.

Label the result accurately as one of:

- infrastructure smoke;
- full rule-only run;
- full LLM-judge run;
- full dual-judge run.

## Stage 10: preserve evidence and clean up

Preserve:

- code and dataset commits;
- benchmark and vLLM dependency manifests;
- GPU, driver, Python, vLLM, torch, transformers, and CUDA versions;
- model identifier, checkpoint revision when known, served name, and label;
- complete vLLM command and model-specific arguments;
- vLLM log;
- `benchmark_summary.json`;
- `benchmark_full.json`;
- case failures and output-health summary.

After validation, stop only the verified vLLM target:

1. Read the recorded PID.
2. Inspect the exact `vllm serve` command for that PID.
3. Send `SIGTERM` to the verified API-server process.
4. Wait for child processes to exit.
5. Confirm the GPU no longer lists the process.

Never use an unscoped `pkill`.

## Report completion

Return a compact evidence-based report containing:

- code and dataset paths and commits;
- model identifier, revision, served name, label, and dependency versions;
- GPU model and driver;
- server readiness result;
- case IDs and attempted/completed/failed counts;
- structured-output and parse/normalization health;
- result and server-log paths;
- model-specific arguments, deviations, retries, and warnings;
- whether vLLM was stopped and GPU memory released;
- whether changes were committed or pushed.

Do not expose credentials or overinterpret three-case metrics. If a required
gate failed, name the failed stage and preserved evidence rather than calling
the reproduction successful.
