---
name: run-ddx-trace-medgemma
description: Set up, run, validate, resume, and report DDx-TRACE with a local multimodal MedGemma checkpoint served through vLLM. Use when an agent must download or verify the DDx-TRACE GitHub code and Hugging Face dataset, extract the image archive, create clean Python 3.10 and vLLM 0.26.0 environments, enter an LRZ/MCML Slurm allocation through screen, start or diagnose MedGemma, run a three-case integration test or full benchmark, add the Vertex LLM judge, inspect benchmark output health, or produce reproducibility evidence.
---

# Run DDx-TRACE with MedGemma

Complete the workflow as a sequence of gated stages. Do not advance merely
because a command returned zero; confirm the success conditions listed for
that stage.

Treat the three-case run as an integration test. Do not use its clinical
metrics as a model-quality result.

## Use the bundled resources

Use these scripts instead of rebuilding fragile commands:

- `scripts/prepare_data.sh`: clone or reuse code/data, pull LFS files, verify
  the archive, and extract images.
- `scripts/create_envs.sh`: create isolated benchmark and serving environments
  and write dependency manifests.
- `scripts/start_vllm.sh`: configure quota-safe caches, launch vLLM, and
  optionally wait for readiness.
- `scripts/run_smoke.sh`: run the local vLLM candidate on three cases with the
  rule judge and invoke the validator.
- `scripts/validate_run.py`: validate a completed `benchmark_full.json`.

Read references selectively:

1. Always read `references/tested-stack.md` before setup.
2. Read `references/lrz-slurm.md` before using an LRZ/MCML allocation.
3. Read `references/troubleshooting.md` when a stage fails or stalls.

## Preserve safety and provenance

- Use absolute paths for work, model, environment, data, output, and log
  locations.
- Keep code, data, environments, caches, logs, and outputs on project storage.
- Keep only the serving `TMPDIR` on a short node-local path.
- Never print, copy, commit, or embed a Hugging Face or provider token.
- Never install into the user's base Conda environment.
- Never run the GPU server on a login node.
- Never weaken preflight with `--allow-preflight-errors` for a reported run.
- Never reuse an output directory for a rerun.
- Preserve existing user changes. Do not pull, reset, or overwrite a dirty
  repository without explicit authorization.

## Establish paths and inputs

Resolve these values before changing state:

```bash
WORK=/absolute/path/to/ddx-trace-work
REPO="$WORK/DDx-TRACE"
DATA="$WORK/DDx-TRACE-data"
MODEL=/absolute/path/to/google-medgemma-27b-it
BENCH_ENV="$WORK/envs/ddx-benchmark-py310"
VLLM_ENV="$WORK/envs/vllm-0.26.0-py310"
SKILL_DIR=/absolute/path/to/run-ddx-trace-medgemma
```

Also establish:

- whether the model is already local;
- whether the user accepted the gated MedGemma terms;
- whether Git LFS, `zstd`/`unzstd`, Conda, and `curl` are available;
- whether execution is local GPU or Slurm;
- the Slurm job ID, login-node hostname, and desired `screen` name when
  applicable.

Resolve `SKILL_DIR` to the directory containing this `SKILL.md`. It may be
inside `$REPO/skills/` or installed independently in the agent's skill
directory. Use its bundled scripts even before `$REPO` exists.

If information can be discovered read-only, inspect it instead of asking.
Request user input only when a missing credential, gated-model acceptance, or
allocation choice blocks progress.

## Stage 1: prepare code and data

From an existing DDx-TRACE clone, run:

```bash
bash "$SKILL_DIR/scripts/prepare_data.sh" \
  --work-dir "$WORK"
```

The script is safe to rerun when `$REPO` and `$DATA` are Git repositories. It
refuses to replace unrelated non-Git paths.

Require all of the following:

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

Record both commits. Verify that the script reported a successful SHA256 check
and a nonzero image count. The tested release contains 191 cases and 6,802
extracted image files; treat different counts as a reason to inspect the
selected dataset revision, not automatically as corruption.

Do not point `--image-root` at the archive or the `data/` directory. Use the
dataset root (`$DATA`) or `$DATA/images`.

If the archive checksum fails, stop. Run `git -C "$DATA" lfs pull` again and
verify that the archive is a real LFS object rather than a pointer file.

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
environment invalid and recreate or reinstall it with user-site packages
disabled.

If the system Python lacks `pip` or `ensurepip`, do not repair it. Continue
with the Conda environments created by the script.

## Stage 3: verify the model checkpoint

Use a multimodal instruction-tuned checkpoint. Do not use
`medgemma-27b-text-it`.

When downloading:

```bash
"$VLLM_ENV/bin/hf" auth login
"$VLLM_ENV/bin/hf" download google/medgemma-27b-it --local-dir "$MODEL"
```

Authenticate interactively. Do not pass the token on the command line.

Require:

```bash
test -f "$MODEL/config.json"
test -f "$MODEL/tokenizer_config.json"
find "$MODEL" -name '*.safetensors' -type f | head
```

Inspect `config.json` and confirm a multimodal Gemma 3 conditional-generation
architecture. Check that all referenced weight shards are present before
occupying a GPU.

For the tested 27B checkpoint, plan for approximately 52 GB of weights and a
96 GB GPU when using a 16,384-token context and a 0.90 memory-utilization
target. Use a multimodal 4B MedGemma checkpoint when the available GPU cannot
hold the 27B configuration. When changing models, keep the vLLM
`--served-name`, smoke `--target-model`, and reporting label consistent.

## Stage 4: enter a durable GPU session

On Slurm, create `screen` before entering the compute node:

```bash
hostname
screen -S ddx-medgemma
squeue -j "$JOB_ID"
srun --jobid="$JOB_ID" --overlap --pty bash -l
hostname
nvidia-smi
```

Record the login-node hostname because `screen` sockets are node-local and a
load-balanced login hostname may reconnect elsewhere.

Require:

- the shell is inside `screen`;
- `hostname` is a compute node, not a login node;
- the expected job is active;
- `nvidia-smi` shows the intended GPU;
- the GPU has enough free memory;
- `$WORK`, `$REPO`, `$DATA`, and `$MODEL` are visible from the compute node.

Detach with `Ctrl-a d`; reattach on the same login node with
`screen -r ddx-medgemma`.

For a non-Slurm machine, use another durable terminal and perform the same GPU
and filesystem checks.

## Stage 5: start vLLM

Run on the compute node:

```bash
cd "$WORK"

bash "$SKILL_DIR/scripts/start_vllm.sh" \
  --env "$VLLM_ENV" \
  --model "$MODEL" \
  --served-name medgemma-27b-it \
  --cache-dir "$WORK/cache/vllm" \
  --tmp-dir "/tmp/$USER/ddx-vllm" \
  --log "$WORK/vllm-medgemma.log" \
  --pid-file "$WORK/vllm-medgemma.pid" \
  --background \
  --wait-seconds 1200
```

Allow several minutes for a 27B cold start. The script:

- isolates user-site packages;
- keeps ZeroMQ IPC under a short node-local `TMPDIR`;
- redirects vLLM, Hugging Face, FlashInfer, Triton, TorchInductor, and CUDA
  caches to project storage;
- disables vLLM usage-stat writes;
- serves bfloat16 with a 16,384-token maximum context;
- limits concurrent sequences to reduce integration-test memory pressure.

Watch progress when needed:

```bash
tail -f "$WORK/vllm-medgemma.log"
```

Require both endpoints:

```bash
curl --fail --silent http://127.0.0.1:8000/health
curl --fail --silent http://127.0.0.1:8000/v1/models
```

Inspect `/v1/models` and require an `id` exactly equal to
`medgemma-27b-it`. Do not begin the benchmark after only seeing loaded weight
shards; wait for `Application startup complete` and a successful health check.

Useful healthy log milestones include:

- resolved `Gemma3ForConditionalGeneration`;
- all checkpoint shards loaded;
- model loading completed;
- KV cache created;
- CUDA graph capture completed;
- `Application startup complete`.

If startup fails, preserve the complete log. Fix cache paths, IPC length, or
memory configuration and restart from this stage; do not redownload data or
recreate working environments unnecessarily.

## Stage 6: run three real cases

Run the client on the same compute node as the `127.0.0.1` server:

```bash
SMOKE_OUT="$REPO/runs/medgemma-rule-smoke-$(date +%Y%m%d-%H%M%S)"

bash "$SKILL_DIR/scripts/run_smoke.sh" \
  --repo "$REPO" \
  --data-root "$DATA" \
  --benchmark-env "$BENCH_ENV" \
  --target-model medgemma-27b-it \
  --base-url http://127.0.0.1:8000/v1 \
  --out-dir "$SMOKE_OUT" \
  --limit 3
```

The script first checks server health, then uses:

- `--provider vllm`;
- OpenAI-compatible multimodal chat;
- native JSON Schema in `auto` mode when supported;
- the deterministic rule judge;
- `--continue-on-error` so the result records per-case failures;
- a 300-second request timeout;
- automatic final validation.

Expect one `[OK] case_id=...` line per completed case. Monitor the vLLM log for
HTTP 200 chat-completion requests and active generation. Do not infer failure
from an initially quiet benchmark log; Python output may be buffered while
the model is generating.

## Stage 7: validate the smoke result

The smoke script invokes the validator automatically. Run it explicitly when
reviewing an existing output:

```bash
python3 "$SKILL_DIR/scripts/validate_run.py" \
  "$SMOKE_OUT" \
  --expected-cases 3 \
  --expected-provider vllm \
  --expected-model medgemma-27b-it
```

Require every core check:

| Field | Required value |
|---|---|
| `run_progress.is_final` | `true` |
| `run_progress.attempted_cases` | `3` |
| `run_progress.completed_cases` | `3` |
| `run_progress.failed_cases` | `0` |
| `run_config.provider` | `vllm` |
| `run_config.target_model` | `medgemma-27b-it` |
| `run_health.has_blocking_preflight_error` | `false` |
| `target_output_health.json_parse_failed_turns` | `0` |

Also inspect:

- `target_output_health.parse_warning_turns`;
- `probability_renormalized_turns`;
- `trimmed_to_top4_turns`;
- `uniform_probabilities_used_turns`;
- `agent_structured_output_method_counts`;
- dataset preflight image counts;
- the `failures` array;
- server-side non-200 responses.

Treat isolated normalization warnings as integration evidence to report, not
automatically as a failed run. Treat frequent warnings, fallback, padding, or
trimming as an output-contract problem that must be diagnosed before a full
run.

Require the literal final line `Validation passed`.

## Stage 8: diagnose and resume

Read `references/troubleshooting.md` and resume from the earliest failed gate:

- Missing archive or images: resume at Stage 1.
- Wrong Python, polluted imports, or wrong vLLM: resume at Stage 2.
- Incomplete or text-only model: resume at Stage 3.
- Lost allocation or wrong login node: resume at Stage 4.
- Server load, cache, IPC, or OOM failure: resume at Stage 5.
- HTTP, model-name, image, or JSON-schema failure: resume at Stage 6.
- Missing or inconsistent benchmark metadata: resume at Stage 7.

Never discard a useful environment or downloaded checkpoint merely because a
later stage failed. Use a new server log and a new run output directory for
each retry so evidence remains attributable.

If the server rejects `response_format`, retry the candidate with
`--agent-structured-output never` and report that native schema was not used.
Do not silently change this policy.

## Stage 9: add the LLM judge or full dataset

Advance only after the rule-only smoke passes.

For the Vertex judge:

1. Export `VERTEX_API_KEY` without printing it.
2. Use `--judge-modes both`.
3. Use `--judge-provider vertex`.
4. Use `--judge-model gemini-3-flash-preview`.
5. Set `--judge-max-output-tokens 8192`.
6. Use a fresh output directory.

For a full run, remove `--limit 3`. Preserve `--provider vllm`,
`--base-url http://127.0.0.1:8000/v1`, and the exact served model name.

In addition to smoke checks, require
`judge_transport.n_llm_success_cases` to equal completed cases when LLM
scoring is enabled. Record judge failures separately from target-model
failures.

Do not claim paper reproduction from a three-case rule-only smoke. State
clearly whether a result is:

- infrastructure smoke;
- full rule-only run;
- full LLM-judge run;
- full dual-judge run.

## Stage 10: preserve evidence and clean up

Preserve:

- code and dataset commits;
- benchmark and vLLM dependency manifests;
- GPU, driver, Python, vLLM, torch, transformers, and CUDA versions;
- model path and served name;
- vLLM log;
- `benchmark_summary.json`;
- `benchmark_full.json`;
- case failures and output-health summary;
- Slurm job, login node, compute node, and `screen` name when applicable.

After validation, stop only the verified vLLM target:

1. Inspect `ps` for the exact `vllm serve` model command.
2. Send `SIGTERM` to that API-server process.
3. Wait for the EngineCore child to exit.
4. Confirm `nvidia-smi` no longer lists the process.

Do not use an unscoped `pkill` on a shared compute node. Preserve the `screen`
session when the user may want to inspect logs or continue with a full run.

## Report completion

Return a compact evidence-based report containing:

- code and dataset paths and commits;
- exact model and dependency versions;
- GPU and compute node;
- server readiness result;
- case IDs, attempted/completed/failed counts;
- structured-output method and parse/normalization health;
- exact result and server-log paths;
- deviations, retries, and warnings;
- whether vLLM was stopped and GPU memory released;
- whether changes were committed or pushed.

Do not expose credentials or overinterpret three-case metrics. If a required
gate did not pass, name the failed stage and the preserved evidence instead of
calling the reproduction successful.
