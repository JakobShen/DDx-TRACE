# EuroRad main-benchmark refactor

This package implements the **official main experiment** aligned to the revised paper:

- hidden inventory
- open-ended free-form exam requests
- image evidence plus minimal metadata target-model access (no captions/findings/final diagnosis revealed)
- turn-wise updated 4-item differential diagnosis with probabilities
- dual scoring modes: **LLM judge** and **deterministic rule-based judge**
- Table 2 metrics + additional main-run diagnostics

## Files

- `run_main.py` — entrypoint, provider adapters, run orchestration, compact output packaging
- `pipeline_main.py` — official main pipeline, dataset adaptation, matcher, metrics, preflight
- `judge.py` — single-call LLM judge + deterministic scorer

## Generation defaults

The benchmark now uses **provider/model default sampling**. The runner does **not** send `temperature` to any provider. Deprecated `--temperature` and `--judge-temperature` are accepted only for compatibility and are recorded as ignored.

Reasoning/thinking controls are also omitted by default. If you do not pass `--reasoning-effort` or `--gemini-thinking-level`, the request body does not include these parameters and the provider/model default behavior applies.

## Structured output

- Target agent: defaults to `--agent-structured-output auto`, which uses provider-native JSON schema when available and falls back to prompt-JSON if unsupported.
- LLM judge: defaults to `--judge-structured-output auto`, also using provider-native JSON schema with prompt-JSON fallback.

## Supported providers

Target model and LLM judge model can be selected independently.

- `openai` — Responses API with `previous_response_id`; structured output uses `text.format` JSON Schema
- `anthropic` — Messages API with local history management; structured output uses `output_config.format`
- `gemini` — `generateContent` with local history management; structured output uses `responseJsonSchema`
- `vertex` — Vertex AI Gemini through `google-genai` express mode; structured output uses `response_json_schema`
- `qwen` — DashScope router; `--qwen-transport auto|chat|responses`; structured output uses chat `response_format` JSON Schema or Responses `text.format`

### Important Qwen note

For the benchmark, local images are expected through `--image-root`. In `auto` mode, the Qwen adapter defaults to the OpenAI-compatible **chat** transport because this is the safest path for local image payloads. If you use hosted/public image URLs and want server-side response chaining, pass `--qwen-transport responses`.

## Example

```bash
python run_main.py \
  --data-path /path/to/eurorad_neuro_01.json \
  --image-root /path/to/project_root_or_images_dir \
  --out-dir /path/to/runs \
  --provider openai \
  --target-model gpt-5.4 \
  --judge-modes both \
  --reveal-unit eurorad \
  --budget 6
```

By default, fixed-horizon trajectory metrics use `T_max = budget + 2`. To override:

```bash
--trajectory-horizon 8 --diagnostic-threshold 0.6666667
```

### Rule-only preflight / pilot mode

```bash
python run_main.py \
  --data-path /path/to/eurorad_neuro_01.json \
  --image-root /path/to/project_root_or_images_dir \
  --out-dir /path/to/runs \
  --provider openai \
  --target-model gpt-5.4 \
  --judge-modes rule \
  --limit 20 \
  --budget 6
```


## Ablation studies

Non-official ablations are implemented in `ablation_main.py` so the official sequential code path remains unchanged.

Implemented settings:

- `history-only`: clinical history only; no images and no requests.
- `all-images-at-once`: all requestable EuroRad-style evidence units are attached in one call with minimal metadata.
- `random-order-reveal`: no active requests; evidence units are passively revealed one at a time in deterministic random order, up to the request budget, after a history-only baseline turn.
- `gold-order-reveal`: no active requests; evidence units are passively revealed one at a time by preferred/gold order, up to the request budget, after a history-only baseline turn.
- `oracle-findings`: active request-based sequential setting; when a request matches, the target receives the image, minimal metadata, and oracle imaging findings for the matched evidence unit.

Run all ablations:

```bash
python ablation_main.py \
  --setting all \
  --data-path /path/to/eurorad_neuro_01_fixed.json \
  --image-root /path/to/project_root_or_images_dir \
  --out-dir /path/to/ablations \
  --provider openai \
  --target-model gpt-5.4 \
  --reveal-unit eurorad \
  --budget 6
```

Run one setting:

```bash
python ablation_main.py \
  --setting oracle-findings \
  --data-path /path/to/eurorad_neuro_01_fixed.json \
  --image-root /path/to/project_root_or_images_dir \
  --out-dir /path/to/ablations \
  --provider openai \
  --target-model gpt-5.4
```

For passive ablations (`history-only`, `all-images-at-once`, `random-order-reveal`, `gold-order-reveal`), route/request metrics are not applicable and are marked undefined. For `oracle-findings`, route/request metrics remain applicable because the agent still actively requests evidence.

Judge defaults for both official and ablation entrypoints are fixed to:

```bash
--judge-provider vertex --judge-model gemini-3-flash-preview
```

## Environment variables

- OpenAI: `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`
- Anthropic: `ANTHROPIC_API_KEY`, optional `ANTHROPIC_BASE_URL`
- Gemini: `GEMINI_API_KEY` or `GOOGLE_API_KEY`, optional `GEMINI_BASE_URL`
- Vertex AI Gemini: `VERTEX_API_KEY`
- Qwen/DashScope: `DASHSCOPE_API_KEY`, optional `DASHSCOPE_BASE_URL`

## Image root handling

`--image-root` can point either to the dataset project root **or** directly to the `images/` directory.
The path resolver handles both `images/12789/f1_a.jpg`-style relative paths and paths already rooted inside `images/`.

## Main outputs

Each run writes a timestamped directory with two main result files:

- `benchmark_summary.json`
  - run config and code SHA256 hashes
  - dataset preflight
  - failures
  - aggregate metrics / Table 2 rows for every enabled scoring mode
  - target output health: parse warnings, schema fallback counts, probability renormalization counts
  - request outcome breakdown: matched, no-match, duplicate, unavailable, already-revealed, etc. Broad/tied requests that have an eligible above-threshold candidate are resolved to the best-scoring bundle rather than counted invalid.
  - judge transport summary and judge-mode agreement summary
  - metric notes and null/worst-case policy
- `benchmark_full.json`
  - everything in `benchmark_summary.json`
  - full per-case logs (turns, requests, judge outputs, metrics, debug metadata)

## Implemented Table 2 metrics

For each enabled scoring mode (`llm`, `rule`):

- `S_dx`
- `S_loc`
- `S_ddx`
- `S_ER`
- `Opt_burden`
- `Inv_req_rate`
- `S_order`
- `S_traj`
- `S_conf`

## Additional main-run outputs

- `Brier_top1`
- fixed-horizon trajectory metrics: `S_traj_dx_Tmax`, `S_traj_dx_actual`, `S_traj_conf_actual`, `S_traj_conf_Tmax`
- timing metrics: `time_to_hit4`, `time_to_diagnostic_guess`, `time_to_clinically_acceptable_diagnosis`
- optional/nonessential proxy metrics: `optional_request_rate`, `nonessential_request_rate_proxy`
- average requests / turns
- reliability bins for calibration plots
- subgroup slices by difficulty / rarity / section / area of interest
- dataset preflight
- target output health
- request outcome breakdown
- judge agreement diagnostics when both `llm` and `rule` are enabled

## Deterministic scorer design

The rule-based scorer is intentionally strict and versioned.

- structured JSON output is parsed deterministically
- diagnosis strings are normalized with spelling harmonization and filler removal
- multi-diagnosis strings are scored conservatively using the **first diagnosis concept only**
- negated/empty diagnosis strings receive no credit
- diagnosis buckets are derived from:
  - the case-specific diagnosis reference answer
  - quoted examples and case-specific accepted-term rubric prose in the 0–3 diagnosis rubric when present
  - the reference DDx set from `item.mcq.options`
- localization is scored component-wise from the structured reference answer
- trajectory labels (`E/A/U`) and 0–3 trajectory diagnosis scores are produced deterministically from the same buckets

## Benchmark-specific assumptions in code

- reference DDx set for `S_ddx` is taken from `item.mcq.options`
- default evidence unit is `eurorad`: each EuroRad `imaging_examination` / figure-protocol entry is one requestable evidence bundle. Routine MRI sequences such as T1/T2/FLAIR/STIR/DWI are not merged into a clinical study-level request.
- exams with delayed/follow-up/post-treatment metadata are **not** automatically excluded when local images exist; explicit follow-up requests can match them
- non-imaging evidence such as histopathology/photo/illustration is kept in provenance but excluded from imaging-route denominators
- exams without accessible **local** images under `--image-root` are excluded from the official pool
- unlabeled official exams are treated as `optional`
- matched reveal passes **images + minimal metadata**: source figure, modality, acquisition, view, region, contrast, and timepoint metadata
- duplicate requests count as invalid requests
- ambiguous free-form requests are not penalized when at least one candidate exceeds the match threshold: the resolver reveals the highest-scoring EuroRad evidence unit and records the event as `matched_ambiguous_*_best_effort` in debug/request traces.
- final diagnosis is the rank-1 diagnosis from the final differential list
- `S_order` uses tie-aware pairwise concordance when matched comparable pairs exist. If the case has gold order structure but no matched comparable pair, it is `1` only when all essential exams were recovered (`S_ER = 1`); otherwise it is `0`. If the case has no gold order structure, it is structurally undefined.
- primary `S_traj` is fixed-horizon top-1 diagnosis-rubric trajectory: average normalized diagnosis score over `T_max` turns, carrying the final answer forward after early stopping. Default `T_max = B + 2`.
- older confidence-weighted belief trajectory is still logged as `S_traj_conf_actual` and `S_traj_conf_Tmax`; these can be negative and are not calibration metrics.
- `time_to_hit4 = B + 2` when the exact diagnosis never appears in the four-way differential
- `time_to_diagnostic_guess = T_max + 1` if the top-1 diagnosis never reaches the threshold `tau`
- `time_to_clinically_acceptable_diagnosis = T_max + 1` if no turn satisfies both top-1 diagnosis score >= `tau` and EssentialRecall before the turn = 1
- `nonessential_request_rate_proxy` uses optional-labeled matched exams because explicit unnecessary labels are unavailable in the current data
- exported display metrics avoid bare `null`: structurally undefined metrics are shown as display `0` with explicit `defined=false` status and a raw counterpart preserved under `*_raw` fields

## Calibration and null semantics

- `p_max` is the final-turn top-1 probability after renormalizing the four reported probabilities to sum to 1.
- Reliability bins summarize calibration, not route quality. `avg_p_max` is mean claimed confidence in a bin; `avg_S_dx` is mean realized normalized diagnosis score in the same bin.
- When a reliability bin has `count = 0`, both `avg_p_max` and `avg_S_dx` are intrinsically undefined and remain `null`. They must not be imputed as `0`, because `0` would falsely imply observed low-confidence failures rather than an empty bin.
- `S_conf` is separate from calibration. It is the final-turn confidence-weighted semantic alignment score over the whole four-item differential and can be negative.

## Preflight gate

Before any API calls, the runner validates that local image files can be resolved and that the official exam pool is non-empty. Blocking preflight errors abort the run by default and write a preflight-only `benchmark_summary.json` / `benchmark_full.json`. Use `--allow-preflight-errors` only for debugging.

## Deterministic scorer version

`rule_v8_rubric_extracted_terms` keeps exact gold aliases specific: relaxed/subset gold variants are scored as near-gold/acceptable rather than exact. It extracts straight/curly quoted rubric examples plus case-specific rubric prose patterns such as “acceptable phrasing includes”, “such as”, “e.g.”, and “or equivalent”. It does not use a separate hand-built diagnosis synonym dictionary. It emits both E/A/U trajectory labels and 0–3 trajectory diagnosis scores.

## Evidence-unit convention

The default benchmark setting is intentionally close to EuroRad. A requestable evidence unit is one source `imaging_examination` / figure-protocol entry, i.e. a EuroRad figure/protocol-level evidence bundle. This is not the same as hospital order entry for a complete clinical study. The benchmark preserves EuroRad-style sequence/protocol granularity: if EuroRad separates T1, T2, DWI/ADC, SWI/GRE, CTA, or post-contrast images into distinct entries, they remain separately requestable bundles. Therefore, a request may include sequence/acquisition/contrast when useful, e.g. `T2-weighted MRI lumbosacral spine`, `DWI/ADC MRI brain`, `SWI/GRE MRI brain`, `CTA head and neck`, or `non-contrast CT head`.

Ambiguity is treated as a property of the EuroRad-style inventory, not as an agent error. If a broad or tied request has at least one eligible above-threshold candidate, the matcher reveals the best-scoring eligible bundle and logs the candidate scores plus the resolution reason, e.g. `matched_ambiguous_score_tie_best_effort`. The request is counted as matched, not invalid. Exact duplicate, already-revealed, unavailable, and no-match requests can still be invalid.

To avoid unreasonable matches, the matcher applies a simple spine-level penalty: cervical, thoracic, lumbar, and lumbosacral spine requests are treated as distinct unless the request is generic (`MRI spine`). Generic spine requests can still be resolved best-effort rather than penalized as ambiguous.

## Repository data policy

This public-release repository contains code and documentation only. Datasets, medical images, generated run outputs, API key files, service-account credentials, and machine-specific launcher scripts are intentionally excluded. Provide your own dataset JSON and image root paths when running the benchmark.

Copy `aa_api_key.env.example` to a local, untracked environment file and fill in only the providers you use. Never commit real API keys or cloud service-account JSON files.
