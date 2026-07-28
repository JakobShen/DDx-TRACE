# DDx-TRACE benchmark reference

This document contains implementation details omitted from the quick-start
README.

## Main experiment

The official sequential setting uses:

- a hidden evidence inventory;
- open-ended radiology exam requests;
- image evidence with minimal metadata and no findings or diagnosis leakage;
- a four-item differential diagnosis with probabilities after every turn;
- deterministic rule scoring, LLM-judge scoring, or both;
- a default request budget of 6 and trajectory horizon of `budget + 2`.

`run_main.py` is the main entrypoint. `ddx_trace/pipeline.py` implements
dataset adaptation, evidence matching, preflight, and metrics.
`ddx_trace/judge.py` implements
the LLM and deterministic judges.

## Provider behavior

Target and judge providers can be selected independently:

| Provider | Transport |
|---|---|
| `openai` | Responses API |
| `anthropic` | Messages API |
| `gemini` | Gemini `generateContent` |
| `vertex` | Vertex Gemini through `google-genai` |
| `qwen` | DashScope or configurable OpenAI-compatible transport |
| `vllm` | Local OpenAI-compatible `/v1/chat/completions` |

Environment variables:

- OpenAI: `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`
- Anthropic: `ANTHROPIC_API_KEY`, optional `ANTHROPIC_BASE_URL`
- Gemini: `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- Vertex: `VERTEX_API_KEY`
- Qwen: `DASHSCOPE_API_KEY`, optional `DASHSCOPE_BASE_URL`
- vLLM: optional `VLLM_API_KEY` and `VLLM_BASE_URL`

Sampling uses provider defaults. The runner does not send `temperature`.
Reasoning controls are omitted unless explicitly supplied.

Structured output defaults to `auto`: try provider-native JSON Schema first,
then fall back to prompt-only JSON when unsupported.

## Evidence units and matching

The default reveal unit is one EuroRad `imaging_examination` or
figure-protocol entry. Sequence- or protocol-level entries remain separately
requestable when EuroRad separates them.

The matcher:

- accepts free-form requests;
- reveals the best eligible above-threshold bundle for broad or tied requests;
- records candidate scores and the resolution reason;
- counts duplicate, unavailable, already-revealed, and no-match requests
  separately;
- distinguishes cervical, thoracic, lumbar, and lumbosacral requests unless
  the request is generic.

Matched evidence contains images plus minimal source, modality, acquisition,
view, region, contrast, and timepoint metadata. Findings and final diagnosis
are not exposed in the main setting.

Non-imaging evidence remains in provenance but is excluded from imaging-route
denominators. Evidence without accessible local images is excluded from the
official pool. Follow-up evidence is not automatically excluded when its local
images exist.

## Image paths and preflight

`--image-root` may point to the dataset root or directly to `images/`. Before
any model request, preflight verifies that local images resolve and the
official evidence pool is non-empty.

Blocking errors abort by default and still write preflight-only result files.
Use `--allow-preflight-errors` only for debugging, never for a reported
reproduction.

## Main metrics

For each enabled scoring mode, the Table 2 row contains:

- `S_dx`: final diagnosis score;
- `S_loc`: localization score;
- `S_ddx`: differential diagnosis score;
- `S_ER`: essential evidence recall;
- `Opt_burden`: optional investigation burden;
- `Inv_req_rate`: invalid request rate;
- `S_order`: evidence-order score;
- `S_traj`: fixed-horizon diagnosis trajectory;
- `S_conf`: confidence-weighted semantic alignment.

Additional outputs include Brier score, timing metrics, request/turn counts,
reliability bins, subgroup slices, request outcomes, target-output health, and
judge agreement.

Structurally undefined display metrics are accompanied by an explicit
`defined=false` status and a raw counterpart. Empty calibration bins remain
`null`; they are not imputed as zero.

## Deterministic judge

The deterministic judge:

- parses structured JSON;
- normalizes spelling and filler text;
- scores multi-diagnosis strings using the first diagnosis concept;
- gives no credit to negated or empty diagnoses;
- builds case-specific diagnosis buckets from the reference answer, rubric
  language, and `item.mcq.options`;
- scores localization component-wise;
- emits trajectory labels and normalized diagnosis scores.

The current scorer identifier is `rule_v8_rubric_extracted_terms`.

## Ablations

`ablation_main.py` keeps non-official settings separate from the main pipeline:

- `history-only`
- `all-images-at-once`
- `random-order-reveal`
- `gold-order-reveal`
- `oracle-findings`

Passive ablations do not have applicable route/request metrics. The
`oracle-findings` setting remains request based, so those metrics are defined.

## Output contract

`benchmark_summary.json` contains:

- run configuration and hashes;
- run progress and failures;
- dataset preflight;
- aggregate metrics and Table 2 rows;
- target-output health;
- request outcomes;
- judge transport and agreement summaries.

`benchmark_full.json` adds complete per-case logs, turns, requests, judge
outputs, metrics, and debug metadata.

Always retain code and dataset commits, dependency manifests, server logs, and
both benchmark JSON files for a reported reproduction.
