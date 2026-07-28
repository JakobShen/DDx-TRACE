# DDx-TRACE

DDx-TRACE evaluates whether a model can actively request radiology evidence,
update a four-item differential diagnosis, and reach the correct diagnosis
with an efficient investigation path.

- Dataset: [Anonym001/DDx-TRACE](https://huggingface.co/datasets/Anonym001/DDx-TRACE)
- Open-source MedGemma: [README_MEDGEMMA.md](README_MEDGEMMA.md)
- Benchmark and metric details: [docs/BENCHMARK_REFERENCE.md](docs/BENCHMARK_REFERENCE.md)

## Install

Python 3.10 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Get the data

```bash
git clone https://huggingface.co/datasets/Anonym001/DDx-TRACE DDx-TRACE-data
cd DDx-TRACE-data
git lfs pull
sha256sum -c data/DDx-TRACE-images.tar.zst.sha256
tar --use-compress-program=unzstd -xf data/DDx-TRACE-images.tar.zst
cd ..
```

After extraction, these paths must exist:

```text
DDx-TRACE-data/data/eurorad_neuro_01_release.json
DDx-TRACE-data/images/
```

## Run a smoke test

Set the credential for the target provider, then run three cases with the
deterministic rule judge:

```bash
export OPENAI_API_KEY="..."

python run_main.py \
  --data-path DDx-TRACE-data/data/eurorad_neuro_01_release.json \
  --image-root DDx-TRACE-data \
  --out-dir runs/openai-smoke \
  --provider openai \
  --target-model gpt-5.4 \
  --judge-modes rule \
  --limit 3
```

Supported providers are `openai`, `anthropic`, `gemini`, `vertex`, `qwen`,
and `vllm`. Run `python run_main.py --help` for optional arguments.

## Run MedGemma locally

Use the tested vLLM 0.26.0 workflow in
[README_MEDGEMMA.md](README_MEDGEMMA.md). An agent can execute the companion
[MedGemma skill](skills/run-ddx-trace-medgemma/SKILL.md).

## Add the LLM judge

After the rule-only smoke test passes:

```bash
export VERTEX_API_KEY="..."
```

Replace `--judge-modes rule` with:

```bash
--judge-modes both \
--judge-provider vertex \
--judge-model gemini-3-flash-preview
```

Remove `--limit 3` only after the smoke test succeeds.

## Outputs

Each run creates a timestamped directory containing:

- `benchmark_summary.json`: configuration, preflight checks, failures, and
  aggregate metrics;
- `benchmark_full.json`: the summary plus per-case and per-turn traces.

A valid smoke test has:

- `run_progress.is_final = true`;
- three completed and zero failed cases;
- no blocking preflight error;
- `target_output_health.json_parse_failed_turns = 0`.

## Ablations

Use `ablation_main.py` for `history-only`, `all-images-at-once`,
`random-order-reveal`, `gold-order-reveal`, and `oracle-findings`.

```bash
python ablation_main.py --help
```

## Data and credentials

This repository contains code and documentation only. Do not commit datasets,
medical images, generated runs, API keys, or service-account files. Copy
`aa_api_key.env.example` to a local untracked file when needed.
