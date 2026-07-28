# Tested stack

Validation date: 2026-07-27

- DDx-TRACE code commit: `7e206a8af9161843109b4a4fd5d1395788f35fc3`
- DDx-TRACE dataset commit: `566c96646c9c8713ecda36860150b3cafd50e306`
- Dataset cases: 191
- Extracted image files: 6802
- Python: 3.10.20 in isolated Conda environments
- vLLM: 0.26.0
- torch: 2.11.0+cu130
- CUDA runtime bundled with torch: 13.0
- transformers: 5.14.1
- tokenizers: 0.22.2
- flashinfer-python: 0.6.14
- Model: local `google-medgemma-27b-it`, bfloat16, approximately 52 GB
- GPU: NVIDIA H100, 95,830 MiB
- NVIDIA driver: 580.159.03
- vLLM maximum model length for the smoke test: 16,384
- vLLM GPU memory utilization target: 0.90
- vLLM health endpoint: passed
- vLLM served model: `medgemma-27b-it`

Three-case rule-judge validation:

- Case IDs: `12789`, `13789`, `13843`
- Attempted/completed/failed: 3/3/0
- Agent turns: 21
- Native JSON-schema turns: 21
- JSON parse failures: 0
- Parse or normalization warnings: 0
- Blocking dataset preflight errors: 0
- Final validator result: passed

On the tested quota-constrained cluster, all serving caches had to use project
storage. The successful server process set `XDG_CACHE_HOME`, `HF_HOME`,
`FLASHINFER_WORKSPACE_BASE`, `TRITON_CACHE_DIR`,
`TORCHINDUCTOR_CACHE_DIR`, and `CUDA_CACHE_PATH`; its short ZeroMQ `TMPDIR`
used node-local `/tmp`.

Top-level benchmark dependencies resolved during validation:

- `httpx==0.28.1`
- `anthropic==0.120.0`
- `google-genai==2.14.0`

Use the generated `benchmark-pip-freeze.txt` and `vllm-pip-freeze.txt` as the
complete machine-readable manifests. Record new commits and manifests when the
Hub repositories or package indexes change.
