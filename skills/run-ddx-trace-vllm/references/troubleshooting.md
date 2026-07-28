# Troubleshooting

## Environment creation

- `No module named pip` or `No module named ensurepip`: the system Python
  cannot create a usable venv. Create a Conda environment with
  `python=3.10 pip`.
- `Disk quota exceeded`: move `CONDA_PKGS_DIRS`, `PIP_CACHE_DIR`, and `TMPDIR`
  under the selected working directory or another filesystem with enough
  space.
- Packages resolve from `~/.local/lib/python3.10`: export
  `PYTHONNOUSERSITE=1` for installation and execution.
- Unexpected dependency versions: inspect the manifests written by
  `create_envs.sh`; do not rely on an activated base environment.

## Dataset

- Zero images: the Hugging Face repository stores images in
  `data/DDx-TRACE-images.tar.zst`; verify the checksum and extract it.
- Checksum failure: rerun `git lfs pull` inside the dataset repository. Do not
  evaluate a partial archive.
- Blocking preflight errors: fix manifest/image-root resolution. Do not report
  a run made with `--allow-preflight-errors`.

## vLLM

- Server never becomes healthy: inspect the complete server log and
  `nvidia-smi`. Model loading can take several minutes.
- `ZMQError` reports an IPC path longer than 107 characters: the serving
  `TMPDIR` is too long. Use a short local path such as
  `/tmp/$USER/ddx-vllm`; keep package caches under the selected cache
  directory.
- vLLM, Torch, Hugging Face, FlashInfer, Triton, or CUDA caches hit the home
  quota: export `VLLM_NO_USAGE_STATS=1` and point `XDG_CACHE_HOME`, `HF_HOME`,
  `FLASHINFER_WORKSPACE_BASE`, `TRITON_CACHE_DIR`,
  `TORCHINDUCTOR_CACHE_DIR`, and `CUDA_CACHE_PATH` to the selected cache
  directory.
- GPU out of memory: lower `--max-model-len`, reduce
  `--gpu-memory-utilization`, or select a smaller compatible multimodal model.
- `404 model not found`: make `--target-model` exactly match
  `--served-model-name`.
- Chat-template role error: confirm the checkpoint includes a compatible chat
  template and is instruction tuned.
- Image request error: confirm the checkpoint is multimodal and the request
  reaches `/v1/chat/completions` with `image_url` content.
- JSON-schema error: the benchmark's `auto` mode should retry without native
  schema. If needed, set `--agent-structured-output never` and inspect
  `target_output_health`.

## Benchmark output

- Failed cases with HTTP timeout: raise `--timeout` only after confirming the
  server is alive and GPU utilization is progressing.
- Nonzero `json_parse_failed_turns`: inspect raw model output before accepting
  the run.
- Frequent padding, trimming, or probability renormalization: treat the run as
  an output-format integration problem even when all cases completed.
- Vertex judge failure: verify `VERTEX_API_KEY` is present in the environment.
  Do not reuse a candidate API-key flag as the judge credential.
