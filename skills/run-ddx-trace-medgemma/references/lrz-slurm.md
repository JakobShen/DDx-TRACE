# LRZ/MCML Slurm workflow

Use a durable `screen` session before entering a compute allocation:

```bash
ssh USER@login.ai.lrz.de
hostname
screen -S ddx-medgemma
squeue -j JOB_ID
srun --jobid=JOB_ID --overlap --pty bash -l
hostname
nvidia-smi
```

Detach with `Ctrl-a d`. Reattach with:

```bash
screen -r ddx-medgemma
```

`screen` sockets are local to the login node. `login.ai.lrz.de` may route new
connections to a different login node. Record the login-node hostname when
creating the session and reconnect to the same node before using `screen -r`.
Do not assume a missing session was terminated until the node is confirmed.

Keep code, data, Conda environments, caches, logs, and benchmark outputs on
project storage. Home quotas are commonly too small for Conda metadata, vLLM,
torch wheels, and model caches:

```bash
export WORK=/project/path/ddx-trace-work
export CONDA_PKGS_DIRS="$WORK/cache/conda-pkgs"
export PIP_CACHE_DIR="$WORK/cache/pip"
export TMPDIR="$WORK/cache/tmp"
export PYTHONNOUSERSITE=1
```

Run the vLLM server and benchmark client on the same compute node when using
`127.0.0.1`. Do not start the GPU server on a login node. Use
`scripts/start_vllm.sh` so FlashInfer, Triton, TorchInductor, CUDA, Hugging
Face, and vLLM caches are also redirected away from home storage.
