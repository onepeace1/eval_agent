# Self-hosting the training backend with SkyRL and Docker (local GPUs)

Docker replacement for [`../skyrl_tinker_modal/`](../skyrl_tinker_modal/README.md): same SkyRL
revision, same image contents, same backend settings. Only GPU placement adapts to the GPUs you
pass. The Modal files are untouched and still work.

```bash
python3 skyrl_tinker_docker/serve.py build                  # once, ~20 min, ~43 GB image
python3 skyrl_tinker_docker/serve.py smoke --gpus 3         # CUDA / library check
python3 skyrl_tinker_docker/serve.py up sync --gpus 3       # start server (port 9000)
export TINKER_BASE_URL=http://localhost:9000
uv run python skyrl_tinker_docker/smoke_client.py           # optional: one real train step
uv run python scripts/train_sync.py --config configs/p2b_number_search_grpo.yaml
python3 skyrl_tinker_docker/serve.py down sync              # free the GPU
```

| Profile | Parts | Placement | Min GPUs |
|---|---|---|---|
| `up sync` | 2, 4.1 | trainer + vLLM colocated on all GPUs | 1 |
| `up async` | 3, 4.3 | 1 trains, rest sample | 2 |
| `up textcraft` | 5 | 1 trains, rest sample | 2 (Modal used 4) |
| `eval --adapter DIR` | 5 eval | vLLM data-parallel | 1 |

- **One run per server.** Start a fresh server (`down`, then `up`) for each run. A second LoRA
  model on the same server crashes in SkyRL's adapter swap. Use `--name` and `--port` to run
  several servers at once (e.g. `up sync --gpus 3 --name p2a` and
  `up sync --gpus 4 --name p41 --port 9001`).
- State (checkpoints, SkyRL database) lives in `skyrl_state/<name>/`, so it survives restarts and
  `--set train.resume=true` works. Part 5's `modal volume get .../sampler_weights/<snap>.tar.gz`
  becomes plain `skyrl_state/textcraft/checkpoints/<model-id>/sampler_weights/<snap>.tar.gz`.
- First request after `up` takes about 5 min (Megatron + vLLM load and CUDA-graph capture).
- Debugging: `serve.py logs <name> -f`, `serve.py status`, `up ... --env CUDA_LAUNCH_BLOCKING=1`,
  `up ... --override generator.inference_engine.gpu_memory_utilization=0.6`.

## Blackwell (RTX PRO 6000, sm_120) fixes applied by `serve.py`

| Env var | Why |
|---|---|
| `VLLM_USE_FLASHINFER_SAMPLER=0` | FlashInfer won't JIT for SM 12.x with CUDA < 12.9 (stack is cu128), so vLLM crashed at startup. Uses vLLM's PyTorch sampler instead. |
| `FLA_TILELANG=0` | SkyRL forces fla's Hopper-tuned TileLang kernel for Qwen3.5's Gated-DeltaNet; its backward kernel faults (`MISALIGNED_ADDRESS`) on sm_120. SkyRL's own code comment says to set this on Blackwell. |

Rootless Docker is detected automatically (containers run as container-root, which is your user).
