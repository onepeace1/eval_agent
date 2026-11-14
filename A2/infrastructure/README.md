# Model endpoints and the agent runner

## Local Docker (default)

Everything below the "Modal" heading can also run on your own GPU. The Docker
files live in `infrastructure/docker/`; they serve the same models with the
same vLLM versions and flags, and run the same pinned mini-SWE-agent runner.

```bash
python3 infrastructure/docker/serve.py build          # vLLM 0.13 + agent-runner images
python3 infrastructure/docker/serve.py build --glm    # only if you need GLM-4.7-Flash
python3 infrastructure/docker/serve.py up validator --gpus 3
python3 infrastructure/docker/serve.py up qwen --gpus 3        # Part 3 models, as needed
python3 infrastructure/docker/serve.py env            # paste the printed lines into .env
uv run python -m workflow generate --models qwen      # runs in local containers
python3 infrastructure/docker/serve.py down all
```

| Name | Model | Alias | Port | Default GPU share |
| --- | --- | --- | --- | --- |
| `validator` | `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` | `validator` | 8201 | 0.45 |
| `qwen` | `Qwen/Qwen2.5-Coder-3B-Instruct` | `qwen` | 8202 | 0.15 |
| `ministral` | `mistralai/Ministral-3-14B-Instruct-2512` | `ministral` | 8203 | 0.35 |
| `glm` | `zai-org/GLM-4.7-Flash` | `glm-4.7` | 8204 | 0.85 |

- Validator, qwen and ministral fit on one 96 GB GPU together. GLM needs most
  of a GPU, so run it alone or on another GPU (`--gpus 4`). Change a share with
  `--gpu-memory-utilization`.
- Weights are cached in `~/.cache/huggingface`, so each model downloads once.
  Starting a cached model takes a few minutes (load + compile + warmup).
- `workflow generate` starts one fresh `hw2-agent-runner` container per task
  (`workflow/docker_runner.py`). The container runs
  `infrastructure/agent_runner.py` unchanged; the only substitution is that
  its gateway call runs in-process (`docker/local_runner.py`) instead of
  through a deployed Modal function. The endpoint registry is removed from the
  environment before the agent starts, so agent-written commands cannot see it.
- Runner containers reach the models over the `hw2-net` Docker network, so
  localhost URLs in `.env` are rewritten to `http://hw2-<name>:8000/v1` for
  them. Non-local URLs (e.g. Modal) are used as written.
- Set `HW2_BACKEND=modal` to use the deployed Modal runner instead.

## Modal

These files are the canonical model deployments for both student development
and staff grading. Each Modal workspace gets its own apps, volumes, endpoint
URLs, and usage charges.

From the student release directory:

```bash
uv sync
uv run modal setup
scripts/deploy_validator_model.sh
scripts/deploy_generation_models.sh  # Part 3 model servers
# Fill in the generation endpoint settings in .env.
scripts/deploy_agent_runner.sh       # Part 3 gateway and mini-SWE-agent runner
```

The validator deployment serves `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` with the
stable model alias `validator`. The generation deployment creates three
independently scaling containers:

| Endpoint class | Model | Alias | GPU |
| --- | --- | --- | --- |
| `QwenServer` | `Qwen/Qwen2.5-Coder-3B-Instruct` | `qwen` | L4 |
| `MinistralServer` | `mistralai/Ministral-3-14B-Instruct-2512` | `ministral` | L40S |
| `GlmServer` | `zai-org/GLM-4.7-Flash` | `glm-4.7` | H100 |

All endpoints require Modal proxy authentication and expose vLLM's
OpenAI-compatible API. Copy `.env.example` to `.env`, append `/v1` to the
validator URL printed by Modal, and add a Modal proxy token in combined
`<token-id>.<token-secret>` form.

The apps scale to zero after five idle minutes. Deploying creates stable URLs;
the first request after an idle period starts a GPU container and may take
several minutes while the model loads. A completely empty GLM cache can take
about 12 minutes on its first run because it downloads roughly 58 GiB of model
weights; later starts reuse the workspace volumes and compile cache.

For generation, append `/v1` to each URL printed for `QwenServer.serve`,
`MinistralServer.serve`, and `GlmServer.serve`. Put them in `.env` as
`GENERATION_QWEN_BASE_URL`, `GENERATION_MINISTRAL_BASE_URL`, and
`GENERATION_GLM_BASE_URL`, respectively, and set `GENERATION_API_KEY` to the
combined proxy token. The Part 3 generation command is documented in
`ASSIGNMENT.md`.

`deploy_agent_runner.sh` stores those settings in the private Modal secret
`hw2-generation-endpoints`, deploys `generation_gateway.py` with access to that
secret, and deploys `agent_runner.py` without it. The latter is the original
`mini-swe-agent==2.4.5` bash-tool scaffold used for the released trajectories;
agent-written commands execute in its fresh Modal container and cannot read the
generation endpoint credentials.
