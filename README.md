# eval_agent

Study copies of the CMU 11-768 (AI Agents) assignments, set up to run on local
GPUs with Docker instead of Modal.

| Folder | Assignment | Upstream |
| --- | --- | --- |
| [A1/](A1/) | Build an agent harness (code + chess agents) | [cmu-agents/assignment-1](https://github.com/cmu-agents/assignment-1) |
| [A2/](A2/) | Validator for agentic data-visualization runs (eval) | [cmu-agents/assignment-2](https://github.com/cmu-agents/assignment-2) |
| [A3/](A3/) | RL training with a SkyRL Tinker server | [cmu-agents/assignment-3](https://github.com/cmu-agents/assignment-3) |

Each folder keeps its upstream history (merged with `git subtree`); the Docker
changes are one commit per assignment on top. Modal still works in each one:

| | Docker (default) | Modal |
| --- | --- | --- |
| A1 sandbox | `SANDBOX_BACKEND=docker`; local model via `scripts/local_llm.py` | `SANDBOX_BACKEND=modal` |
| A2 models / runner | `infrastructure/docker/serve.py`, `HW2_BACKEND=docker` | `scripts/deploy_*.sh`, `HW2_BACKEND=modal` |
| A3 SkyRL server | `skyrl_tinker_docker/serve.py` | `skyrl_tinker_modal/` |

See [A1/README.md](A1/README.md), [A2/infrastructure/README.md](A2/infrastructure/README.md),
and [A3/skyrl_tinker_docker/README.md](A3/skyrl_tinker_docker/README.md).

After cloning: `git submodule update --init` (A1's chess app).

Pull later upstream changes into a folder with, for example:

    git subtree pull --prefix=A2 https://github.com/cmu-agents/assignment-2 main
