"""Run one generation task in this container, as agent_runner.run_task does on Modal.

Started by workflow/docker_runner.py, one fresh container per task:

    python /opt/hw2/docker/local_runner.py /io

reads /io/request.pkl and writes /io/result.pkl. The task runs through the
unchanged `agent_runner.run_task` body; the only substitution is the gateway
lookup, which calls `generation_gateway.complete` in this process instead of a
deployed Modal function. The endpoint registry is taken out of the environment
before the agent starts, so model-written commands cannot read it, which is the
separation the Modal secret provides.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modal  # noqa: E402

import agent_runner  # noqa: E402
import generation_gateway  # noqa: E402

REGISTRY = json.loads(os.environ.pop("GENERATION_ENDPOINTS_JSON", "{}"))


def _endpoint(model: str) -> dict[str, str]:
    try:
        configured = REGISTRY[model]
    except KeyError as error:
        raise ValueError(f"no generation endpoint configured for {model!r}") from error
    return {"api_base": configured["api_base"], "api_key": configured["api_key"]}


class _LocalGateway:
    def remote(self, request):
        return generation_gateway.complete.local(request)


def _lookup(app_name: str, function_name: str, *args, **kwargs):
    if (app_name, function_name) != (agent_runner.GATEWAY_APP, agent_runner.GATEWAY_FUNCTION):
        raise RuntimeError(f"unexpected Modal lookup {app_name}/{function_name}")
    return _LocalGateway()


def main() -> None:
    io = Path(sys.argv[1])
    generation_gateway._endpoint = _endpoint
    modal.Function.from_name = _lookup
    request = pickle.loads((io / "request.pkl").read_bytes())
    result = agent_runner.run_task.local(request)
    (io / "result.pkl").write_bytes(pickle.dumps(result))


if __name__ == "__main__":
    main()
