"""Run generation tasks in local Docker containers instead of the Modal runner.

`DockerAgentRunner().remote(request)` has the same contract as the deployed
`hw2-agent-runner/run_task` Modal function: one fresh container per task, the
same pinned mini-SWE-agent scaffold (infrastructure/agent_runner.py, mounted
read-only), and the same result dictionary back.

The model endpoints are the local vLLM servers started with
`infrastructure/docker/serve.py up ...`, reached over the `hw2-net` Docker
network. A non-local URL in .env (for example a Modal one) is used as given.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv

RELEASE = Path(__file__).resolve().parents[1]
INFRASTRUCTURE = RELEASE / "infrastructure"
IMAGE = "hw2-agent-runner:2.4.5"
NETWORK = "hw2-net"
#: The Modal function's own timeout (agent_runner.run_task).
TIMEOUT_SECONDS = 20 * 60
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

#: url_variable -> container name of the local server (infrastructure/docker/serve.py).
LOCAL_SERVERS = {
    "GENERATION_QWEN_BASE_URL": "hw2-qwen",
    "GENERATION_MINISTRAL_BASE_URL": "hw2-ministral",
    "GENERATION_GLM_BASE_URL": "hw2-glm",
}


def backend() -> str:
    value = os.environ.get("HW2_BACKEND", "docker").strip().lower() or "docker"
    if value not in {"docker", "modal"}:
        raise ValueError(f"HW2_BACKEND={value!r}; expected docker or modal")
    return value


def endpoint_registry() -> dict[str, dict[str, str]]:
    """Model name -> endpoint, as the gateway's GENERATION_ENDPOINTS_JSON holds it."""
    from .generate import AGENTS

    load_dotenv()
    registry = {}
    for agent in AGENTS:
        url = os.getenv(agent.url_variable, "")
        if not url or urlparse(url).hostname in LOCAL_HOSTS or "<workspace>" in url:
            # Inside the runner container, localhost is the container itself.
            registry[agent.model] = {
                "api_base": f"http://{LOCAL_SERVERS[agent.url_variable]}:8000/v1",
                "api_key": "local",
            }
        else:
            registry[agent.model] = {
                "api_base": url.rstrip("/"),
                "api_key": os.getenv("GENERATION_API_KEY", ""),
            }
    return registry


class DockerAgentRunner:
    def __init__(self, image: str = IMAGE, network: str = NETWORK):
        self.image = image
        self.network = network
        self.registry = endpoint_registry()
        if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode:
            raise RuntimeError(
                f"{image} is not built; run `python3 infrastructure/docker/serve.py build`"
            )
        if subprocess.run(["docker", "network", "inspect", network], capture_output=True).returncode:
            raise RuntimeError(
                f"Docker network {network} does not exist; start a model with "
                "`python3 infrastructure/docker/serve.py up <name>` first"
            )

    def remote(self, request: dict[str, Any]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="hw2-run-") as scratch:
            io = Path(scratch)
            os.chmod(io, 0o777)
            (io / "request.pkl").write_bytes(pickle.dumps(request))
            command = [
                "docker", "run", "--rm", "--network", self.network,
                "--label", "hw2-local=1", "--label", "hw2-agent-run=1",
                "-v", f"{io}:/io",
                "-v", f"{INFRASTRUCTURE}:/opt/hw2:ro",
                "-e", "GENERATION_ENDPOINTS_JSON",
                self.image, "python", "/opt/hw2/docker/local_runner.py", "/io",
            ]
            environment = {**os.environ, "GENERATION_ENDPOINTS_JSON": json.dumps(self.registry)}
            finished = subprocess.run(
                command, env=environment, capture_output=True, text=True, timeout=TIMEOUT_SECONDS
            )
            result = io / "result.pkl"
            if finished.returncode != 0 or not result.is_file():
                log = (finished.stdout + finished.stderr).strip()[-3000:]
                raise RuntimeError(f"agent runner container exited {finished.returncode}:\n{log}")
            return pickle.loads(result.read_bytes())
