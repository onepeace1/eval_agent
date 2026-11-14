#!/usr/bin/env python3
"""Serve the A2 models on your own GPU with Docker, in place of the Modal apps.

Standard library only, so it runs with plain `python3`:

    python3 infrastructure/docker/serve.py build            # vLLM + agent-runner images
    python3 infrastructure/docker/serve.py build --glm      # also the GLM-4.7 image
    python3 infrastructure/docker/serve.py up validator --gpus 3
    python3 infrastructure/docker/serve.py up qwen --gpus 3
    python3 infrastructure/docker/serve.py env              # .env lines for what is up
    python3 infrastructure/docker/serve.py status | logs NAME [-f] | down NAME|all

Each model runs exactly as its Modal class does (same model, served names, and
vLLM flags; see validator_model.py and generation_models.py). The differences:

* GPU memory is shared. Modal gives each model its own GPU at 0.95; here each
  gets a fraction of the GPUs you name (`--gpu-memory-utilization`), with
  defaults that let validator + qwen + ministral fit on one 96 GB card at once.
  GLM-4.7-Flash (about 58 GB of weights) needs most of a card on its own.
* Endpoints are http://localhost:<port>/v1 with any API key, instead of Modal
  URLs behind proxy auth. All servers join the Docker network `hw2-net`, where
  the agent-runner containers reach them as http://hw2-<name>:8000/v1.
* Weights are cached in ~/.cache/huggingface, and vLLM's compile cache under
  .local_state/, instead of the hw2-* Modal volumes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1]
STATE = RELEASE / ".local_state"
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
NETWORK = "hw2-net"
LABEL = "hw2-local=1"

VLLM_IMAGE = "hw2-vllm:0.13.0"
GLM_IMAGE = "hw2-vllm-glm:0.26.1rc1"
RUNNER_IMAGE = "hw2-agent-runner:2.4.5"
IMAGES = {
    VLLM_IMAGE: "vllm.Dockerfile",
    RUNNER_IMAGE: "runner.Dockerfile",
    GLM_IMAGE: "vllm-glm.Dockerfile",
}

COMMON = ["--max-model-len", "32768", "--max-num-batched-tokens", "32768"]


@dataclass(frozen=True)
class Server:
    model: str
    alias: str
    port: int
    memory: float  # default --gpu-memory-utilization
    flags: tuple[str, ...]
    image: str = VLLM_IMAGE
    url_variable: str = ""


#: Mirrors the Modal classes. Keep the flags in step with those files.
SERVERS: dict[str, Server] = {
    "validator": Server(
        "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8", "validator", 8201, 0.45,
        ("--tool-call-parser", "hermes", "--enable-auto-tool-choice",
         "--max-num-seqs", "2", *COMMON),
    ),
    "qwen": Server(
        "Qwen/Qwen2.5-Coder-3B-Instruct", "qwen", 8202, 0.15,
        ("--tool-call-parser", "hermes", "--enable-auto-tool-choice",
         "--max-num-seqs", "4", *COMMON),
        url_variable="GENERATION_QWEN_BASE_URL",
    ),
    "ministral": Server(
        "mistralai/Ministral-3-14B-Instruct-2512", "ministral", 8203, 0.35,
        ("--tokenizer-mode", "mistral", "--config-format", "mistral",
         "--load-format", "mistral", "--tool-call-parser", "mistral",
         "--enable-auto-tool-choice", "--limit-mm-per-prompt", '{"image": 0}',
         "--max-num-seqs", "2", *COMMON),
        url_variable="GENERATION_MINISTRAL_BASE_URL",
    ),
    "glm": Server(
        "zai-org/GLM-4.7-Flash", "glm-4.7", 8204, 0.85,
        ("--tool-call-parser", "glm47", "--reasoning-parser", "glm45",
         "--enable-auto-tool-choice", "--max-num-seqs", "2", *COMMON),
        image=GLM_IMAGE,
        url_variable="GENERATION_GLM_BASE_URL",
    ),
}


def docker(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=check, text=True, capture_output=capture)


def container(name: str) -> str:
    return f"hw2-{name}"


def running(name: str) -> bool:
    state = docker("inspect", "-f", "{{.State.Running}}", container(name), capture=True, check=False)
    return state.stdout.strip() == "true"


def user_args() -> list[str]:
    """Rootless Docker already maps container root to you; otherwise run as you."""
    info = docker("info", "--format", "{{json .SecurityOptions}}", capture=True, check=False)
    return [] if "rootless" in info.stdout else ["--user", f"{os.getuid()}:{os.getgid()}"]


def ensure_network() -> None:
    if docker("network", "inspect", NETWORK, capture=True, check=False).returncode != 0:
        docker("network", "create", "--label", LABEL, NETWORK, capture=True)


def build(args: argparse.Namespace) -> None:
    for image, dockerfile in IMAGES.items():
        if image == GLM_IMAGE and not args.glm:
            continue
        print(f"building {image} from {dockerfile}", flush=True)
        docker("build", "-t", image, "-f", str(HERE / dockerfile), str(HERE))


def warmup(port: int, alias: str) -> None:
    """One request through the server, as serving._warmup does on Modal."""
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({
            "model": alias,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 8,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        reply = json.load(response)["choices"][0]["message"].get("content")
    print(f"warmup reply from {alias}: {reply!r}")


def wait_ready(name: str, server: Server, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not running(name):
            docker("logs", "--tail", "80", container(name), check=False)
            raise SystemExit(f"{container(name)} exited before it was ready (logs above).")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
                served = [item["id"] for item in json.load(response)["data"]]
            print(f"ready: {', '.join(served)} on http://localhost:{port}/v1")
            warmup(port, server.alias)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
            time.sleep(5)
    raise SystemExit(f"{container(name)} not ready after {timeout:.0f}s; see `serve.py logs {name}`.")


def up(args: argparse.Namespace) -> None:
    server = SERVERS[args.name]
    model = args.model or server.model
    port = args.port or server.port
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if docker("inspect", container(args.name), capture=True, check=False).returncode == 0:
        raise SystemExit(f"{container(args.name)} already exists; `serve.py down {args.name}` first.")
    if docker("image", "inspect", server.image, capture=True, check=False).returncode != 0:
        flag = " --glm" if server.image == GLM_IMAGE else ""
        raise SystemExit(f"{server.image} is not built; run `serve.py build{flag}`.")

    ensure_network()
    vllm_cache = STATE / "vllm-cache"
    for directory in (HF_CACHE, vllm_cache):
        directory.mkdir(parents=True, exist_ok=True)
    command = [
        "run", "-d", "--name", container(args.name), "--label", LABEL, *user_args(),
        "--network", NETWORK, "--ipc=host", "-p", f"127.0.0.1:{port}:8000",
        "-v", f"{HF_CACHE}:/cache/huggingface", "-v", f"{vllm_cache}:/cache/vllm",
    ]
    for gpu in gpus:
        command += ["--device", f"nvidia.com/gpu={gpu}"]
    if os.environ.get("HF_TOKEN"):
        command += ["-e", "HF_TOKEN"]
    command += [
        server.image, "vllm", "serve", model,
        "--served-model-name", model, server.alias,
        "--host", "0.0.0.0", "--port", "8000",
        "--tensor-parallel-size", str(len(gpus)),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization or server.memory),
        *server.flags, *args.extra,
    ]
    docker(*command, capture=True)
    print(f"started {container(args.name)} ({model}) on GPU {','.join(gpus)}; waiting "
          "(the first start downloads the weights)...", flush=True)
    wait_ready(args.name, server, port, args.wait)


def env_lines(_: argparse.Namespace) -> None:
    """The .env settings for the servers that are running now."""
    lines = []
    if running("validator"):
        lines += [f"VALIDATOR_BASE_URL=http://localhost:{port_of('validator')}/v1",
                  "VALIDATOR_API_KEY=local", "VALIDATOR_MODEL=validator"]
    for name, server in SERVERS.items():
        if server.url_variable and running(name):
            lines.append(f"{server.url_variable}=http://localhost:{port_of(name)}/v1")
    if any(SERVERS[name].url_variable and running(name) for name in SERVERS):
        lines.append("GENERATION_API_KEY=local")
    print("\n".join(lines) if lines else "no hw2 servers are running")


def port_of(name: str) -> str:
    published = docker("port", container(name), "8000/tcp", capture=True, check=False).stdout
    return published.strip().rsplit(":", 1)[-1] if published.strip() else str(SERVERS[name].port)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    images = commands.add_parser("build", help="build the vLLM and agent-runner images")
    images.add_argument("--glm", action="store_true", help="also build the GLM-4.7 vLLM image")
    start = commands.add_parser("up", help="start one model server and wait until it answers")
    start.add_argument("name", choices=SERVERS)
    start.add_argument("--gpus", default="3", help="comma-separated GPU indices (default: 3)")
    start.add_argument("--port", type=int, help="host port (default: per model, 8201-8204)")
    start.add_argument("--gpu-memory-utilization", type=float)
    start.add_argument("--model", help="serve a different checkpoint under the same alias")
    start.add_argument("--wait", type=float, default=3 * 3600)
    commands.add_parser("env", help="print .env lines for the running servers")
    commands.add_parser("status")
    stop = commands.add_parser("down", help="stop and remove a server (or `all`)")
    stop.add_argument("name", choices=[*SERVERS, "all"])
    logs = commands.add_parser("logs")
    logs.add_argument("name", choices=SERVERS)
    logs.add_argument("-f", "--follow", action="store_true")
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    args = parser.parse_args(argv)
    args.extra = extra  # `vllm serve` flags after a literal --

    if args.command == "build":
        build(args)
    elif args.command == "up":
        up(args)
    elif args.command == "env":
        env_lines(args)
    elif args.command == "status":
        docker("ps", "-a", "--filter", "label=hw2-local", "--format",
               "{{.Names}}\t{{.Status}}\t{{.Ports}}")
    elif args.command == "down":
        names = list(SERVERS) if args.name == "all" else [args.name]
        for name in names:
            docker("rm", "-f", container(name), check=False, capture=True)
            print(f"removed {container(name)}")
    elif args.command == "logs":
        docker("logs", *(["-f"] if args.follow else ["--tail", "100"]), container(args.name), check=False)


if __name__ == "__main__":
    main()
