#!/usr/bin/env python3
"""Serve the agent's model on a local GPU, in place of the course endpoint.

Runs vLLM's OpenAI-compatible server in Docker on the GPUs you name and prints
the three `.env` lines that point the harness at it. Standard library only:

    python3 scripts/local_llm.py up --gpus 3                    # openai/gpt-oss-120b
    python3 scripts/local_llm.py up --gpus 3 --model Qwen/Qwen3.5-4B
    python3 scripts/local_llm.py status | logs [-f] | down

The model is served under its own name, so OPENAI_MODEL / --model stay the
Hugging Face id (e.g. openai/gpt-oss-120b, which the Makefile already names as
GPT_OSS_MODEL). Weights go to ~/.cache/huggingface and are downloaded once.
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
from pathlib import Path

NAME = "a1-llm"
DEFAULT_PORT = 8100
DEFAULT_MODEL = "openai/gpt-oss-120b"
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))

#: Any image with a recent `vllm` on PATH works. The default is the one A3's
#: `skyrl_tinker_docker/serve.py build` produces (vLLM 0.26, tested on
#: Blackwell); `vllm/vllm-openai:<tag>` from Docker Hub works as well.
DEFAULT_IMAGE = os.environ.get("A1_VLLM_IMAGE", "scottyrl-skyrl:491e132")

#: Per-model vLLM flags: tool calls and reasoning must be parsed out of the
#: completion, because the harness uses OpenAI function calling with
#: `reasoning_effort`.
PROFILES: dict[str, list[str]] = {
    "openai/gpt-oss-120b": [
        "--tool-call-parser", "openai", "--reasoning-parser", "openai_gptoss",
        "--max-model-len", "65536",
    ],
    "openai/gpt-oss-20b": [
        "--tool-call-parser", "openai", "--reasoning-parser", "openai_gptoss",
        "--max-model-len", "65536",
    ],
    "Qwen/Qwen3.5-4B": [
        "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
        "--max-model-len", "65536",
    ],
}

#: vLLM 0.26 in a CUDA 12.8 image cannot JIT FlashInfer's sampler for sm_120
#: (Blackwell workstation GPUs); the PyTorch sampler is used instead.
CONTAINER_ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


def docker(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=check, text=True, capture_output=capture)


def user_args() -> list[str]:
    """Rootless Docker already maps container root to you; otherwise run as you."""
    info = docker("info", "--format", "{{json .SecurityOptions}}", capture=True, check=False)
    if "rootless" in info.stdout:
        return []
    return ["--user", f"{os.getuid()}:{os.getgid()}"]


def wait_ready(port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        state = docker("inspect", "-f", "{{.State.Running}}", NAME, capture=True, check=False).stdout.strip()
        if state != "true":
            docker("logs", "--tail", "60", NAME, check=False)
            raise SystemExit(f"{NAME} exited before it was ready (logs above).")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                models = [item["id"] for item in json.load(response)["data"]]
                print(f"ready: {', '.join(models)}")
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
            time.sleep(5)
    raise SystemExit(f"{NAME} not ready after {timeout:.0f}s; see `local_llm.py logs`.")


def up(args: argparse.Namespace) -> None:
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if docker("inspect", NAME, capture=True, check=False).returncode == 0:
        raise SystemExit(f"{NAME} already exists; run `local_llm.py down` first.")
    if docker("image", "inspect", args.image, capture=True, check=False).returncode != 0:
        raise SystemExit(
            f"Image {args.image} not found. Build it with A3's `skyrl_tinker_docker/serve.py build`, "
            "or pass --image vllm/vllm-openai:<tag>."
        )
    flags = PROFILES.get(args.model)
    if flags is None:
        print(f"note: no profile for {args.model}; serving without tool/reasoning parsers", file=sys.stderr)
        flags = []
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    command = [
        "run", "-d", "--name", NAME, "--label", "assignment-llm=1", *user_args(),
        "--ipc=host", "-p", f"127.0.0.1:{args.port}:8000",
        "-v", f"{HF_CACHE}:/cache/huggingface", "-e", "HF_HOME=/cache/huggingface",
    ]
    for gpu in gpus:
        command += ["--device", f"nvidia.com/gpu={gpu}"]
    for key, value in CONTAINER_ENV.items():
        command += ["-e", f"{key}={value}"]
    if os.environ.get("HF_TOKEN"):
        command += ["-e", "HF_TOKEN"]
    command += [
        args.image, "vllm", "serve", args.model,
        "--host", "0.0.0.0", "--port", "8000",
        "--tensor-parallel-size", str(len(gpus)),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--enable-auto-tool-choice", *flags, *args.extra,
    ]
    docker(*command)
    print(f"started {NAME} on GPU {','.join(gpus)}; waiting (first run downloads the weights)...")
    wait_ready(args.port, args.wait)
    print("\nPut these in .env:\n")
    print(f"OPENAI_BASE_URL=http://localhost:{args.port}/v1")
    print("OPENAI_API_KEY=local")
    print(f"OPENAI_MODEL={args.model}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("up", help="start the server and wait until it answers")
    start.add_argument("--gpus", default="3", help="comma-separated GPU indices (default: 3)")
    start.add_argument("--model", default=DEFAULT_MODEL)
    start.add_argument("--port", type=int, default=DEFAULT_PORT)
    start.add_argument("--image", default=DEFAULT_IMAGE)
    start.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    start.add_argument("--wait", type=float, default=3 * 3600, help="seconds to wait for readiness")
    commands.add_parser("down", help="stop and remove the server")
    commands.add_parser("status")
    logs = commands.add_parser("logs")
    logs.add_argument("-f", "--follow", action="store_true")
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    args = parser.parse_args(argv)
    args.extra = extra  # `vllm serve` flags after a literal --

    if args.command == "up":
        up(args)
    elif args.command == "down":
        docker("rm", "-f", NAME, check=False)
    elif args.command == "status":
        docker("ps", "-a", "--filter", f"name=^{NAME}$", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}")
    elif args.command == "logs":
        docker("logs", *(["-f"] if args.follow else ["--tail", "100"]), NAME, check=False)


if __name__ == "__main__":
    main()
