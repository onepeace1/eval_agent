#!/usr/bin/env python3
"""Run the SkyRL Tinker server (and the evaluation vLLM server) on local GPUs.

Docker replacement for ../skyrl_tinker_modal/. Same image contents, same
SkyRL revision, same backend settings; only the GPU placement adapts to the
GPUs you pass. Standard library only, so plain `python3` runs it.

    python3 skyrl_tinker_docker/serve.py build
    python3 skyrl_tinker_docker/serve.py smoke --gpus 3
    python3 skyrl_tinker_docker/serve.py up sync --gpus 3 --port 9000
    export TINKER_BASE_URL=http://localhost:9000
    uv run python scripts/train_sync.py --config configs/p2b_number_search_grpo.yaml
    python3 skyrl_tinker_docker/serve.py down sync

Profiles (Modal equivalent in brackets):
    sync       Parts 2, 4.1      colocated on all GPUs   [2x H100]
    async      Parts 3, 4.3      1 trains, rest sample   [2x H100], needs >= 2 GPUs
    textcraft  Part 5            1 trains, rest sample   [4x H100], needs >= 2 GPUs
    eval       Part 5 eval       vLLM only, data-parallel over the GPUs [2x H100]

State (checkpoints, SkyRL's model database) is kept per server name under
skyrl_state/<name>/, so `up` with the same name resumes where it left off.
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Keep in sync with ../skyrl_tinker_modal/skyrl_server.py.
MODEL_NAME = os.environ.get("SKYRL_BASE_MODEL", "Qwen/Qwen3.5-4B")
SKYRL_COMMIT = "491e1329c314ef43818fb6824d9f2ae61883416c"
BACKEND = "megatron"
IMAGE = os.environ.get("SKYRL_IMAGE", f"scottyrl-skyrl:{SKYRL_COMMIT[:7]}")
CONTAINER_PORT = 9000
EVAL_CONTAINER_PORT = 8000
STATE_ROOT = Path(os.environ.get("SKYRL_STATE_ROOT", REPO / "skyrl_state"))
HF_CACHE_HOST = Path(
    os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
)

BASE_BACKEND_CONFIG = {
    "trainer.placement.policy_num_nodes": 1,
    "trainer.policy.megatron_config.tensor_model_parallel_size": 1,
    "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
    "trainer.policy.megatron_config.context_parallel_size": 1,
    "trainer.policy.language_model_only": True,
    "generator.inference_engine.language_model_only": True,
    "trainer.policy.megatron_config.lora_config.merge_lora": False,
    "trainer.policy.optimizer_config.max_grad_norm": 1.0,
    "trainer.policy.optimizer_config.adam_betas": [0.9, 0.95],
    "trainer.policy.optimizer_config.weight_decay": 0.0,
    "trainer.use_expandable_segments": True,
    "trainer.max_tokens_per_microbatch": 32768,
    "generator.inference_engine.max_num_batched_tokens": 50000,
    "generator.inference_engine.max_num_seqs": 256,
    "generator.inference_engine.enable_prefix_caching": True,
    "generator.inference_engine.enforce_eager": False,
    "generator.inference_engine.engine_init_kwargs.gdn_prefill_backend": "triton",
    "generator.inference_engine.engine_init_kwargs.compilation_config": {
        "cudagraph_mode": "FULL_DECODE_ONLY"
    },
    "generator.inference_engine.use_expandable_segments": False,
    "generator.inference_engine.enable_ray_prometheus_stats": True,
    "generator.inference_engine.engine_init_kwargs.enable_mfu_metrics": True,
}


def colocated(num_gpus: int) -> dict:
    return {
        "trainer.placement.colocate_all": True,
        "trainer.placement.policy_num_gpus_per_node": num_gpus,
        "generator.inference_engine.num_engines": num_gpus,
    }


def split(train_gpus: int, inference_gpus: int) -> dict:
    return {
        "trainer.placement.colocate_all": False,
        "trainer.placement.policy_num_gpus_per_node": train_gpus,
        "generator.inference_engine.num_engines": inference_gpus,
    }


def backend_config(profile: str, num_gpus: int) -> dict:
    """The Modal profile's settings with placement sized to `num_gpus`."""
    if profile == "sync":
        return {
            **BASE_BACKEND_CONFIG,
            **colocated(num_gpus),
            "trainer.logprobs_chunk_size": 1024,
            "trainer.fused_lm_head_logprob": True,
            "generator.inference_engine.gpu_memory_utilization": 0.7,
            "generator.inference_engine.engine_init_kwargs.max_model_len": 32768,
        }
    if num_gpus < 2:
        raise SystemExit(
            f"profile '{profile}' trains and samples at the same time, so it needs "
            "separate GPUs for each: pass at least 2 with --gpus (e.g. --gpus 3,4)."
        )
    if profile == "async":
        return {
            **BASE_BACKEND_CONFIG,
            **split(1, num_gpus - 1),
            "trainer.logprobs_chunk_size": 1024,
            "trainer.fused_lm_head_logprob": True,
            "generator.inference_engine.gpu_memory_utilization": 0.85,
            "generator.inference_engine.engine_init_kwargs.max_model_len": 32768,
        }
    if profile == "textcraft":
        return {
            **BASE_BACKEND_CONFIG,
            **split(1, num_gpus - 1),
            "trainer.fused_lm_head_logprob": True,
            "generator.inference_engine.gpu_memory_utilization": 0.85,
            "generator.inference_engine.engine_init_kwargs.max_model_len": 65000,
        }
    raise SystemExit(f"unknown profile {profile!r}")


# --------------------------------------------------------------------------- helpers


def run(cmd: list[str], check: bool = True, **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, **kw)


def parse_gpus(text: str) -> list[str]:
    gpus = [g.strip() for g in text.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("--gpus needs at least one GPU index, e.g. --gpus 3 or --gpus 3,4")
    return gpus


def user_args() -> list[str]:
    """Run containers so files they write on mounts belong to you.

    Rootful Docker: run as your uid/gid. Rootless Docker (this course
    machine): container root already IS your user on the host, while
    `--user <uid>` would map to an unprivileged subuid that cannot write
    the mounted state and cache directories.
    """
    info = subprocess.run(
        ["docker", "info", "--format", "{{json .SecurityOptions}}"],
        capture_output=True, text=True,
    ).stdout
    if "rootless" in info:
        return []
    return ["--user", f"{os.getuid()}:{os.getgid()}"]


def container_name(name: str) -> str:
    return f"scottyrl-{name}"


def docker_run_args(
    name: str, gpus: list[str], host_port: int, container_port: int, extra_mounts=(), env=()
) -> list[str]:
    """`docker run` flags shared by every server container."""
    HF_CACHE_HOST.mkdir(parents=True, exist_ok=True)
    args = [
        "docker", "run", "-d", "--name", container_name(name),
        *user_args(),
        # Ray, NCCL and vLLM use shared memory heavily. (No --ulimit memlock:
        # this host's Docker daemon may not raise it, and Modal doesn't either.)
        "--ipc=host",
        "-p", f"127.0.0.1:{host_port}:{container_port}",
        "-v", f"{HF_CACHE_HOST}:/cache/huggingface",
        "-e", f"SKYRL_BASE_MODEL={MODEL_NAME}",
        # Blackwell (sm_120) fix. FlashInfer refuses to JIT for SM 12.x unless
        # CUDA >= 12.9, and the pinned stack is torch cu128, so vLLM's FlashInfer
        # top-k/top-p sampler dies at startup ("FlashInfer requires GPUs with
        # sm75 or higher"). Use vLLM's PyTorch sampler instead; logprobs and
        # attention (FLASH_ATTN) are unaffected. Harmless on H100/A100.
        "-e", "VLLM_USE_FLASHINFER_SAMPLER=0",
        # Blackwell fix #2. SkyRL forces fla's TileLang backend for Qwen3.5's
        # Gated-DeltaNet layers (FLA_TILELANG=1, tuned for Hopper); its backward
        # kernel faults with CUDA_ERROR_MISALIGNED_ADDRESS on sm_120. SkyRL's own
        # comment (skyrl/train/utils/utils.py) says to export FLA_TILELANG=0 on
        # Blackwell, which falls back to fla's Triton kernels.
        "-e", "FLA_TILELANG=0",
        "--label", "scottyrl=1",
    ]
    for token_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(token_var):
            args += ["-e", token_var]
    for gpu in gpus:
        # CDI device names: the container sees them renumbered from 0.
        args += ["--device", f"nvidia.com/gpu={gpu}"]
    for host, inside in extra_mounts:
        args += ["-v", f"{host}:{inside}"]
    for item in env:
        args += ["-e", item]
    return args


def wait_http(url: str, name: str, timeout: float) -> None:
    """Block until `url` answers 2xx, or fail with the container's last logs."""
    deadline = time.monotonic() + timeout
    print(f"waiting for {url} (up to {timeout / 60:.0f} min) ...", flush=True)
    while time.monotonic() < deadline:
        state = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_name(name)],
            capture_output=True, text=True,
        ).stdout.strip()
        if state != "true":
            subprocess.run(["docker", "logs", "--tail", "60", container_name(name)])
            raise SystemExit(f"container {container_name(name)} exited during startup")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    print(f"ready: {url}", flush=True)
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(10)
    subprocess.run(["docker", "logs", "--tail", "60", container_name(name)])
    raise SystemExit(f"{url} did not come up within {timeout:.0f}s")


# --------------------------------------------------------------------------- commands


def cmd_build(args) -> None:
    run(["docker", "build", "-t", IMAGE, str(HERE)])


def cmd_smoke(args) -> None:
    """The Modal image's import check, plus a CUDA kernel launch on each GPU."""
    gpus = parse_gpus(args.gpus)
    code = f"""
import pathlib, subprocess
# skyrl first, as in the Modal image check: importing it installs skyrl._compat,
# which makes flash-attn's FA4 module (broken against cutlass-dsl 4.6) skippable
# by megatron-core.
import skyrl
import torch, transformer_engine.pytorch, megatron.core, megatron.bridge, vllm, ray, flash_attn
assert pathlib.Path('/opt/SkyRL/pyproject.toml').is_file()
commit = subprocess.check_output(['git', '-C', '/opt/SkyRL', 'rev-parse', 'HEAD'], text=True).strip()
assert commit == {SKYRL_COMMIT!r}, commit
assert torch.cuda.is_available(), 'CUDA not visible in the container'
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    arch = f'sm_{{cap[0]}}{{cap[1]}}'
    assert arch in torch.cuda.get_arch_list(), f'torch has no kernels for {{arch}}: {{torch.cuda.get_arch_list()}}'
    a = torch.randn(1024, 1024, device=f'cuda:{{i}}', dtype=torch.bfloat16)
    torch.cuda.synchronize(i)
    assert torch.isfinite((a @ a).float()).all()
    print(f'gpu {{i}}: {{torch.cuda.get_device_name(i)}} {{arch}} matmul ok')
print(dict(skyrl_commit=commit, torch=torch.__version__, torch_cuda=torch.version.cuda,
           vllm=vllm.__version__, transformer_engine=transformer_engine.__version__,
           flash_attn=flash_attn.__version__, ray=ray.__version__))
print('SMOKE OK')
"""
    cmd = ["docker", "run", "--rm", *user_args(), "--ipc=host"]
    for gpu in gpus:
        cmd += ["--device", f"nvidia.com/gpu={gpu}"]
    run(cmd + [IMAGE, "python", "-c", code])


def cmd_up(args) -> None:
    gpus = parse_gpus(args.gpus)
    name = args.name or args.profile
    config = backend_config(args.profile, len(gpus))
    for item in args.override or []:
        key, _, value = item.partition("=")
        try:
            config[key] = json.loads(value)
        except json.JSONDecodeError:
            config[key] = value

    state = STATE_ROOT / name
    (state / "checkpoints").mkdir(parents=True, exist_ok=True)
    (state / "database").mkdir(parents=True, exist_ok=True)
    (state / "backend_config.json").write_text(json.dumps(config, indent=2))

    run(["docker", "rm", "-f", container_name(name)], check=False, capture_output=True)
    command = [
        "uv", "run", "--active", "--no-sync", "--extra", "tinker", "--extra", BACKEND,
        "-m", "skyrl.tinker.api",
        "--base-model", MODEL_NAME,
        "--backend", BACKEND,
        "--host", "0.0.0.0",
        "--port", str(CONTAINER_PORT),
        "--checkpoints-base", "/state/checkpoints",
        "--database-url", "sqlite:////state/database/tinker.db",
        "--backend-config", json.dumps(config),
    ]
    run(
        docker_run_args(name, gpus, args.port, CONTAINER_PORT, [(state, "/state")], args.env or [])
        + [IMAGE] + command
    )
    print(f"\nserver '{name}' on GPUs {','.join(gpus)} -> http://localhost:{args.port}")
    print(f"state: {state}")
    print(f"logs:  python3 {Path(__file__).relative_to(REPO)} logs {name}")
    if args.wait:
        wait_http(f"http://localhost:{args.port}/api/v1/healthz", name, args.wait)
    print(f"\nexport TINKER_BASE_URL=http://localhost:{args.port}")


def cmd_eval(args) -> None:
    """Part 5 evaluation server: vLLM with your LoRA adapter as model `a3`."""
    gpus = parse_gpus(args.gpus)
    name = args.name
    mounts = []
    adapter = args.adapter
    if Path(adapter).is_dir():
        path = Path(adapter).resolve()
        if not (path / "adapter_config.json").is_file():
            raise SystemExit(f"{path} has no adapter_config.json")
        mounts.append((path, "/adapter"))
        adapter = "/adapter"
    run(["docker", "rm", "-f", container_name(name)], check=False, capture_output=True)
    command = [
        "vllm", "serve", MODEL_NAME,
        "--host", "0.0.0.0", "--port", str(EVAL_CONTAINER_PORT),
        "--data-parallel-size", str(len(gpus)),
        "--enable-lora", "--max-lora-rank", "32",
        "--lora-modules", f"a3={adapter}",
        "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
        "--max-model-len", "65000", "--max-num-seqs", "64",
        "--max-num-batched-tokens", "8192",
        "--gpu-memory-utilization", "0.85",
        "--enable-prefix-caching", "--gdn-prefill-backend", "triton",
        "--compilation-config", '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
        "--generation-config", "vllm",
    ]
    run(docker_run_args(name, gpus, args.port, EVAL_CONTAINER_PORT, mounts) + [IMAGE] + command)
    if args.wait:
        wait_http(f"http://localhost:{args.port}/v1/models", name, args.wait)
    print(f"\nexport EVAL_SERVER=http://localhost:{args.port}")


def cmd_down(args) -> None:
    run(["docker", "rm", "-f", container_name(args.name)], check=False)


def cmd_logs(args) -> None:
    cmd = ["docker", "logs", "--tail", str(args.tail)]
    if args.follow:
        cmd.append("-f")
    run(cmd + [container_name(args.name)], check=False)


def cmd_status(args) -> None:
    run(
        ["docker", "ps", "-a", "--filter", "label=scottyrl=1",
         "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"],
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="build the SkyRL image").set_defaults(func=cmd_build)

    p = sub.add_parser("smoke", help="check the image's CUDA stack on the given GPUs")
    p.add_argument("--gpus", default="0")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("up", help="start a SkyRL Tinker server")
    p.add_argument("profile", choices=["sync", "async", "textcraft"])
    p.add_argument("--gpus", required=True, help="host GPU indices, e.g. 3 or 3,4")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--name", help="server name (container + state dir); defaults to the profile. "
                                   "Use distinct names to run two servers of one profile at once.")
    p.add_argument("--override", action="append", metavar="KEY=JSON",
                   help="override one backend config key, e.g. "
                        "generator.inference_engine.gpu_memory_utilization=0.6")
    p.add_argument("--env", action="append", metavar="KEY=VALUE",
                   help="extra environment variable for the server, e.g. CUDA_LAUNCH_BLOCKING=1")
    p.add_argument("--wait", type=float, default=45 * 60,
                   help="seconds to wait for the server to answer (0 = return immediately)")
    p.set_defaults(func=cmd_up)

    p = sub.add_parser("eval", help="serve base model + LoRA adapter with vLLM (Part 5 evaluation)")
    p.add_argument("--adapter", required=True, help="local adapter dir or HF repo id")
    p.add_argument("--gpus", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--name", default="evaluation")
    p.add_argument("--wait", type=float, default=20 * 60)
    p.set_defaults(func=cmd_eval)

    for command, func in (("down", cmd_down), ("logs", cmd_logs)):
        p = sub.add_parser(command)
        p.add_argument("name")
        if command == "logs":
            p.add_argument("-f", "--follow", action="store_true")
            p.add_argument("--tail", type=int, default=200)
        p.set_defaults(func=func)

    sub.add_parser("status", help="list scottyrl containers").set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
