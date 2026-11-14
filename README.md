# 11-768 Assignment 1: Build an Agent Harness

This repository contains the starter code and instructions for Assignment 1 for 11-768, the Carnegie Mellon University course on AI Agents. You can find the problem statement, along with instructions for setup and notes on grading in [ASSIGNMENT.md](ASSIGNMENT.md)

Authors: Weiwei Sun and Saujas Vaduguru, with feedback from 11-768 course staff (instructors Daniel Fried and Graham Neubig, and TAs Aditya Soni, Andy Liu, Apurva Gandhi, Demi Wang, Jiarui Liu, and Yueqi Song).

## Running locally with Docker (default)

Agent commands run in a local Docker container by default instead of a Modal
sandbox (`SANDBOX_BACKEND=docker`; set `SANDBOX_BACKEND=modal` for Modal). The
container is built from the same Dockerfile and pins as the Modal image, with
the SWE-ReX server added in its own Python under `/opt/swerex`, so the testbed's
`python` is untouched. Images are tagged `assignment-sandbox:<hash>` and reused
until their inputs change; containers carry the label `assignment-sandbox`
(`docker ps --filter label=assignment-sandbox`) and remove themselves after the
sandbox timeout even if the harness crashes. Forwarded ports (the chess server)
are published on `127.0.0.1` instead of a Modal tunnel.

```bash
make test-docker test-chess-docker      # real containers, free
python3 scripts/local_llm.py up --gpus 3                          # gpt-oss-120b on GPU 3
python3 scripts/local_llm.py up --gpus 3 --model Qwen/Qwen3.5-4B  # small, for quick checks
python3 scripts/local_llm.py down
```

`local_llm.py up` prints the `OPENAI_*` lines for `.env`; `make doctor` accepts
an `http://localhost` endpoint. It uses the vLLM image built by A3's
`skyrl_tinker_docker/serve.py build` unless you pass `--image`.
