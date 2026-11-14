# Local-GPU equivalent of serving.base_image() in ../serving.py: same CUDA base,
# Python 3.12 and pinned vLLM / huggingface-hub / flashinfer versions, so the
# validator and generation models are served here exactly as on Modal.
#
#     python3 infrastructure/docker/serve.py build
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_INSTALL_DIR=/opt/uv/bin \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache
RUN curl -LsSf https://astral.sh/uv/0.9.4/install.sh | sh \
    && /opt/uv/bin/uv venv --python 3.12 /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/uv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN uv pip install \
        "vllm==0.13.0" \
        "huggingface-hub==0.36.0" \
        "flashinfer-python==0.5.3" \
    && rm -rf /tmp/uv-cache && chmod -R a+rX /opt

# Not in the Modal image: mistral-common is unpinned there, and every release
# since 1.9 needs huggingface_hub>=1.0, which the huggingface-hub==0.36.0 pin
# rules out, so Ministral fails to load its tokenizer. 1.8.8 is the last release
# from vLLM 0.13's time.
RUN uv pip install "mistral-common[image]==1.8.8" \
    && rm -rf /tmp/uv-cache && chmod -R a+rX /opt

# serving.ENVIRONMENT, plus writable scratch for running as the host user.
ENV HF_XET_HIGH_PERFORMANCE=1 \
    TORCHINDUCTOR_COMPILE_THREADS=1 \
    HF_HOME=/cache/huggingface \
    VLLM_CACHE_ROOT=/cache/vllm \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache
