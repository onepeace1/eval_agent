# Local equivalent of `glm_image` in ../generation_models.py: GLM-4.7-Flash
# needs a vLLM nightly and a Transformers revision newer than the shared image.
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_INSTALL_DIR=/opt/uv/bin \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_HTTP_TIMEOUT=300
RUN curl -LsSf https://astral.sh/uv/0.9.4/install.sh | sh \
    && /opt/uv/bin/uv venv --python 3.12 /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/uv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN uv pip install \
        "https://wheels.vllm.ai/fcdc7c2e9c9b75c923751ea7d2dd79ef66572c81/vllm-0.26.1rc1.dev224%2Bgfcdc7c2e9-cp38-abi3-manylinux_2_28_x86_64.whl" \
    && uv pip install \
        "git+https://github.com/huggingface/transformers.git@4a2e450e601214024c8a37abdaca35df2a4a3ba3" \
    && rm -rf /tmp/uv-cache && chmod -R a+rX /opt

ENV HF_XET_HIGH_PERFORMANCE=1 \
    TORCHINDUCTOR_COMPILE_THREADS=1 \
    HF_HOME=/cache/huggingface \
    VLLM_CACHE_ROOT=/cache/vllm \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache
