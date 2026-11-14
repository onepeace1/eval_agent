# Local-GPU equivalent of the Modal image in ../skyrl_tinker_modal/skyrl_server.py.
#
# Same base image, same uv version, same pinned SkyRL fork and extras, so the
# server you run here is the one the Modal deployments run. Build it with
#
#     python skyrl_tinker_docker/serve.py build
#
# The GPU import check that the Modal build runs is `serve.py smoke` here,
# because `docker build` has no GPU access.

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG SKYRL_GIT_URL=https://github.com/adityasoni9998/SkyRL.git
ARG SKYRL_COMMIT=491e1329c314ef43818fb6824d9f2ae61883416c
ARG BACKEND=megatron

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libnuma-dev numactl \
    && rm -rf /var/lib/apt/lists/*

# uv and its managed Python live under /opt so the container can run as the
# host user (see serve.py), not only as root.
ENV UV_INSTALL_DIR=/opt/uv/bin \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON=3.12 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/skyrl-venv \
    UV_CACHE_DIR=/tmp/uv-cache
RUN curl -LsSf https://astral.sh/uv/0.9.4/install.sh | sh \
    && /opt/uv/bin/uv python install 3.12

ENV PATH=/opt/skyrl-venv/bin:/opt/uv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook \
    SKYRL_REPO_ROOT=/opt/SkyRL

# Fetch only the pinned commit and verify it, exactly as the Modal image does.
RUN git init /opt/SkyRL \
    && git -C /opt/SkyRL remote add origin ${SKYRL_GIT_URL} \
    && git -C /opt/SkyRL fetch --depth=1 origin ${SKYRL_COMMIT} \
    && git -C /opt/SkyRL checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/SkyRL rev-parse HEAD)" = "${SKYRL_COMMIT}"

WORKDIR /opt/SkyRL
RUN uv sync --extra tinker --extra ${BACKEND} \
    && rm -rf /tmp/uv-cache \
    && chmod -R a+rX /opt \
    && git config --system --add safe.directory /opt/SkyRL

# Writable scratch for a non-root user: Ray, Triton, torch.compile and vLLM
# all write under HOME or /tmp.
ENV HOME=/tmp USER=skyrl XDG_CACHE_HOME=/tmp/.cache
