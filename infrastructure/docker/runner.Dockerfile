# Local equivalent of the `image` in ../agent_runner.py (Modal's debian_slim
# with Python 3.12, git, and the same pinned packages), plus `modal` itself so
# agent_runner.py and generation_gateway.py import unchanged, and `requests`
# for the gateway. The infrastructure directory is mounted at run time, so
# edits to those files need no rebuild.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "mini-swe-agent==2.4.5" \
        "matplotlib==3.10.5" \
        "numpy==2.3.2" \
        "pandas==2.3.1" \
        "pillow==11.3.0" \
        "scipy==1.16.1" \
        "seaborn==0.13.2" \
        "requests==2.32.4" \
        "modal==1.5.5"

ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
