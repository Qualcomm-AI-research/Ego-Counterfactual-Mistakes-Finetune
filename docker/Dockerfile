FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

RUN apt update && \
    apt-get install --yes apt-utils && \
    DEBIAN_FRONTEND=noninteractive apt install --yes --no-install-recommends \
    ffmpeg \
    git \
    make \
    wget \
&& rm -rf /var/lib/apt/lists/*

# pip installations
COPY docker/requirements-pip.txt requirements-pip.txt
RUN pip install --no-cache-dir --upgrade -r requirements-pip.txt

# causal-conv1d provides the fused CUDA kernels the Qwen3.5 hybrid-attention
# layers import (transformers/models/qwen3_5/modeling_qwen3_5.py). No prebuilt
# wheel is published for this torch/CUDA pair, so it is compiled from source here
# rather than installed by hand in each running container.
#
# Kept after requirements-pip.txt so the extension is always compiled against
# the torch that ends up in the image, and before `COPY . /streaming_vlms` so
# editing project source does not invalidate this (expensive) layer.
ARG CAUSAL_CONV1D_REPO=https://github.com/Dao-AILab/causal-conv1d.git
ARG CAUSAL_CONV1D_REF=v1.6.2.post1
# MAX_JOBS bounds peak memory during the nvcc fan-out; raise on a bigger builder.
ARG MAX_JOBS=4
# causal-conv1d's setup.py picks -gencode targets from the *nvcc version*, not from
# the build host's GPU (it never probes the device), so this image is safe to build
# on a CPU-only machine and run on H100s. It emits sm_75/80/87 always, sm_90
# (Hopper) only when nvcc >= 11.8, and sm_100/120 when nvcc >= 12.8. The base image
# ships CUDA 12.8, which covers Hopper. Assert it, so swapping to an older CUDA tag
# cannot silently produce an image with no H100 kernels -- that failure would only
# appear at training time on the GPU host, as a "no kernel image is available"
# error, long after the build and push succeeded.
RUN nvcc --version && \
    python -c "\
import re, subprocess;\
out = subprocess.check_output(['nvcc','--version'], text=True);\
m = re.search(r'release (\d+)\.(\d+)', out);\
assert m, 'could not parse nvcc version';\
v = (int(m.group(1)), int(m.group(2)));\
assert v >= (11, 8), f'CUDA {v} is too old for sm_90; causal-conv1d would build without Hopper kernels';\
print(f'nvcc CUDA {v[0]}.{v[1]} -> causal-conv1d will emit sm_90 (Hopper) kernels')"
RUN git clone --depth 1 --branch "${CAUSAL_CONV1D_REF}" "${CAUSAL_CONV1D_REPO}" /tmp/causal-conv1d && \
    # FORCE_BUILD skips setup.py's prebuilt-wheel probe, which cannot match this
    # torch/CUDA combination and would otherwise silently fall back anyway.
    # --no-build-isolation reuses the image's torch instead of letting the
    # pyproject build backend pull a second, mismatched copy.
    CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS}" \
    pip install --no-cache-dir --no-build-isolation /tmp/causal-conv1d && \
    rm -rf /tmp/causal-conv1d

# Assert the Qwen3.5 fast path is installed, so a missing kernel package fails the
# build instead of silently degrading every run to the torch fallback ("The fast
# path is not available because one of the required library is not installed").
#
# --build restricts this to package metadata and the causal-conv1d extension:
# `docker build` has no GPU, and importing fla.ops.* here would die with
# "0 active drivers" because its module-level @triton.autotune decorators
# initialize the Triton driver at import time. The full kernel import check is
# `make check-kernels`, which runs in a GPU container.
COPY scripts/check_kernels.py /tmp/check_kernels.py
RUN python /tmp/check_kernels.py --build && rm /tmp/check_kernels.py

COPY . /streaming_vlms
WORKDIR /streaming_vlms

ENV PYTHONPATH="/streaming_vlms/src:${PYTHONPATH:-}"
