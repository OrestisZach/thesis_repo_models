# ---------------------------------------------------------------------------
# refer-lidar Docker image (Modernized for L40S & CUDA 12, PyTorch 2.4)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ── System packages ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        ca-certificates \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python 3.10 via Miniforge ─────────────────────────────────────────────
ENV PATH="/opt/conda/bin:${PATH}"
RUN curl -sSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o miniforge.sh \
    && bash miniforge.sh -b -p /opt/conda \
    && rm miniforge.sh \
    && /opt/conda/bin/conda install -y python=3.10 \
    && /opt/conda/bin/conda clean -ya

RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja

# ── Core Dependencies ─────────────────────────────────────────────────────
# PyTorch 2.4 with CUDA 12.4 (Required for torchtext C++ backend)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.0+cu124 \
    torchvision==0.19.0+cu124

# Spconv for sparse convolutions
RUN pip install --no-cache-dir spconv-cu124==2.3.8

# Torchtext 0.18.0 (Final stable release, compatible with PT 2.4)
RUN pip install --no-cache-dir torchtext==0.18.0 \
    && sed -i 's/_init_extension()/pass/g' /opt/conda/lib/python3.10/site-packages/torchtext/_extension.py

WORKDIR /workspace

# ── Install PointPillars package ──────────────────────────────────────────
COPY pointpillar-repo/ /tmp/pointpillar-repo/
# Explicitly target Ada Lovelace (L40S)
ENV TORCH_CUDA_ARCH_LIST="8.9"
RUN cd /tmp/pointpillar-repo \
    && sed -i -E '/^(torch|torchvision|torchaudio|numpy|setuptools)/d' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt \
    && python setup.py build_ext --inplace \
    && pip install --no-cache-dir --no-build-isolation -e . \
    && cd /workspace

# ── Project Python dependencies ───────────────────────────────────────────
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Overrides based on your local pip list
RUN pip install --no-cache-dir \
    einops \
    tensorboardX \
    transformers==4.57.6 \
    nuscenes-devkit>=1.1.9 \
    pyquaternion>=0.9.9 \
    numba==0.59.0 \
    numpy==1.26.4

# ── Build MultiScaleDeformableAttention CUDA op ───────────────────────────
COPY models/ops/ /workspace/models/ops/
ENV FORCE_CUDA=1
RUN cd /workspace/models/ops \
    # 1. Patch old PyTorch 1.x C++ API calls to PyTorch 2.x standards
    && find src -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.h" \) -exec sed -i 's/\.type()\.is_cuda()/.is_cuda()/g' {} + \
    && find src -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.h" \) -exec sed -i 's/\.type()/.scalar_type()/g' {} + \
    # 2. Use modern pip install without build isolation
    && pip install --no-cache-dir --no-build-isolation . \
    && cd /workspace

# ── Copy project source ───────────────────────────────────────────────────
COPY . /workspace/

CMD ["bash"]