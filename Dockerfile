FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

#COPY sources.list /etc/apt/sources.list
ARG DEBIAN_FRONTEND=noninteractive

# Base tools and repo setup
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    build-essential \
    curl \
    ca-certificates \
    cmake \
    vim \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
 && rm -rf /var/lib/apt/lists/*

# Create a dedicated Python 3.10 venv and make it default on PATH
RUN python3.10 -m venv /opt/py310 \
 && /opt/py310/bin/python -m pip install --upgrade pip setuptools wheel

ENV VIRTUAL_ENV=/opt/py310
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PIP_NO_CACHE_DIR=1

# Sanity check
RUN python -V && pip -V

RUN pip config set global.extra-index-url "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
RUN pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install onnxruntime-gpu==1.16.0 onnx==1.14.1

# install app
WORKDIR /workspace
ADD . Dipoorlet 
RUN cd Dipoorlet \
    && pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    && python3 setup.py install 
