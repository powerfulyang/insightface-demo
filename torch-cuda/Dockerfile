FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# 设置非交互模式
ENV DEBIAN_FRONTEND=noninteractive

# 安装 Python venv 依赖
RUN apt-get update && apt-get install -y \
    python3-venv python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建虚拟环境
RUN python3 -m venv /opt/venv

# 让 pip、python 默认走虚拟环境
ENV PATH="/opt/venv/bin:$PATH"

# 升级 pip
RUN pip install --upgrade pip setuptools wheel

# 安装 torch
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

