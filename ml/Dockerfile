# Godseye AI ML Training Dockerfile
# Multi-stage build for optimized training and inference

# Stage 1: Base environment with CUDA support
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3.9-venv \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libgfortran5 \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Stage 2: Training environment
FROM base as training

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install additional ML libraries
RUN pip install --no-cache-dir \
    ultralytics==8.0.196 \
    opencv-python==4.8.1.78 \
    opencv-contrib-python==4.8.1.78 \
    albumentations==1.3.1 \
    imgaug==0.4.0 \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    plotly==5.17.0 \
    pandas==2.1.4 \
    numpy==1.24.3

# Install MLflow and experiment tracking
RUN pip install --no-cache-dir \
    mlflow==2.8.1 \
    wandb==0.16.1 \
    tensorboard==2.15.1

# Install video processing libraries
RUN pip install --no-cache-dir \
    ffmpeg-python==0.2.0 \
    imageio==2.31.6 \
    imageio-ffmpeg==0.4.9 \
    moviepy==1.0.3 \
    decord==0.6.0

# Install optimization libraries
RUN pip install --no-cache-dir \
    onnx==1.15.0 \
    onnxruntime-gpu==1.16.3

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/artifacts /app/shared

# Set permissions
RUN chmod +x /app/download_dataset.py /app/train.py

# Stage 3: Inference environment (optimized for deployment)
FROM base as inference

# Set working directory
WORKDIR /app

# Install minimal dependencies for inference
RUN pip install --no-cache-dir \
    torch==2.1.1+cu118 \
    torchvision==0.16.1+cu118 \
    torchaudio==2.1.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install inference-specific libraries
RUN pip install --no-cache-dir \
    ultralytics==8.0.196 \
    opencv-python==4.8.1.78 \
    numpy==1.24.3 \
    onnxruntime-gpu==1.16.3 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    python-multipart==0.0.6

# Copy only necessary files for inference
COPY models/ ./models/
COPY utils/ ./utils/
COPY inference/ ./inference/

# Create directories
RUN mkdir -p /app/models /app/data /app/shared

# Stage 4: CPU-only environment (for systems without GPU)
FROM ubuntu:22.04 as cpu-training

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3.9-venv \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libgfortran5 \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch==2.1.1+cpu \
    torchvision==0.16.1+cpu \
    torchaudio==2.1.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/data /app/models /app/logs /app/artifacts /app/shared

# Set permissions
RUN chmod +x /app/download_dataset.py /app/train.py

# Stage 5: Edge deployment (optimized for Jetson/Raspberry Pi)
FROM arm64v8/ubuntu:22.04 as edge-deployment

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libgfortran5 \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Install optimized dependencies for edge devices
RUN pip install --no-cache-dir \
    torch==2.1.1+cpu \
    torchvision==0.16.1+cpu \
    torchaudio==2.1.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install edge-optimized libraries
RUN pip install --no-cache-dir \
    ultralytics==8.0.196 \
    opencv-python==4.8.1.78 \
    numpy==1.24.3 \
    onnxruntime==1.16.3 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    python-multipart==0.0.6

# Copy only necessary files for edge deployment
COPY models/ ./models/
COPY utils/ ./utils/
COPY inference/ ./inference/
COPY edge/ ./edge/

# Create directories
RUN mkdir -p /app/models /app/data /app/shared

# Set permissions
RUN chmod +x /app/edge/start_edge_service.sh

# Default to training stage
FROM training
