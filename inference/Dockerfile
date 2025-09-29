# Godseye AI Inference Service Dockerfile
# Optimized for production inference with multiple deployment targets

# Stage 1: Base environment
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as base

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

# Stage 2: GPU Inference
FROM base as gpu-inference

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.1+cu118 \
    torchvision==0.16.1+cu118 \
    torchaudio==2.1.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install inference dependencies
RUN pip install --no-cache-dir \
    ultralytics==8.0.196 \
    opencv-python==4.8.1.78 \
    numpy==1.24.3 \
    onnxruntime-gpu==1.16.3 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    pydantic==2.5.0 \
    aiofiles==23.2.1 \
    pillow==10.1.0 \
    requests==2.31.0 \
    redis==5.0.1 \
    celery==5.3.4

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/models /app/data /app/shared /app/logs

# Set permissions
RUN chmod +x /app/serve.py

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default command
CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "8001"]

# Stage 3: CPU Inference
FROM ubuntu:22.04 as cpu-inference

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

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch==2.1.1+cpu \
    torchvision==0.16.1+cpu \
    torchaudio==2.1.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install inference dependencies
RUN pip install --no-cache-dir \
    ultralytics==8.0.196 \
    opencv-python==4.8.1.78 \
    numpy==1.24.3 \
    onnxruntime==1.16.3 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    pydantic==2.5.0 \
    aiofiles==23.2.1 \
    pillow==10.1.0 \
    requests==2.31.0 \
    redis==5.0.1 \
    celery==5.3.4

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/models /app/data /app/shared /app/logs

# Set permissions
RUN chmod +x /app/serve.py

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default command
CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "8001"]

# Stage 4: Edge Inference (ARM64 for Jetson/Raspberry Pi)
FROM arm64v8/ubuntu:22.04 as edge-inference

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
    python-multipart==0.0.6 \
    pydantic==2.5.0 \
    aiofiles==23.2.1 \
    pillow==10.1.0 \
    requests==2.31.0

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/models /app/data /app/shared /app/logs

# Set permissions
RUN chmod +x /app/serve.py

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default command
CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "8001"]

# Stage 5: TensorRT Inference (for Jetson devices)
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 as tensorrt-inference

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install additional dependencies
RUN apt-get update && apt-get install -y \
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

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Install TensorRT optimized dependencies
RUN pip install --no-cache-dir \
    ultralytics==8.0.196 \
    opencv-python==4.8.1.78 \
    numpy==1.24.3 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    python-multipart==0.0.6 \
    pydantic==2.5.0 \
    aiofiles==23.2.1 \
    pillow==10.1.0 \
    requests==2.31.0

# Install TensorRT
RUN pip install --no-cache-dir \
    tensorrt==8.6.1 \
    torch-tensorrt==1.4.0

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/models /app/data /app/shared /app/logs

# Set permissions
RUN chmod +x /app/serve.py

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default command
CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "8001"]

# Default to GPU inference
FROM gpu-inference
