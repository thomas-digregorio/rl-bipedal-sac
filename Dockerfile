# Use NVIDIA CUDA 12.8 base image
FROM nvidia/cuda:12.8.1-base-ubuntu24.04

# Set env vars to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# - git, wget: for setup
# - ffmpeg: for gym video recording
# - swig: for box2d build
# - python3-pip: fallback
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    swig \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Copy environment file
COPY environment.yml /tmp/environment.yml

# Create Conda Environment
RUN conda env create -f /tmp/environment.yml

# Set default shell to run in the env
SHELL ["conda", "run", "-n", "bipedal-sac", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app
COPY . /app

# Default command
CMD ["python", "experiments/train_custom.py"]
