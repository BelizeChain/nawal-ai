# Nawal Federated Learning Server Dockerfile
# Supports CPU and GPU builds via COMPUTE build arg.
#   docker build --build-arg COMPUTE=cpu -t nawal:cpu .
#   docker build --build-arg COMPUTE=gpu -t nawal:gpu .

ARG COMPUTE=cpu

# GPU: NVIDIA CUDA 12.4 + Python 3.11 runtime (includes cuDNN)
# CPU: slim Python (no CUDA overhead)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base-gpu
FROM python:3.11.11-slim                          AS base-cpu
FROM base-${COMPUTE}                              AS base

# GPU images don't ship Python — install 3.11 from deadsnakes PPA
ARG COMPUTE
RUN if [ "$COMPUTE" = "gpu" ]; then \
      apt-get update && \
      apt-get install -y --no-install-recommends software-properties-common gpg-agent && \
      add-apt-repository -y ppa:deadsnakes/ppa && \
      apt-get update && \
      apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3.11-distutils && \
      update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
      update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
      curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
      rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

# Install system + runtime dependencies in one layer, then clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Install PyTorch — GPU gets CUDA 12.4 variant, CPU gets lightweight wheel
ARG COMPUTE
RUN if [ "$COMPUTE" = "gpu" ]; then \
      uv pip install --no-cache --system \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu124; \
    else \
      uv pip install --no-cache --system \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Copy and install remaining requirements
# Strip flwr[simulation] → flwr (drops Ray ~2 GB) and wandb for production.
COPY requirements.txt ./
RUN sed \
      -e 's/flwr\[simulation\]/flwr/g' \
      -e 's/cryptography>=42.0.4,<43.0.0/cryptography>=44.0.0,<45.0.0/g' \
      -e '/^wandb/d' \
      requirements.txt > /tmp/prod-req.txt && \
    uv pip install --no-cache --system -r /tmp/prod-req.txt && \
    rm -f /tmp/prod-req.txt

# Remove build tools no longer needed at runtime (shrinks image)
RUN apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r nawal && useradd -r -g nawal -d /app -s /sbin/nologin nawal

# Copy application code (rely on .dockerignore to exclude secrets/dev files)
COPY . .

# Set PYTHONPATH so flat-layout modules are importable
ENV PYTHONPATH=/app

# Create directories matching config.prod.yaml storage paths
RUN mkdir -p /app/models /app/logs /app/data \
             /var/lib/nawal/checkpoints /var/log/nawal /var/lib/nawal/data && \
    chown -R nawal:nawal /app

# Switch to non-root user
USER nawal

# Expose FL server port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FL_SERVER_ADDRESS=0.0.0.0:8080
ENV NAWAL_API_HOST=0.0.0.0
ENV NAWAL_PORT=8080
ENV NAWAL_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the FL aggregator
CMD ["python", "api_server.py"]
