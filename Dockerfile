# Nawal Federated Learning Server Dockerfile
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, backtrack-free dependency resolution
RUN pip install --no-cache-dir uv

# Install PyTorch CPU first to avoid resolution conflicts
RUN uv pip install --no-cache --system torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install remaining requirements
# Strip flwr[simulation] → flwr (drops Ray ~2 GB) and wandb for production.
# Dev installs that inflate build time but aren't needed at runtime.
COPY requirements.txt ./
RUN sed \
      -e 's/flwr\[simulation\]/flwr/g' \
      -e 's/cryptography>=42.0.4,<43.0.0/cryptography>=44.0.0/g' \
      -e '/^wandb/d' \
      requirements.txt > /tmp/prod-req.txt && \
    uv pip install --no-cache --system -r /tmp/prod-req.txt

# --- Production stage ---
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user
RUN groupadd -r nawal && useradd -r -g nawal -d /app -s /sbin/nologin nawal

# Copy application code (rely on .dockerignore to exclude secrets/dev files)
COPY . .

# Set PYTHONPATH so flat-layout modules are importable
ENV PYTHONPATH=/app

# Create directories for models and logs with correct ownership
RUN mkdir -p /app/models /app/logs /app/data && \
    chown -R nawal:nawal /app

# Switch to non-root user
USER nawal

# Expose FL server port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FL_SERVER_ADDRESS=0.0.0.0:8080
ENV NAWAL_API_HOST=0.0.0.0
ENV NAWAL_API_PORT=8080
ENV NAWAL_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the FL aggregator
CMD ["python", "api_server.py", "--host", "0.0.0.0", "--port", "8080"]
