# Nawal Federated Learning Server Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from nawal directory
COPY requirements.txt requirements-ml.txt* ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ -f requirements-ml.txt ]; then pip install --no-cache-dir -r requirements-ml.txt; fi

# Copy nawal code
COPY nawal/ ./nawal/
COPY pyproject.toml setup.py* ./

# Install nawal package if setup.py exists
RUN if [ -f setup.py ]; then pip install -e .; fi

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs /app/data

# Expose FL server port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FL_SERVER_ADDRESS=0.0.0.0:8080
ENV NAWAL_API_HOST=0.0.0.0
ENV NAWAL_API_PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the FL aggregator
CMD ["python", "-m", "nawal.api_server"]
