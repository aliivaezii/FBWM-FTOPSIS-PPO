# Stage 1: Base image with Python 3.9
FROM python:3.9-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies needed for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY src/ src/
COPY scripts/ scripts/
COPY test/ test/
COPY results/mcdm/ results/mcdm/
COPY run_all_experiments.sh .

# Create output directories
RUN mkdir -p results/experiments results/figures

# Default command: run environment validation
CMD ["python", "test/test_environment.py"]
