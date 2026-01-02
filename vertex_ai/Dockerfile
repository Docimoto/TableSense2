# Dockerfile for Vertex AI Training
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "scripts/train_detector.py", "--config", "configs/detector_config_minimal.yaml"]
