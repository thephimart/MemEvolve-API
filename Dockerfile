# MemEvolve API Server Dockerfile

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Set environment variables
ENV MEMEVOLVE_API_HOST=0.0.0.0
ENV MEMEVOLVE_API_PORT=8001
ENV MEMEVOLVE_STORAGE_PATH=/app/data/memory.json
ENV MEMEVOLVE_LOG_LEVEL=INFO

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the API server
CMD ["python", "scripts/start_api.py"]