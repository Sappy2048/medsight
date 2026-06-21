# MedSight Production Dockerfile
# Optimized for Render deployment with FastEmbed model pre-caching

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for build and runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for persistent data and cache
RUN mkdir -p /app/data/fastembed_cache

# Set FastEmbed cache directory environment variable
ENV FASTEMBED_CACHE_DIR=/app/data/fastembed_cache

# Pre-download FastEmbed model to avoid cold start downloads
# This runs during build so the model is baked into the image
RUN python -c "import os; os.environ['FASTEMBED_CACHE_DIR'] = '/app/data/fastembed_cache'; from fastembed import TextEmbedding; print('Pre-downloading FastEmbed model...'); embedding_model = TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); print('FastEmbed model cached successfully')"

# Copy application code
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY data/icmr_chunks.json ./data/
COPY data/icmr_chunks2.json ./data/
COPY data/updated_indian_medicine_data.csv ./data/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (Render sets PORT env var)
EXPOSE 8000

# Health check to ensure app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start the application using uvicorn
# Render provides PORT environment variable
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 120"]