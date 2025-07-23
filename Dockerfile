# Multi-stage build for Doser Responser
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim

# Install runtime dependencies (including curl for health check)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy application code first
COPY --chown=app:app . .

# Copy Python packages from builder stage to the app user's directory
COPY --from=builder --chown=app:app /root/.local /home/app/.local

# Switch to non-root user
USER app

# Make sure scripts in .local are usable
ENV PATH=/home/app/.local/bin:$PATH

# Add the current directory to Python path so imports work correctly
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7090/_dash-layout || exit 1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:7090", "--workers", "1", "--timeout", "300", "wsgi:server"]
