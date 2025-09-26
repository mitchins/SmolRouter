

# Multi-stage build for smaller production image
FROM ghcr.io/astral-sh/uv:python3.12-alpine AS builder

# Set working directory
WORKDIR /app

# Install build dependencies for any compiled packages
RUN apk add --no-cache gcc musl-dev libffi-dev openssl-dev

# Copy all necessary files for building
COPY pyproject.toml uv.lock ./
COPY smolrouter/ ./smolrouter/

# Install dependencies using uv
RUN uv sync --no-dev --locked

# Production stage
FROM python:3.12-alpine AS runtime

# Set working directory
WORKDIR /app

# Copy the virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code (excluding secrets and crud)
COPY smolrouter/ ./smolrouter/
COPY templates/ ./templates/
COPY pyproject.toml ./

# Create volume mount point for configuration
RUN mkdir -p /app/config

# Create non-root user for security
RUN addgroup -g 1000 smolrouter && \
    adduser -u 1000 -G smolrouter -s /bin/sh -D smolrouter && \
    chown -R smolrouter:smolrouter /app

USER smolrouter

# Default environment variables (can be overridden at runtime)
ENV LISTEN_HOST=0.0.0.0
ENV LISTEN_PORT=8088
ENV PATH="/app/.venv/bin:$PATH"
ENV ROUTES_CONFIG=/app/config/routes.yaml

# Expose the listening port
EXPOSE 8088

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8088/health')" || exit 1

# Launch the FastAPI app with Uvicorn
CMD ["python", "-m", "smolrouter"]