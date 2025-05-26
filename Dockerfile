

# Use a lightweight Alpine-based Python image
FROM python:3.12-alpine

# Set working directory
WORKDIR /app

# Install build dependencies for any compiled packages
RUN apk add --no-cache gcc musl-dev libffi-dev openssl-dev

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default environment variables (can be overridden at runtime)
ENV LISTEN_HOST=0.0.0.0
ENV LISTEN_PORT=1234

# Expose the listening port
EXPOSE 1234

# Launch the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "1234"]