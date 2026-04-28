# Base image — slim keeps the image small
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install dependencies as root before switching users
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --no-dev --frozen

# Copy source code and trained model
COPY src/ src/
COPY models/ models/

# Create non-root user for security and switch to it
RUN adduser --disabled-password --no-create-home appuser
USER appuser

# Tell Docker the app listens on port 8000
EXPOSE 8000

# Start the API server
CMD [".venv/bin/uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

