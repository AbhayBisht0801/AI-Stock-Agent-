FROM python:3.10-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first
COPY pyproject.toml uv.lock ./


# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Run application
CMD ["uv", "run", "python", "main.py"]