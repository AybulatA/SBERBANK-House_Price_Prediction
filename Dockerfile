# Use official Python slim image
FROM python:3.12.1-slim-bookworm

# Copy uv executables
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /code

# Add virtual environment to PATH
ENV PATH="/code/.venv/bin:$PATH"

# Copy dependency files
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies exactly as locked
RUN uv sync --locked

# Copy application code
COPY app ./app
COPY models ./models
COPY data ./data

# Expose FastAPI port
EXPOSE 9696

# Start FastAPI
ENTRYPOINT ["uvicorn", "app.predict:app", "--host", "0.0.0.0", "--port", "9696"]
