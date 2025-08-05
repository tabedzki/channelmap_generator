FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
   git \
   build-essential \
   curl \
   && rm -rf /var/lib/apt/lists/*
# Set the working directory
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/root/.cache/uv

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the project files to the working directory

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-dev --frozen --no-install-workspace


COPY . /app

# Install the Python packages using uv
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev --frozen
ENV PATH="/app/.venv/bin:$PATH"

# Expose the port
ENV INTERNAL_PORT=5008
ENV NUM_PROCS=2
ENV NUM_THREADS=0
EXPOSE ${INTERNAL_PORT}

# Copy the Dockerfile to /dockerfile within the container
COPY Dockerfile /Dockerfile


# Health check
HEALTHCHECK CMD curl --fail http://localhost:${INTERNAL_PORT}/

# Run the Panel app
CMD ["sh", "-c", "uv run panel serve /app/app.py --address 0.0.0.0 --port $INTERNAL_PORT --allow-websocket-origin '*' --num-procs $NUM_PROCS --num-threads $NUM_THREADS --index app --show"]

