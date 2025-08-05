#!/bin/bash

# Set default values if environment variables are not set
INTERNAL_PORT=${INTERNAL_PORT:-5008}
NUM_PROCS=${NUM_PROCS:-2}
NUM_THREADS=${NUM_THREADS:-0}

# Start the Panel application
exec uv run panel serve /app/app.py \
    --address 0.0.0.0 \
    --port "$INTERNAL_PORT" \
    --allow-websocket-origin '*' \
    --num-procs "$NUM_PROCS" \
    --num-threads "$NUM_THREADS" \
    --index app \
    --show
