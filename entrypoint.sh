#!/bin/bash

# Set default values if environment variables are not set
INTERNAL_PORT=${INTERNAL_PORT:-5006}
NUM_PROCS=${NUM_PROCS:-1}
NUM_THREADS=${NUM_THREADS:-2}
ADDRESS=${ADDRESS:-localhost}
ALLOW_WEBSOCKET_ORIGIN=${ALLOW_WEBSOCKET_ORIGIN:-localhost:$INTERNAL_PORT}
# ALLOW_WEBSOCKET_ORIGIN=${ALLOW_WEBSOCKET_ORIGIN:-''}

echo "$(date '+%Y-%m-%d %H:%M:%S') INTERNAL_PORT $INTERNAL_PORT"
echo "$(date '+%Y-%m-%d %H:%M:%S') ADDRESS $ADDRESS"
echo "$(date '+%Y-%m-%d %H:%M:%S') ALLOW_WEBSOCKET_ORIGIN $ALLOW_WEBSOCKET_ORIGIN"
echo "$(date '+%Y-%m-%d %H:%M:%S') NUM_PROCS $NUM_PROCS"
echo "$(date '+%Y-%m-%d %H:%M:%S') NUM_THREADS $NUM_THREADS"


# Start the Panel application
exec uv run panel serve ./app.py \
    --address "$ADDRESS" \
    --port "$INTERNAL_PORT" \
    --allow-websocket-origin "$ALLOW_WEBSOCKET_ORIGIN" \
    --num-procs "$NUM_PROCS" \
    --num-threads "$NUM_THREADS" \
    --show
