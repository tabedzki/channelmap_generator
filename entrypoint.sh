#!/bin/bash

# Set default values if environment variables are not set
INTERNAL_PORT=${INTERNAL_PORT:-5006}
NUM_PROCS=${NUM_PROCS:-1}
ADDRESS=${ADDRESS:-localhost}
ALLOW_WEBSOCKET_ORIGIN=${ALLOW_WEBSOCKET_ORIGIN:-localhost:$INTERNAL_PORT}

echo "$(date '+%Y-%m-%d %H:%M:%S') INTERNAL_PORT $INTERNAL_PORT"
echo "$(date '+%Y-%m-%d %H:%M:%S') ADDRESS $ADDRESS"
echo "$(date '+%Y-%m-%d %H:%M:%S') ALLOW_WEBSOCKET_ORIGIN $ALLOW_WEBSOCKET_ORIGIN"
echo "$(date '+%Y-%m-%d %H:%M:%S') NUM_PROCS $NUM_PROCS"


# Start the Panel application
exec uv run panel serve ./app.py \
    --address "$ADDRESS" \
    --port "$INTERNAL_PORT" \
    --allow-websocket-origin "$ALLOW_WEBSOCKET_ORIGIN" \
    --num-procs "$NUM_PROCS" \
    --show
