#!/bin/bash
# Wrapper for NVIDIA Jetson / other ARM boards where PyTorch's OpenMP runtime
# hits a libgomp TLS allocation issue unless it's preloaded explicitly.
# Not needed on x86_64 - run the script directly there.

LIBGOMP="/usr/lib/aarch64-linux-gnu/libgomp.so.1"

if [ ! -f "$LIBGOMP" ]; then
    echo "Warning: $LIBGOMP not found - running without LD_PRELOAD." >&2
    exec python3 "$@"
fi

exec env LD_PRELOAD="$LIBGOMP" python3 "$@"
