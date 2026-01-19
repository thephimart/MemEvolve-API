#!/bin/bash

# Script to remove all memory data
# This removes memory storage files and directories

set -e

echo "Removing memory data..."

# Default memory storage path
MEMORY_PATH="./data/memory"

if [ -d "$MEMORY_PATH" ]; then
    echo "Removing directory $MEMORY_PATH"
    rm -rf "$MEMORY_PATH"
else
    echo "Memory data directory not found: $MEMORY_PATH"
fi

# Also check for other possible memory storage locations based on config
# But for now, just the default

echo "Memory data cleanup completed."