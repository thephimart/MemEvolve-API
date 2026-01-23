#!/bin/bash

# Script to remove all evolution data
# This removes evolution state and cache files

set -e

echo "Removing evolution data..."

# Remove evolution directory
EVOLUTION_DIR="./data/evolution"
if [ -d "$EVOLUTION_DIR" ]; then
    echo "Removing directory $EVOLUTION_DIR"
    rm -rf "$EVOLUTION_DIR"
else
    echo "Evolution data directory not found: $EVOLUTION_DIR"
fi

echo "Evolution data cleanup completed."