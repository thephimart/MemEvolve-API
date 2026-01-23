#!/bin/bash

# Script to remove all evolution data
# This removes evolution state and cache files

set -e

echo "Removing evolution data..."

# Remove evolution state file
EVOLUTION_STATE="./data/evolution_state.json"
if [ -f "$EVOLUTION_STATE" ]; then
    echo "Removing $EVOLUTION_STATE"
    rm "$EVOLUTION_STATE"
else
    echo "Evolution state file not found: $EVOLUTION_STATE"
fi

# Optionally remove entire cache directory if it only contains evolution data
# For now, just remove the specific file to be safe

echo "Evolution data cleanup completed."