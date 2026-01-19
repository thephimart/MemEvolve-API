#!/bin/bash

# Script to delete all log files
# This removes the logs directory and all contents

set -e

echo "Removing log files..."

LOGS_DIR="./logs"

if [ -d "$LOGS_DIR" ]; then
    echo "Removing directory $LOGS_DIR"
    rm -rf "$LOGS_DIR"
else
    echo "Logs directory not found: $LOGS_DIR"
fi

echo "Log cleanup completed."