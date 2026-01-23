#!/bin/bash

# Fresh Install Cleanup Script
# WARNING: This removes ALL data, logs, and cache files
# Use only if you want to return to a completely fresh install state

set -e

echo "🧹 MemEvolve Fresh Install Cleanup"
echo "=================================="
echo ""
echo "⚠️  WARNING: This will delete ALL data!"
echo ""
echo "The following will be permanently removed:"
echo "  - All memory data (./data/)"
echo "  - All log files (./logs/)"
echo "  - All cached files (./cache/)"
echo ""
echo "This cannot be undone. Your MemEvolve installation will be"
echo "returned to a fresh state, as if newly installed."
echo ""
echo "Configuration files (.env) will be preserved."
echo ""

if ! prompt_yes_no "Are you sure you want to delete ALL data?"; then
    echo "Operation cancelled."
    exit 0
fi

if ! prompt_yes_no "This is your final confirmation. Delete ALL data?"; then
    echo "Operation cancelled."
    exit 0
fi

# Function to prompt yes/no
prompt_yes_no() {
    local prompt="$1"
    echo -n "$prompt (y/N): " >&2
    read response
    case "$response" in
        [yY]|[yY][eE][sS]) return 0 ;;
        [nN]|[nN][oO]|"") [[ "$2" == "y" ]] && return 0 || return 1 ;;
        *) echo "Please answer y or n."; prompt_yes_no "$1" "$2" ;;
    esac
}

echo ""
echo "Starting cleanup..."

# Remove data directory
if [ -d "./data" ]; then
    echo "Removing ./data/ directory..."
    rm -rf "./data"
    echo "✅ Removed data directory"
else
    echo "⚠️  Data directory not found"
fi

# Remove logs directory
if [ -d "./logs" ]; then
    echo "Removing ./logs/ directory..."
    rm -rf "./logs"
    echo "✅ Removed logs directory"
else
    echo "⚠️  Logs directory not found"
fi

# Remove cache directory
if [ -d "./cache" ]; then
    echo "Removing ./cache/ directory..."
    rm -rf "./cache"
    echo "✅ Removed cache directory"
else
    echo "⚠️  Cache directory not found"
fi

echo ""
echo "🎉 Fresh install cleanup completed!"
echo ""
echo "Your MemEvolve installation has been returned to a fresh state."
echo "You can now run ./scripts/setup.sh to reconfigure."
echo ""
echo "Note: Your .env file and virtual environment were preserved."