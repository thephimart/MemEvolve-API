#!/bin/bash

# MemEvolve Code Linting Script
# Runs code quality checks on the codebase

set -e  # Exit on any error

echo "🔍 Running MemEvolve code linting..."

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Not in a virtual environment. Activating .venv..."
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo "❌ Virtual environment not found. Run ./scripts/setup.sh first."
        exit 1
    fi
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:src"

echo "Running flake8..."
flake8 src/ --max-line-length=100 --extend-ignore=E203,W503

FLAKE8_EXIT_CODE=$?

if [ $FLAKE8_EXIT_CODE -eq 0 ]; then
    echo "✅ Linting passed!"
else
    echo "❌ Linting failed. Please fix the issues above."
    exit $FLAKE8_EXIT_CODE
fi

echo ""
echo "🎯 Linting complete!"