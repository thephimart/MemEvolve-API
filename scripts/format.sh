#!/bin/bash

# MemEvolve Code Formatting Script
# Formats Python code according to project standards

set -e  # Exit on any error

echo "🎨 Formatting MemEvolve codebase..."

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

echo "Running autopep8..."
autopep8 --in-place --recursive --max-line-length=100 --aggressive --aggressive src/

echo "✅ Code formatting complete!"

echo ""
echo "💡 Tip: Run ./scripts/lint.sh to check if formatting meets standards."