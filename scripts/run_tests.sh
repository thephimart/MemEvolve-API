#!/bin/bash

# MemEvolve Test Runner Script
# Runs the complete test suite with proper configuration

set -e  # Exit on any error

echo "🧪 Running MemEvolve test suite..."

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

# Default test arguments
TEST_ARGS="src/tests/ --timeout=600 -v"

# Allow custom test arguments
if [ $# -gt 0 ]; then
    TEST_ARGS="$*"
fi

echo "Running: pytest $TEST_ARGS"
echo ""

# Run tests with coverage if pytest-cov is available
if python3 -c "import pytest_cov" 2>/dev/null; then
    echo "📊 Running tests with coverage..."
    python3 -m pytest $TEST_ARGS --cov=src --cov-report=term-missing --cov-report=html:htmlcov
    COVERAGE_EXIT_CODE=$?
else
    echo "📊 Running tests (coverage not available)..."
    python3 -m pytest $TEST_ARGS
    COVERAGE_EXIT_CODE=$?
fi

# Check exit code
if [ $COVERAGE_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
    if [ -d "htmlcov" ]; then
        echo "📈 Coverage report available at: htmlcov/index.html"
    fi
else
    echo ""
    echo "❌ Some tests failed. Exit code: $COVERAGE_EXIT_CODE"
    exit $COVERAGE_EXIT_CODE
fi