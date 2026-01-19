#!/bin/bash

# MemEvolve Development Environment Setup Script
# This script sets up the development environment for MemEvolve

set -e  # Exit on any error

echo "🚀 Setting up MemEvolve development environment..."

# Check if Python 3.12+ is available
if ! python3 --version >/dev/null 2>&1; then
    echo "❌ Python 3 is not installed. Please install Python 3.12 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.12"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION detected. MemEvolve requires Python $REQUIRED_VERSION or higher."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "📚 Installing dependencies..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "⚠️  requirements.txt not found. Please ensure dependencies are installed manually."
fi

# Install development dependencies if dev_requirements.txt exists
if [ -f "dev_requirements.txt" ]; then
    echo "🔧 Installing development dependencies..."
    pip install -r dev_requirements.txt
    echo "✅ Development dependencies installed"
fi

# Run basic checks
echo "🔍 Running basic checks..."

# Check if basic imports work
python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from memory_system import MemorySystem
    from components.encode import ExperienceEncoder
    from components.store import StorageBackend
    from components.retrieve import RetrievalStrategy
    from components.manage import ManagementStrategy
    print('✅ Core imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

# Check if tests can be discovered
if python3 -m pytest --collect-only -q src/tests/ >/dev/null 2>&1; then
    echo "✅ Test discovery successful"
else
    echo "⚠️  Test discovery failed - tests may not run correctly"
fi

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "To activate the environment in future sessions:"
echo "  source .venv/bin/activate"
echo ""
echo "Available commands:"
echo "  ./scripts/run_tests.sh    - Run the test suite"
echo "  ./scripts/lint.sh         - Run code linting"
echo "  ./scripts/format.sh       - Format code"
echo "  pytest src/tests/          - Run tests manually"
echo "  flake8 src/               - Run linting manually"