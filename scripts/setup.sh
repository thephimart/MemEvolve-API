#!/bin/bash

# MemEvolve One-Click Setup Script
# Interactive setup with configuration prompts
#
# ⚠️  WARNING: This script is UNTESTED - use at your own risk!
# It has been developed but not thoroughly tested. Please report any issues.

set -e  # Exit on any error

# Check if running interactively
if [ ! -t 0 ] || [ ! -t 1 ]; then
    echo "❌ This script requires an interactive terminal with visible output."
    echo "Please run it directly in a terminal: ./scripts/setup.sh"
    exit 1
fi

echo "🚀 MemEvolve One-Click Setup"
echo "============================"
echo ""

# Function to prompt user for yes/no
prompt_yes_no() {
    local prompt="$1"
    local default="${2:-n}"
    local response

    read -p "$prompt (y/N): " response
    case "$response" in
        [yY]|[yY][eE][sS]) return 0 ;;
        [nN]|[nN][oO]|"") [[ "$default" == "y" ]] && return 0 || return 1 ;;
        *) echo "Please answer y or n."; prompt_yes_no "$prompt" "$default" ;;
    esac
}

# Function to prompt for input with default
prompt_input() {
    local prompt="$1"
    local default="$2"
    local response

    echo -n "$prompt [$default]: " >&2
    read response
    echo "${response:-$default}"
}

# Function to select model from API or allow custom input
# Returns: model name via echo, return code 0 if auto-detected, 1 if manual/custom
select_model() {
    local base_url="$1"

    echo "🔍 Checking for available models at $base_url/v1/models..." >&2
    if command -v curl >/dev/null 2>&1; then
        local models_response
        models_response=$(curl -s --max-time 10 "$base_url/v1/models" 2>/dev/null)
        if [ $? -eq 0 ]; then
            echo "✅ Connected to API successfully" >&2
            # Try to parse models using Python
            local models_output
            models_output=$(python3 -c "
import json
import sys
data = json.load(sys.stdin)
models = [m['id'] for m in data.get('data', []) if 'id' in m]
if models:
    print('\n'.join(models))
    sys.exit(0)
sys.exit(1)
" <<< "$models_response" 2>/dev/null)

            if [ $? -eq 0 ] && [ -n "$models_output" ]; then
                # Parse the output into array
                local models=()
                while IFS= read -r line; do
                    models+=("$line")
                done <<< "$models_output"

                if [ ${#models[@]} -gt 0 ]; then
                    echo "✅ Found ${#models[@]} models:" >&2
                    local i=1
                    for model in "${models[@]}"; do
                        echo "  $i) $model" >&2
                        ((i++))
                    done
                    echo "  0) Enter custom model name" >&2
                    echo "" >&2

                    local choice
                    echo -n "Select model (1-${#models[@]} or 0): " >&2
                    read choice

                    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#models[@]} ]; then
                        echo "${models[$((choice-1))]}"
                        return 0  # Auto-detected
                    elif [ "$choice" -eq 0 ]; then
                        echo -n "Enter custom model name []: " >&2
                        read custom_model
                        echo "${custom_model:-""}"
                        return 1  # Manual
                    else
                        echo "⚠️ Invalid choice, proceeding with manual input" >&2
                    fi
                else
                    echo "⚠️ API responded but no models found, using manual input" >&2
                fi
            else
                echo "⚠️ API responded but response format unexpected, using manual input" >&2
            fi
        else
            echo "⚠️ Could not connect to API, using manual input" >&2
        fi
    else
        echo "⚠️ curl not available, using manual input" >&2
    fi

    # Fallback to manual input
    local manual_model=$(prompt_input "Model name" "")
    echo "$manual_model"
    return 1  # Manual
}

# Check Python version
echo "🐍 Checking Python version..."
if ! python3 --version >/dev/null 2>&1; then
    echo "❌ Python 3 is not installed. Please install Python 3.12 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.12"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "⚠️  Python $PYTHON_VERSION detected. MemEvolve recommends Python $REQUIRED_VERSION or higher."
    if ! prompt_yes_no "Continue anyway?"; then
        echo "Setup cancelled."
        exit 1
    fi
else
    echo "✅ Python $PYTHON_VERSION detected"
fi

# Warn about virtual environment
echo ""
echo "📦 Virtual Environment Setup"
echo "-----------------------------"
echo "This script will create and activate a virtual environment in ./.venv"
echo "If you prefer conda, uv, or other Python managers, you must be advanced enough to not use this script!"
echo ""
if ! prompt_yes_no "Continue with venv setup?"; then
    echo "Setup cancelled. Please use advanced setup methods."
    exit 1
fi

# Create virtual environment
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

# Upgrade pip and core packages
echo "⬆️  Upgrading pip and core packages..."
pip install --upgrade pip setuptools wheel

# Configuration prompts
echo ""
echo "⚙️  API Configuration"
echo "===================="

# Upstream API (Required)
echo ""
echo "🔗 Upstream API (REQUIRED - Your primary LLM service)"
UPSTREAM_BASE_URL=$(prompt_input "Upstream API Base URL" "http://localhost:11434")
UPSTREAM_API_KEY=$(prompt_input "Upstream API Key (optional for local)" "")

if [[ "$UPSTREAM_BASE_URL" == http*localhost* ]] || [[ "$UPSTREAM_BASE_URL" == http*127.0.0.1* ]]; then
    UPSTREAM_API_KEY=${UPSTREAM_API_KEY:-""}
else
    UPSTREAM_API_KEY=${UPSTREAM_API_KEY:-$UPSTREAM_API_KEY}
fi

UPSTREAM_MODEL=$(select_model "$UPSTREAM_BASE_URL")
if [ $? -eq 0 ]; then
    UPSTREAM_AUTO_RESOLVE="true"
else
    UPSTREAM_AUTO_RESOLVE="false"
fi

UPSTREAM_TIMEOUT=$(prompt_input "Upstream Timeout (seconds)" "720")

# Embedding API (Optional)
echo ""
echo "🔗 Embedding API (OPTIONAL - for vector storage)"
if prompt_yes_no "Configure separate embedding API?"; then
    EMBEDDING_BASE_URL=$(prompt_input "Embedding API Base URL" "http://localhost:11435")
    EMBEDDING_API_KEY=$(prompt_input "Embedding API Key (optional for local)" "")

    EMBEDDING_MODEL=$(select_model "$EMBEDDING_BASE_URL")
    if [ $? -eq 0 ]; then
        EMBEDDING_AUTO_RESOLVE="true"
    else
        EMBEDDING_AUTO_RESOLVE="false"
    fi

    EMBEDDING_TIMEOUT=$(prompt_input "Embedding Timeout (seconds)" "30")
else
    EMBEDDING_BASE_URL=""
    EMBEDDING_API_KEY=""
    EMBEDDING_MODEL=""
    EMBEDDING_AUTO_RESOLVE="false"
    EMBEDDING_TIMEOUT="30"
fi

# Memory API (Optional)
echo ""
echo "🔗 Memory API (OPTIONAL - dedicated LLM for memory encoding)"
if prompt_yes_no "Configure separate memory API?"; then
    MEMORY_BASE_URL=$(prompt_input "Memory API Base URL" "http://localhost:11433")
    MEMORY_API_KEY=$(prompt_input "Memory API Key (optional for local)" "")

    MEMORY_MODEL=$(select_model "$MEMORY_BASE_URL")
    if [ $? -eq 0 ]; then
        MEMORY_AUTO_RESOLVE="true"
    else
        MEMORY_AUTO_RESOLVE="false"
    fi

    MEMORY_TIMEOUT=$(prompt_input "Memory Timeout (seconds)" "60")
else
    MEMORY_BASE_URL=""
    MEMORY_API_KEY=""
    MEMORY_MODEL=""
    MEMORY_AUTO_RESOLVE="false"
    MEMORY_TIMEOUT="60"
fi

# Storage Backend
echo ""
echo "💾 Storage Backend"
echo "Choose your memory storage backend:"
echo "  json   - Simple file-based storage (recommended for getting started)"
echo "  vector - Vector embeddings with FAISS (requires additional setup)"
echo "  graph  - Graph-based storage with Neo4j (requires Neo4j database)"
STORAGE_BACKEND=$(prompt_input "Storage Backend (json/vector/graph)" "json")

# Evolution System
echo ""
echo "🧬 Evolution System"
echo "Evolution automatically optimizes memory architectures over time using genetic algorithms."
echo "This improves performance but uses more resources and may take time to show benefits."
if prompt_yes_no "Enable evolution system?"; then
    ENABLE_EVOLUTION="true"
else
    ENABLE_EVOLUTION="false"
fi

# Logging
echo ""
echo "📝 Logging Configuration"
echo "API server logging: Essential for debugging API issues (recommended ON)"
echo "Middleware logging: Tracks memory operations (OFF by default to reduce noise)"
echo "Memory logging: Detailed memory system operations (OFF by default)"
echo "Experiment logging: Evolution system progress (OFF by default)"
ENABLE_API_LOG="false"
ENABLE_MIDDLEWARE_LOG="false"
ENABLE_MEMORY_LOG="false"
ENABLE_EXPERIMENT_LOG="false"

if prompt_yes_no "Enable API server logging?" "y"; then
    ENABLE_API_LOG="true"
fi

if prompt_yes_no "Enable middleware logging?"; then
    ENABLE_MIDDLEWARE_LOG="true"
fi

if prompt_yes_no "Enable memory logging?"; then
    ENABLE_MEMORY_LOG="true"
fi

if prompt_yes_no "Enable experiment logging?"; then
    ENABLE_EXPERIMENT_LOG="true"
fi

# Create .env file
echo ""
echo "📄 Creating .env configuration file..."

cat > .env << EOF
# MemEvolve Environment Configuration
# Generated by setup.sh on $(date)

# =============================================================================
# API ENDPOINTS - LLM Services Configuration
# =============================================================================

# Upstream LLM API (REQUIRED)
MEMEVOLVE_UPSTREAM_BASE_URL=$UPSTREAM_BASE_URL
MEMEVOLVE_UPSTREAM_API_KEY=$UPSTREAM_API_KEY
MEMEVOLVE_UPSTREAM_MODEL=$UPSTREAM_MODEL
MEMEVOLVE_UPSTREAM_AUTO_RESOLVE_MODELS=$UPSTREAM_AUTO_RESOLVE
MEMEVOLVE_UPSTREAM_TIMEOUT=$UPSTREAM_TIMEOUT

# Embedding API
MEMEVOLVE_EMBEDDING_BASE_URL=$EMBEDDING_BASE_URL
MEMEVOLVE_EMBEDDING_API_KEY=$EMBEDDING_API_KEY
MEMEVOLVE_EMBEDDING_MODEL=$EMBEDDING_MODEL
MEMEVOLVE_EMBEDDING_AUTO_RESOLVE_MODELS=$EMBEDDING_AUTO_RESOLVE
MEMEVOLVE_EMBEDDING_TIMEOUT=$EMBEDDING_TIMEOUT

# Memory LLM API
MEMEVOLVE_MEMORY_BASE_URL=$MEMORY_BASE_URL
MEMEVOLVE_MEMORY_API_KEY=$MEMORY_API_KEY
MEMEVOLVE_MEMORY_MODEL=$MEMORY_MODEL
MEMEVOLVE_MEMORY_AUTO_RESOLVE_MODELS=$MEMORY_AUTO_RESOLVE
MEMEVOLVE_MEMORY_TIMEOUT=$MEMORY_TIMEOUT

# =============================================================================
# GLOBAL API SETTINGS
# =============================================================================
MEMEVOLVE_API_MAX_RETRIES=3
MEMEVOLVE_DEFAULT_TOP_K=5

# =============================================================================
# API SERVER CONFIGURATION
# =============================================================================
MEMEVOLVE_API_ENABLE=true
MEMEVOLVE_API_HOST=127.0.0.1
MEMEVOLVE_API_PORT=11436
MEMEVOLVE_API_MEMORY_INTEGRATION=true

# =============================================================================
# STORAGE & DATA MANAGEMENT
# =============================================================================
MEMEVOLVE_DATA_DIR=./data
MEMEVOLVE_CACHE_DIR=./cache
MEMEVOLVE_LOGS_DIR=./logs
MEMEVOLVE_STORAGE_BACKEND_TYPE=$STORAGE_BACKEND
MEMEVOLVE_STORAGE_INDEX_TYPE=flat

# =============================================================================
# MEMORY SYSTEM BEHAVIOR
# =============================================================================
MEMEVOLVE_RETRIEVAL_STRATEGY_TYPE=hybrid
MEMEVOLVE_RETRIEVAL_SEMANTIC_WEIGHT=0.7
MEMEVOLVE_RETRIEVAL_KEYWORD_WEIGHT=0.3
MEMEVOLVE_RETRIEVAL_ENABLE_CACHING=true
MEMEVOLVE_RETRIEVAL_CACHE_SIZE=1024
MEMEVOLVE_MANAGEMENT_ENABLE_AUTO_MANAGEMENT=true
MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD=1024
MEMEVOLVE_MANAGEMENT_AUTO_CONSOLIDATE_INTERVAL=128
MEMEVOLVE_MANAGEMENT_DEDUPLICATE_THRESHOLD=0.9
MEMEVOLVE_MANAGEMENT_FORGETTING_STRATEGY=lru
MEMEVOLVE_MANAGEMENT_MAX_MEMORY_AGE_DAYS=365
MEMEVOLVE_ENCODER_ENCODING_STRATEGIES=lesson,skill
MEMEVOLVE_ENCODER_ENABLE_ABSTRACTION=true
MEMEVOLVE_ENCODER_ABSTRACTION_THRESHOLD=10
MEMEVOLVE_ENCODER_ENABLE_TOOL_EXTRACTION=true

# =============================================================================
# ADVANCED SETTINGS
# =============================================================================
MEMEVOLVE_ENABLE_EVOLUTION=$ENABLE_EVOLUTION
MEMEVOLVE_EVOLUTION_POPULATION_SIZE=10
MEMEVOLVE_EVOLUTION_GENERATIONS=20
MEMEVOLVE_EVOLUTION_MUTATION_RATE=0.1
MEMEVOLVE_EVOLUTION_CROSSOVER_RATE=0.5
MEMEVOLVE_EVOLUTION_SELECTION_METHOD=pareto
MEMEVOLVE_EVOLUTION_TOURNAMENT_SIZE=3
MEMEVOLVE_LOG_LEVEL=INFO
MEMEVOLVE_LOGGING_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
MEMEVOLVE_LOGGING_ENABLE_OPERATION_LOG=true
MEMEVOLVE_LOGGING_MAX_LOG_SIZE_MB=1024
MEMEVOLVE_LOG_API_SERVER_ENABLE=$ENABLE_API_LOG
MEMEVOLVE_LOG_MIDDLEWARE_ENABLE=$ENABLE_MIDDLEWARE_LOG
MEMEVOLVE_LOG_MEMORY_ENABLE=$ENABLE_MEMORY_LOG
MEMEVOLVE_LOG_EXPERIMENT_ENABLE=$ENABLE_EXPERIMENT_LOG
MEMEVOLVE_PROJECT_NAME=MemEvolve-API
MEMEVOLVE_PROJECT_ROOT=.
EOF

echo "✅ .env file created"

# Install dependencies
echo ""
echo "📚 Installing dependencies..."

# Base requirements
pip install -r requirements.txt

# Additional packages based on selections
if [ "$STORAGE_BACKEND" = "graph" ]; then
    echo "Installing Neo4j driver for graph storage..."
    pip install neo4j>=5.0.0
fi

# Install development dependencies if they exist
if [ -f "dev_requirements.txt" ]; then
    echo "Installing development dependencies..."
    pip install -r dev_requirements.txt
fi

# Run basic checks
echo ""
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

# Create startup scripts
echo ""
echo "🚀 Creating startup scripts..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux startup script
    echo ""
    echo "📂 Linux Installation Location"
    INSTALL_DIR=$(prompt_input "Installation directory for startup script" "$HOME/bin")
    mkdir -p "$INSTALL_DIR"

    SCRIPT_NAME="memevolveapi"
    SCRIPT_PATH="$INSTALL_DIR/$SCRIPT_NAME"

    cat > "$SCRIPT_PATH" << 'EOF'
#!/bin/bash
# MemEvolve API Startup Script

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")/MemEvolve-API"

# Change to project directory
cd "$PROJECT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Start the API server
echo "Starting MemEvolve API server..."
python scripts/start_api.py "$@"
EOF

    chmod +x "$SCRIPT_PATH"

    # Check if install dir is in PATH
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        echo "⚠️  $INSTALL_DIR is not in your PATH. Adding it..."
        echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> ~/.bashrc
        export PATH="$INSTALL_DIR:$PATH"
    fi

    # Add bash alias
    if ! grep -q "alias memevolveapi=" ~/.bashrc; then
        echo "Adding 'memevolveapi' alias..."
        echo "alias memevolveapi='$SCRIPT_NAME'" >> ~/.bashrc
    fi

    echo "✅ Linux startup script created: $SCRIPT_PATH"
    echo "✅ Added to PATH and created 'memevolveapi' alias"
    echo "   Usage: $SCRIPT_NAME"
    echo ""
    echo "🔄 To use the new alias immediately, run:"
    echo "   source ~/.bashrc"

elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows batch file
    cat > MemEvolveAPI.bat << 'EOF'
@echo off
REM MemEvolve API Startup Script for Windows

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%"

REM Change to project directory
cd /d "%PROJECT_DIR%"

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Start the API server
echo Starting MemEvolve API server...
python scripts\start_api.py %*
EOF

    echo "✅ Windows batch file created: MemEvolveAPI.bat"
    echo "   Usage: MemEvolveAPI.bat"

    # Create desktop shortcut (if possible)
    if command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -Command "
        \$WshShell = New-Object -comObject WScript.Shell;
        \$Shortcut = \$WshShell.CreateShortcut('Desktop\MemEvolveAPI.lnk');
        \$Shortcut.TargetPath = '$PWD\MemEvolveAPI.bat';
        \$Shortcut.WorkingDirectory = '$PWD';
        \$Shortcut.Save();
        "
        echo "✅ Desktop shortcut created"
    fi
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Your MemEvolve API is configured and ready to run."
echo ""
echo "To start the server:"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "  memevolveapi"
    echo "  # or"
    echo "  $SCRIPT_PATH"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "  MemEvolveAPI.bat"
fi
echo ""
echo "For manual activation:"
echo "  source .venv/bin/activate"
echo "  python scripts/start_api.py"
echo ""
echo "Available commands:"
echo "  ./scripts/run_tests.sh    - Run the test suite"
echo "  ./scripts/lint.sh         - Run code linting"
echo "  ./scripts/format.sh       - Format code"