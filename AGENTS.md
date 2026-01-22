# AGENTS.md

## Repository Information

- **GitHub**: https://github.com/thephimart/memevolve
- **Branch**: master
- **License**: MIT

---

## Agent-Specific Instructions

### File Location Requirement
- **AGENTS.md must be located in `./`** (root directory of the repository)
- Do not move this file to any subdirectory

### Testing Requirements
- **Never run tests yourself** - Prompt the user to run tests when required
- Due to bash timeout limitations (120 seconds), tests must be run by the user
- Use this command for running tests: `pytest src/tests/ --timeout=600 -v`

### Server Management
- **Never start or restart the MemEvolve server yourself**
- Prompt the user to start/restart the server when these actions are required
- Use this command to start the server: `python scripts/start_api.py`
- Use this command to start with auto-reload: `python scripts/start_api.py --reload`

---

## Project Configuration for Coding Agents

### Environment Setup
- Python version: 3.12.3
- Virtual environment: `.venv`
- Activate via: `source .venv/bin/activate`
- Always start all sessions by activating python virtual environment

---

## Build, Linting & Testing Commands

### General Commands
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt

# Run all tests (10 minute timeout - configured in pytest.ini)
pytest src/tests/ --timeout=600 -v

# Run single test file
pytest src/tests/test_memory_system.py --timeout=600 -v

# Run specific test function
pytest src/tests/test_memory_system.py::test_memory_system_initialization --timeout=600 -v

# Lint code with flake8
flake8 src/ --max-line-length=100

# Format code (autopep8)
autopep8 --in-place --recursive src/
```

### Build Commands
```bash
# Build project (if applicable - depends on framework)
# Example for typical Python projects:
python setup.py build
```

---

## Code Style Guidelines

### Imports
- Import standard libraries first, then third-party, then local imports
- Use absolute imports where possible
- Group imports by category
  ```python
  import os
  import sys
  
  from .utils import helper_function
  ```

### Formatting & Indentation
- Use 4 spaces for indentation
- Line length should not exceed 100 characters
- Trailing whitespace is disallowed

### Naming Conventions
- Functions: `snake_case` (e.g., `process_data()`)
- Variables: `snake_case` (e.g., `data_variable = ...`)
- Classes: `CamelCase` (e.g., `MemorySystem`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES = 3`)
- Private members: `_private_method`

### Types & Annotations
- Use type hints for all function parameters and return values
- Annotate complex data structures clearly

### Error Handling
- Use try/except blocks appropriately
- Catch specific exceptions when possible
- Log errors with appropriate severity levels
- Provide helpful error messages to users

### Documentation
- Add docstrings to all public functions/classes
- Document parameters, returns, and exceptions
- Follow Google Python Style Guide for documentation

---

## Cursor & Copilot Rules (None Found)

No Cursor rules found in `.cursor/rules/` directory.
No Copilot instructions found in `.github/copilot-instructions.md`.

---

## Development Workflow
1. Activate virtual environment first
2. Run tests before committing changes
3. Follow code style guidelines consistently
4. Check for linting errors: `flake8 src/ --max-line-length=100`

---

## Project Documentation

- **README.md** - Quick start, features, and installation
- **docs/index.md** - Main documentation index and navigation
- **docs/development/architecture.md** - System design and implementation
- **docs/development/roadmap.md** - Development priorities and progress

**GitHub Repository**: https://github.com/thephimart/memevolve

Review docs/development/architecture.md first to understand the overall architecture, then consult docs/development/roadmap.md for current development priorities.

---

## Repository Workflow

### Development Commands
```bash
# Clone the repository
git clone https://github.com/thephimart/memevolve.git
cd memevolve

# Create a feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "Description of your changes"

# Push to remote
git push origin feature/your-feature-name
```

### Getting Help
- Issues: https://github.com/thephimart/memevolve/issues
- Documentation: https://github.com/thephimart/memevolve/tree/master/docs

---

## Configuration Management

### Using the Config System

The MemEvolve configuration management system provides a unified way to manage all configuration settings.

```python
from src.utils.config import ConfigManager, load_config, MemEvolveConfig

# Load default configuration (from environment variables and .env file)
config = load_config()

# Use ConfigManager for advanced operations
manager = ConfigManager()

# Update configuration values (primarily through environment variables)
manager.update("llm.base_url")

# Get specific configuration values
base_url = manager.get("llm.base_url")
```

### Architecture Presets

Pre-configured presets for different memory architectures:

```python
from src.utils.config import ConfigManager

# Get preset configuration for an architecture
agentkb_config = ConfigManager.get_architecture_config("agentkb")
lightweight_config = ConfigManager.get_architecture_config("lightweight")
riva_config = ConfigManager.get_architecture_config("riva")
cerebra_config = ConfigManager.get_architecture_config("cerebra")
```

### Environment Variables

Configuration can be overridden using environment variables. See `.env.example` and `.docker.env.example` for available variables:

- `MEMEVOLVE_LLM_BASE_URL`: LLM base URL
- `MEMEVOLVE_LLM_API_KEY`: LLM API key
- `MEMEVOLVE_LLM_MODEL`: LLM model name
- `MEMEVOLVE_EMBEDDING_BASE_URL`: Embedding API base URL
- `MEMEVOLVE_EMBEDDING_API_KEY`: Embedding API key
- `MEMEVOLVE_STORAGE_PATH`: Storage backend path
- `MEMEVOLVE_RETRIEVAL_TOP_K`: Default retrieval top_k
- `MEMEVOLVE_LOG_LEVEL`: Logging level
- `MEMEVOLVE_API_HOST`: API server host
- `MEMEVOLVE_API_PORT`: API server port

### Configuration Files

- `.env`: Runtime configuration file (not accessible by agents)
- `.env.example`: Template configuration file for development setup
- `.docker.env.example`: Template configuration file for Docker deployment
- Configuration is loaded via environment variables and .env files

---

## Logging System

### Basic Logging Setup

```python
from src.utils.logging import setup_logging, get_logger

# Setup basic logging
logger = setup_logging(level="INFO")

# Get a logger instance
my_logger = get_logger("my_module")
my_logger.info("Starting up...")
```

### Logging to File

```python
# Log to file with rotation
logger = setup_logging(
    level="DEBUG",
    log_file="./logs/memevolve.log",
    max_bytes=104857600,  # 100MB
    backup_count=5
)
```

### Operation Logging

Track memory system operations:

```python
from src.utils.logging import OperationLogger

# Create operation logger
op_logger = OperationLogger(max_entries=1000)

# Log operations
op_logger.log("encode", {"type": "lesson", "count": 5})
op_logger.log("retrieve", {"query": "test", "results": 3})

# Get statistics
stats = op_logger.get_stats()
print(f"Total operations: {stats['total']}")

# Get operations by type
encode_ops = op_logger.get_operations("encode")

# Export to file
op_logger.export("operations.json")
```

### Structured Logging

Consistent logging format with context:

```python
from src.utils.logging import StructuredLogger

logger = StructuredLogger("my_component")

# Log with context
logger.info("Processing request", request_id=123, user_id=456)
logger.error("Operation failed", error="timeout", retries=3)
```

### Configure from Config

```python
from src.utils.logging import configure_from_config

config = {
    "level": "DEBUG",
    "log_file": "./logs/app.log",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "max_log_size_mb": 50
}

logger = configure_from_config(config)
```

---

This configuration provides coding agents with necessary information to work effectively within this repository.
