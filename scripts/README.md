# MemEvolve Development Scripts

This directory contains development and utility scripts for the MemEvolve project.

## Scripts Overview

### Environment Setup

- **`setup.sh`** - Complete development environment setup
  - Creates virtual environment
  - Installs dependencies
  - Runs basic validation checks
  - Run this first when setting up a development environment

### Development Workflow

- **`run_tests.sh`** - Run the complete test suite
  - Includes coverage reporting if available
  - Respects pytest timeout settings
  - Can pass additional pytest arguments

- **`lint.sh`** - Run code quality checks
  - Uses flake8 with project-specific settings
  - Checks for style violations and potential issues

- **`format.sh`** - Format Python code
  - Uses autopep8 for consistent formatting
  - Applies project style guidelines

### Data Management

- **`cleanup_evolution.sh`** - Remove all evolution data
  - Cleans evolution state and cache files
  - Safe to run when resetting evolution experiments

- **`cleanup_memory.sh`** - Remove all memory data
  - Deletes memory storage files and directories
  - Use with caution - irreversible

- **`cleanup_logs.sh`** - Remove all log files
  - Cleans the logs directory
  - Useful for log rotation

- **`memory_prune.py`** - Manually prune memory units
  - Remove old or excessive memory units
  - Supports various pruning criteria

- **`memory_consolidate.py`** - Manually consolidate memory units
  - Merge similar memories by type
  - Reduces storage and improves retrieval

- **`memory_deduplicate.py`** - Remove duplicate memory units
  - Eliminates content duplicates
  - Configurable similarity threshold

- **`memory_forget.py`** - Apply forgetting mechanisms
  - LRU or random forgetting strategies
  - Helps manage memory size

## Usage Examples

```bash
# Initial setup
./scripts/setup.sh

# Daily development workflow
./scripts/run_tests.sh
./scripts/lint.sh
./scripts/format.sh

# Run specific tests
./scripts/run_tests.sh src/tests/test_memory_system.py -v

# Memory maintenance examples
./scripts/cleanup_memory.sh                    # Remove all memory data
./scripts/memory_prune.py --max-count 1000    # Keep only 1000 most recent memories
./scripts/memory_deduplicate.py --threshold 0.9  # Remove duplicates
./scripts/memory_forget.py --strategy lru --count 100  # Forget 100 LRU memories
```

## Requirements

- Bash shell
- Python 3.12+
- Virtual environment support

All scripts will automatically activate the virtual environment if available.