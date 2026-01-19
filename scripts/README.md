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

- **`init_memory_system.py`** - Initialize memory system with sample data
  - Creates a memory system with diverse sample experiences
  - Useful for development, testing, and demonstrations
  - Supports custom configuration and storage paths

## Usage Examples

```bash
# Initial setup
./scripts/setup.sh

# Daily development workflow
./scripts/run_tests.sh
./scripts/lint.sh
./scripts/format.sh

# Initialize with sample data
./scripts/init_memory_system.py --verbose

# Run specific tests
./scripts/run_tests.sh src/tests/test_memory_system.py -v

# Initialize with custom config
./scripts/init_memory_system.py --config my_config.yaml --storage-path ./my_data
```

## Requirements

- Bash shell
- Python 3.12+
- Virtual environment support

All scripts will automatically activate the virtual environment if available.