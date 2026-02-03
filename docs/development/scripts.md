# MemEvolve Development Scripts

This directory contains development and utility scripts for the MemEvolve project.

## Data Organization

MemEvolve follows a clear data organization structure:

- **`data/`** - Persistent data that should be backed up and archived
  - `memory.json` - Core memory storage
  - `evolution_state.json` - Evolution history and learned configurations
  - `metrics/` - Performance metrics and benchmarking data
  - Benchmark datasets (`taskcraft/`, `webwalkerqa/`, `xbench/`)

- **`cache/`** - Temporary data that can be recreated

- **`logs/`** - All application logs for debugging and monitoring

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

- **`scrubber.sh`** - Comprehensive project cleanup with interactive prompts
  - **Global Data Deletion**: Option to delete ALL data files at once
  - **Per-Directory Cleanup**: Individual prompts for each data subdirectory
  - **Safe Defaults**: All prompts default to "No" to prevent accidental deletion
  - **Organized Deletion**: Handles logs, data, cache, and build artifacts
  - **Interactive Confirmation**: Shows file counts before deletion
  ```bash
  # Usage
  ./scripts/scrubber.sh
  
  # Interactive prompts:
  # - Delete ALL data files? (y/N) 
  # - Delete files in data/evolution? (y/N)
  # - Delete files in data/memory? (y/N)
  # - ... continues for each subdirectory
  ```

- **`cleanup_fresh.sh`** - Complete fresh install cleanup
  - Removes ALL data, logs, and cache files
  - Returns installation to completely fresh state
  - Preserves .env and virtual environment
  - Use with extreme caution - irreversible

- **`cleanup_evolution.sh`** - Remove all evolution data
  - Cleans evolution state and cache files
  - Safe to run when resetting evolution experiments
  - Preserves memory data

- **`cleanup_memory.sh`** - Remove all memory data
  - Deletes memory storage files and directories
  - Use with caution - irreversible

- **`cleanup_logs.sh`** - Remove all log files
  - Cleans logs directory
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

### Deployment & Development

- **`deploy.sh`** - Deployment management
  - Start/stop services
  - Show service status and logs
  - Health monitoring and cleanup

- **`start_api.py`** - Main API server startup
  - FastAPI server with auto-reload option
  - Environment configuration loading
  - Development server management

### Development Tools

- **`init_memory_system.py`** - Initialize memory with sample data
  - Creates sample experiences for development/testing
  - Populates memory system with diverse content types
  - Useful for testing memory retrieval and evolution

- **`iterative_prompts.sh`** - Load testing script
  - Runs 200 API calls with random questions
  - Tests full system: API, server, middleware, memory, evolution
  - Generates comprehensive logs and performance data

- **`performance_analyzer.py`** - Comprehensive performance analysis tool
  - Analyzes system logs and data files for any time period
  - Generates detailed performance reports with actionable insights
  - No LLM dependencies - pure Python analysis
  - Identifies bottlenecks and optimization opportunities

## Usage Examples

```bash
# Initial setup
./scripts/setup.sh

# Daily development workflow
./scripts/run_tests.sh
./scripts/lint.sh
./scripts/format.sh

# Run specific tests
./scripts/run_tests.sh tests/test_memory_system.py -v

# Memory maintenance examples
./scripts/scrubber.sh                        # Interactive project cleanup (recommended)
./scripts/cleanup_fresh.sh                     # Complete fresh install (removes everything)
./scripts/cleanup_memory.sh                    # Remove all memory data
./scripts/cleanup_evolution.sh                 # Remove evolution data only
./scripts/cleanup_logs.sh                      # Remove all log files
./scripts/memory_prune.py --max-count 1000    # Keep only 1000 most recent memories
./scripts/memory_deduplicate.py --threshold 0.9  # Remove duplicates
./scripts/memory_forget.py --strategy lru --count 100  # Forget 100 LRU memories

# Deployment examples
./scripts/deploy.sh build    # Build for deployment
./scripts/deploy.sh start    # Start services
./scripts/deploy.sh status   # Check service status
./scripts/deploy.sh logs     # View service logs
./scripts/deploy.sh stop     # Stop services

# Development tools
./scripts/init_memory_system.py               # Initialize with sample data
./scripts/iterative_prompts.sh                # Run load testing (200 API calls)
./scripts/performance_analyzer.py --days 1    # Analyze last day's performance
./scripts/performance_analyzer.py --start-date 2026-01-20 --end-date 2026-01-25  # Custom date range
./scripts/start_api.py                        # Start API server
./scripts/start_api.py --reload               # Start with auto-reload
```

## Requirements

- Bash shell
- Python 3.12+
- Virtual environment support

All scripts will automatically activate the virtual environment if available.

## Data Management

### Backup Strategy
- **`data/`** contains all persistent data that should be backed up regularly
- **`logs/`** may contain important debugging information for issues
- **`cache/`** can be safely deleted and will regenerate

### Cleanup Commands
```bash
# Complete fresh install cleanup (removes everything except .env and venv)
./scripts/cleanup_fresh.sh

# Remove temporary cache (safe to delete)
rm -rf cache/

# Remove old logs (use with caution)
find logs/ -name "*.log" -mtime +30 -delete

# Clean evolution state (resets evolution progress)
./scripts/cleanup_evolution.sh

# Remove specific data types
./scripts/cleanup_memory.sh   # Remove memory data
./scripts/cleanup_logs.sh     # Remove log files
```