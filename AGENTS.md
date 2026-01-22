# AGENTS.md

## Repository Information

- **GitHub**: https://github.com/thephimart/MemEvolve-API
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
- Provide full bash command to user when requesting test run including the arguments you would like used

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

## Directory Structure Instructions

MemEvolve follows a specific directory structure to maintain organization and enable proper backup, archiving, and maintenance workflows.

### Root Directory Structure

```
MemEvolve-API/
├── src/             # Source code (version controlled)
├── data/            # Persistent application data
├── cache/           # Temporary/recreatable data
├── logs/            # Application logs
├── docs/            # Documentation (organized by topic)
├── scripts/         # Development and maintenance scripts
├── .env*            # Environment configuration files
├── requirements.txt # Python dependencies
├── pyrightconfig.json # Type checking configuration
├── pytest.ini       # Test configuration
└── AGENTS.md        # This file (agent guidelines)
```

### Directory Purposes and Management

#### **`data/` - Persistent Application Data**
**Purpose:** Stores data that should be backed up and archived
**Retention:** Keep indefinitely (contains valuable application state)
**Backup:** Include in all backups and version control considerations

**Subdirectories:**
- `memory.json` - Core memory storage (primary persistent data)
- `evolution_state.json` - Evolution history and learned configurations
- `metrics/` - Performance metrics and benchmarking data
- `taskcraft/`, `webwalkerqa/`, `xbench/` - Benchmark evaluation datasets

**Management:**
- Archive regularly for disaster recovery
- Include in backup strategies
- May grow large over time - monitor disk usage

#### **`cache/` - Temporary Data**
**Purpose:** Stores temporary or recreatable data
**Retention:** Safe to delete anytime
**Backup:** Exclude from backups

**Contents:**
- Temporary files created during operation
- Session data that can be regenerated
- Cached embeddings or intermediate results

**Management:**
- Clean periodically with `rm -rf cache/`
- Exclude from version control (`.gitignore`)
- Regenerated automatically by application

#### **`logs/` - Application Logs**
**Purpose:** Stores application logs for debugging and monitoring
**Retention:** Keep for analysis, rotate periodically
**Backup:** Selective (recent logs for debugging)

**Subdirectories/Contents:**
- `api-server.log` - Main API server logs
- `middleware.log` - Memory middleware operations
- `memory.log` - Memory system operations
- `experiment.log` - Evaluation framework logs

**Management:**
- Rotate logs periodically: `find logs/ -name "*.log" -mtime +30 -delete`
- Include recent logs in support bundles
- Archive old logs for analysis if needed

#### **`docs/` - Documentation**
**Purpose:** Organized documentation for users and developers
**Retention:** Keep all documentation
**Backup:** Include in backups

**Subdirectories:**
- `user-guide/` - User-facing guides (getting started, config, deployment)
- `api/` - API reference and troubleshooting
- `development/` - Technical docs (architecture, evolution, roadmap)
- `tutorials/` - Advanced usage examples

**Management:**
- Keep all documentation current
- Include in version control
- Update when features change

#### **`scripts/` - Development Scripts**
**Purpose:** Utility scripts for development, testing, and maintenance
**Retention:** Keep all scripts
**Backup:** Include in backups

**Contents:**
- Build and test scripts
- Deployment and configuration scripts
- Data management and cleanup scripts
- Evaluation and benchmarking scripts

**Management:**
- Test scripts before committing changes
- Document script purposes and usage
- Keep scripts updated with project changes

### Data Lifecycle Guidelines

#### **What to Version Control:**
- ✅ Source code (`src/`)
- ✅ Documentation (`docs/`)
- ✅ Scripts (`scripts/`)
- ✅ Configuration templates (`.env.example`)
- ✅ Project metadata (`pyrightconfig.json`, `pytest.ini`)

#### **What to Exclude from Version Control:**
- ❌ Runtime data (`data/memory.json`, `data/evolution_state.json`)
- ❌ Temporary files (`cache/`)
- ❌ Logs (`logs/*.log`)
- ❌ Environment files (`.env`)
- ❌ OS-specific files (`.DS_Store`, `Thumbs.db`)

#### **Backup Strategy:**
- **Daily:** `data/` directory (persistent application data)
- **Weekly:** `logs/` (recent logs for debugging)
- **Monthly:** Full project directory (except `cache/`)
- **Never:** `cache/` directory (temporary data)

#### **Cleanup Commands:**
```bash
# Safe cleanup (won't lose important data)
rm -rf cache/                          # Clear temporary data
find logs/ -name "*.log" -mtime +30 -delete  # Remove old logs

# Backup important data
tar -czf backup_$(date +%Y%m%d).tar.gz data/ logs/
```

### Organization Principles

1. **Separation of Concerns**: Persistent vs temporary vs logs
2. **Backup Awareness**: Know what needs archiving
3. **Version Control Hygiene**: Only commit appropriate files
4. **Maintenance Ease**: Clear cleanup and backup procedures
5. **Developer Experience**: Predictable file locations

---

## Build, Linting & Testing Commands

### Testing Commands
```bash
# Activate virtual environment first
source .venv/bin/activate

# Run all tests (10 minute timeout)
pytest src/tests/ -v

# Run single test file
pytest src/tests/test_memory_system.py -v

# Run specific test function
pytest src/tests/test_memory_system.py::test_memory_system_initialization -v

# Run tests with coverage
pytest src/tests/ -v --cov=src --cov-report=term-missing
```

### Code Quality Commands
```bash
# Lint code with flake8 (max line length: 100)
flake8 src/ --max-line-length=100 --extend-ignore=E203,W503

# Format code (autopep8)
autopep8 --in-place --recursive --max-line-length=100 --aggressive --aggressive src/

# Install dependencies
pip install -r requirements.txt
```

---

## Code Style Guidelines

### Imports
- Standard libraries first, then third-party, then local imports
- Use absolute imports where possible
- Group imports by category with blank lines between groups

```python
import os
import json
from typing import Dict, List, Any, Optional

from openai import OpenAI
import numpy as np

from .utils import helper_function
from .config import ConfigManager
```

### Formatting & Indentation
- Use 4 spaces for indentation (no tabs)
- Line length should not exceed 100 characters
- Trailing whitespace is disallowed
- Use single quotes for strings unless containing single quotes

### Naming Conventions
- Functions: `snake_case` (e.g., `process_data()`, `get_memory_stats()`)
- Variables: `snake_case` (e.g., `data_variable`, `memory_units`)
- Classes: `CamelCase` (e.g., `MemorySystem`, `ExperienceEncoder`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES = 3`, `DEFAULT_TIMEOUT = 60`)
- Private members: `_private_method`, `_private_attribute`
- Module-level variables: `_module_var` (leading underscore)

### Types & Annotations
- Use type hints for all function parameters and return values
- Annotate complex data structures clearly
- Use `Optional` for nullable types
- Use `Union` for multiple possible types

```python
from typing import Dict, List, Any, Optional, Union

def process_experience(
    self,
    experience: Dict[str, Any],
    context: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Process a single experience with optional context."""
    pass

def get_similar_memories(
    self,
    query: str,
    top_k: int = 5,
    threshold: Optional[float] = None
) -> Union[List[Dict[str, Any]], None]:
    """Retrieve similar memories based on query."""
    pass
```

### Error Handling
- Use try/except blocks appropriately
- Catch specific exceptions when possible (avoid bare `except`)
- Log errors with appropriate severity levels
- Provide helpful error messages to users
- Use context managers for resource cleanup

### Documentation
- Add docstrings to all public functions/classes/methods
- Document parameters, returns, and exceptions
- Use Google-style docstring format

---

## Cursor & Copilot Rules (None Found)

No Cursor rules found in `.cursor/rules/` directory.
No Copilot instructions found in `.github/copilot-instructions.md`.

---

## Development Workflow

1. **Activate virtual environment first**: `source .venv/bin/activate`
3. **Follow code style guidelines consistently**
4. **Check for linting errors**: `flake8 src/ --max-line-length=100`
5. **Format code**: `autopep8 --in-place --recursive --max-line-length=100 --aggressive --aggressive src/`
6. **Update documentation**

---

This configuration provides coding agents with necessary information to work effectively within this repository.
