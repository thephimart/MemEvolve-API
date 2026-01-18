# AGENTS.md

## Project Configuration for Coding Agents

### Environment Setup
- Python version: 3.12.3
- Virtual environment: `.venv`
- Activate via: `. .venv/bin/activate`
- Always start all seesions by activating the python virtual environment

---

## Build, Linting & Testing Commands

### General Commands
```bash
# Activate virtual environment
source ./.venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt

# Run all tests
pytest src/tests/

# Run single test file
pytest src/tests/test_memory.py

# Run specific test function
pytest src/tests/test_memory.py::TestMemory::test_basic_operations

# Lint code with flake8
flake8 .

# Format code (autopep8)
autopep8 --in-place --recursive .
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
- Line length should not exceed 79 characters
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

---

## Project Documentation

- **PROJECT.md** - Comprehensive project overview, architecture, and implementation status
- **TODO.md** - Development roadmap with phased task list
- **MemEvolve_systems_summary.md** - System specification and design principles
- **README.md** - Brief project description

Review PROJECT.md first to understand the overall architecture, then consult TODO.md for current development priorities.

--- 
This configuration provides coding agents with necessary information to work effectively within this repository.
