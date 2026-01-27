# MemEvolve-API — Agent Guidelines
These guidelines define **how coding agents must behave** when working in this repository.  

## Project Overview

**MemEvolve-API** is a Python-based, self-evolving memory system that proxies OpenAI-compatible API requests.  
It injects retrieved memory into prompts and continuously evolves its architecture using mutation, selection, and fitness evaluation.

## Agent Execution Model (CRITICAL)

Agents **MUST operate in explicit phases** and MUST NOT attempt to solve large tasks in a single reasoning chain.

### Required Phases
1. **Locate** – Identify relevant files/classes (no edits)
2. **Inspect** – Read *only* minimal required code
3. **Plan** – Summarize findings and propose a concrete change
4. **Implement** – Apply targeted edits
5. **Verify** – Sanity-check logic and consistency

Agents MUST pause or re-plan between phases if uncertainty increases.

## Context & Stability Rules (VERY IMPORTANT)
To prevent runaway context growth and OOMs:

- NEVER brute-force the repository
- NEVER read entire directories
- NEVER read more than **one file at a time**
- NEVER retry the same failed read repeatedly
- Prefer **search → targeted read → summarize**
- After reading large files, **summarize and discard raw details**
- Preserve *intent*, not verbatim code

### Stall Protection
If progress stalls (missing files, repeated errors, uncertainty):
- STOP
- State what is known
- List uncertainties
- Propose 1–2 next actions

Do **not** continue blindly.

## Build / Test / Lint Commands

### Environment Setup
```bash
source .venv/bin/activate
```
- Python command WILL NOT work without the venv activated

### Formatting & Linting
```bash
./scripts/format.sh
./scripts/lint.sh
```

### Testing
```bash
# Run all tests
./scripts/run_tests.sh

# Run single test file
./scripts/run_tests.sh tests/test_file.py

# Run single test function
./scripts/run_tests.sh tests/test_file.py::test_function

# Run with specific marker
pytest -m "not slow"
```

### API Server
```bash
source .venv/bin/activate && python scripts/start_api.py
```

## Code Standards

### General
- Python 3.10+
- Max line length: 100
- autopep8 (aggressive)
- flake8 (E203, W503 ignored)
- Docstrings required for public APIs

### Imports
- Standard library imports first, then third-party, then local imports
- Use `from typing import` for type annotations
- Group imports with blank lines between groups
- Use `__all__` lists in `__init__.py` files for explicit exports

### Naming
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`
- Files: `snake_case.py`

### Type Hints
- All functions must include type hints
- Use `Optional[T]` for nullable types
- Use `List[T]`, `Dict[K, V]` over `list`, `dict` for type safety
- Use `@dataclass` for data containers

### Documentation
- Module docstrings at top of files
- Class docstrings describing purpose
- Method docstrings with Args, Returns, Raises sections
- Use triple quotes `"""` for all docstrings

## Error Handling & Logging
- Use specific exception types
- Include contextual error messages
- Log before re-raising when appropriate
- Use structured logging (`OperationLogger`, `StructuredLogger`)

## Architecture Overview

### Core Components
1. **Encode** – `ExperienceEncoder`
2. **Store** – JSON / Vector / Graph backends
3. **Retrieve** – Semantic / Keyword / Hybrid strategies
4. **Manage** – MemoryManager and management strategies

### Evolution System
- Mutation (strategy, parameters, architecture)
- Selection via multi-dimensional fitness vectors
- Diagnosis-driven mutations (not random tuning)

## Memory Unit Schema

```python
{
    "id": str,
    "type": str,
    "content": str,
    "tags": List[str],
    "metadata": {
        "created_at": str,
        "category": str,
        "encoding_method": str,
        "quality_score": float
    },
    "embedding": Optional[List[float]]
}
```

## Testing Guidelines
- Use fixtures from `conftest.py`
- Test success and failure paths
- Use real whenever possible external dependencies (LLMs, file I/O)
- Include integration tests where meaningful

## Key Locations

- Source: `src/memevolve/`
- API: `src/memevolve/api/`
- Components: `src/memevolve/components/`
- Evolution: `src/memevolve/evolution/`
- Utilities: `src/memevolve/utils/`
- Scripts: `scripts/`
- Tests: `tests/`

## Environment Variables

- `MEMEVOLVE_UPSTREAM_BASE_URL=http://192.168.1.61:11434`
- `MEMEVOLVE_MEMORY_BASE_URL=http://192.168.1.61:11433`
- `MEMEVOLVE_EMBEDDING_BASE_URL=http://192.168.1.61:11435`
- `MEMEVOLVE_API_HOST=127.0.0.1`
- `MEMEVOLVE_API_PORT=11436`
- `MEMEVOLVE_DATA_DIR=./data`
- `MEMEVOLVE_CACHE_DIR=./cache`
- `MEMEVOLVE_LOGS_DIR=./logs`

## Local Model Constraints (IMPORTANT)
This repository may be developed using **slow local models with limited throughput**.

Agents should prefer:
- Precision over breadth
- Incremental changes over refactors
- Early summarization over extended reasoning
- Explicit plans over improvisation

## Final Rule
**Stability > Speed.  
Correctness > Completeness.  
Progress > Brute force.**
