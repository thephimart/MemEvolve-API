# MemEvolve-API Agent Guidelines

This file contains essential information for agentic coding agents working in this repository.

## Project Overview

MemEvolve-API is a Python-based self-evving memory system that proxies OpenAI-compatible API requests. It intercepts API calls, retrieves relevant memory context, injects it into prompts, and continuously evolves its memory architecture through mutations to optimize performance.

## Build/Test/Lint Commands

### Environment Setup
```bash
# Setup development environment
./scripts/setup.sh

# Activate virtual environment (if not already active)
source .venv/bin/activate
```

### Code Quality
```bash
# Format code according to project standards
./scripts/format.sh
# Equivalent: autopep8 --in-place --recursive --max-line-length=100 --aggressive --aggressive src/

# Run linting checks
./scripts/lint.sh  
# Equivalent: flake8 src/ --max-line-length=100 --extend-ignore=E203,W503
```

### Testing
```bash
# Run full test suite
./scripts/run_tests.sh
# Equivalent: python3 -m pytest tests/ --timeout=600 -v

# Run specific test file
python3 -m pytest tests/test_memory_system.py -v

# Run single test
python3 -m pytest tests/test_memory_system.py::TestMemorySystem::test_add_experience -v

# Run tests with coverage
python3 -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html:htmlcov
```

### API Server
```bash
# Start the MemEvolve API server
python3 -m memevolve.api.server
# OR
./scripts/start_api.py
```

## Code Style Guidelines

### General Standards
- **Python**: 3.10+ required
- **Line Length**: 100 characters max
- **Formatting**: autopep8 with aggressive mode
- **Linting**: flake8 with E203 and W503 ignored
- **Documentation**: Docstrings for all public classes and functions

### Import Organization
```python
# Standard library imports first
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

# Third-party imports
import numpy as np
import pytest
from openai import OpenAI

# Local imports
from .components.encode import ExperienceEncoder
from .utils.config import MemEvolveConfig
```

### Class and Function Naming
- **Classes**: PascalCase (e.g., `MemorySystem`, `ExperienceEncoder`)
- **Functions/Methods**: snake_case (e.g., `encode_experience`, `get_health_metrics`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_TIMEOUT`, `MAX_RETRIES`)
- **Private Members**: Prefix with underscore (e.g., `_internal_method`, `_config_field`)

### Type Hints
All functions and methods must include type hints:
```python
def encode_experience(
    self, 
    experience: Dict[str, Any], 
    strategy: Optional[str] = None
) -> Dict[str, Any]:
    """Encode experience into memory unit."""
    pass
```

### Error Handling
- Use specific exception types with descriptive messages
- Include context about what operation failed
- Log errors before re-raising when appropriate

```python
try:
    result = self._encode_with_llm(experience)
except OpenAIError as e:
    logger.error(f"LLM encoding failed for experience {experience.get('id', 'unknown')}: {e}")
    raise EncodingError(f"Failed to encode experience: {e}") from e
```

### Configuration Management
- Use the centralized `MemEvolveConfig` class from `memevolve.utils.config`
- Environment variables should follow `MEMEVOLVE_*` naming convention
- Validate configuration values at startup

### Component Architecture
The system follows a component-based architecture:

1. **Encode**: `ExperienceEncoder` - Converts experiences to memory units
2. **Store**: Storage backends (`JSONFileStore`, `VectorStore`, `GraphStorageBackend`)
3. **Retrieve**: Retrieval strategies (`SemanticRetrievalStrategy`, `KeywordRetrievalStrategy`, `HybridRetrievalStrategy`)
4. **Manage**: Memory management (`SimpleManagementStrategy`, `MemoryManager`)

### Testing Patterns
- Use fixtures from `conftest.py` for common test data
- Test both success and failure scenarios
- Include integration tests for component interactions
- Mock external dependencies (LLM APIs, file I/O)

### Logging and Monitoring
- Use structured logging with the `OperationLogger` and `StructuredLogger` classes
- Include operation timing and key metrics
- Log at appropriate levels (DEBUG, INFO, WARNING, ERROR)

### Memory Unit Structure
All memory units must follow this structure:
```python
{
    "id": str,           # Unique identifier
    "type": str,         # lesson, skill, tool, abstraction
    "content": str,      # Main content
    "tags": List[str],   # Categorization tags
    "metadata": {        # Rich metadata
        "created_at": str,      # ISO timestamp
        "category": str,         # Domain area
        "encoding_method": str,  # How it was encoded
        "quality_score": float,  # Optional quality rating
        # ... additional metadata
    },
    "embedding": Optional[List[float]]  # Vector embedding
}
```

### API Integration
- Handle OpenAI-compatible API endpoints
- Support streaming responses
- Implement proper error handling for upstream failures
- Respect rate limiting and timeouts

### Evolution System
The system continuously evolves through:
- **Mutation**: Changes to encoding strategies, retrieval parameters, storage configurations
- **Selection**: Performance-based fitness evaluation
- **Quality Scoring**: Parity-based evaluation across different model types

## Development Workflow

1. Make code changes
2. Run `./scripts/format.sh` to format code
3. Run `./scripts/lint.sh` to check code quality
4. Run relevant tests: `python3 -m pytest tests/your_test_file.py -v`
5. If all checks pass, the code is ready

## Key File Locations

- **Main Source**: `src/memevolve/`
- **Tests**: `tests/`
- **Scripts**: `scripts/`
- **Configuration**: `src/memevolve/utils/config.py`
- **API Server**: `src/memevolve/api/server.py`
- **Core Memory System**: `src/memevolve/memory_system.py`
- **Components**: `src/memevolve/components/` (encode, retrieve, store, manage)
- **Evolution**: `src/memevolve/evolution/`

## Environment Variables

Key environment variables for development:
- `MEMEVOLVE_MEMORY_BASE_URL`: Memory API endpoint
- `MEMEVOLVE_MEMORY_API_KEY`: API key for memory operations
- `MEMEVOLVE_UPSTREAM_BASE_URL`: Upstream LLM API endpoint
- `MEMEVOLVE_UPSTREAM_API_KEY`: API key for upstream LLM
- `PYTHONPATH`: Should include `src` directory