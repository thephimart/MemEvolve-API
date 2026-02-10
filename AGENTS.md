# MemEvolve-API — Agent Guidelines

These guidelines define **how coding agents must behave** when working in this repository. They are mandatory and optimized for **stability, low-context operation, and local-model constraints**.

---

## Project Overview

**MemEvolve-API** is a Python-based, self-evolving memory system that proxies OpenAI-compatible API requests. It retrieves and injects memory into prompts and evolves its architecture via **mutation, selection, and fitness evaluation**.

---

## Agent Execution Model (CRITICAL)

Agents **MUST operate in explicit phases**. Large tasks must **never** be solved in a single reasoning chain.

### Required Phases
1. **Locate** – Identify relevant files/classes (no edits)
2. **Inspect** – Read *only* the minimum code needed for context
3. **Plan** – Summarize findings and propose concrete changes
4. **Implement** – Apply targeted edits only
5. **Verify** – Sanity-check logic, tests, and consistency

Agents MUST pause or re-plan if uncertainty increases.

---

## Context & Stability Rules (VERY IMPORTANT)

To prevent runaway context growth and OOMs:

- NEVER brute-force the repository
- NEVER read entire directories without purpose
- NEVER retry the same failed read repeatedly
- Prefer **search → targeted read → summarize**
- After large reads, **summarize and discard raw details**
- Preserve *intent*, not verbatim code

### File Reading Strategy
- **<200 lines**: Read fully
- **200–800 lines**: Read key sections, summarize
- **>800 lines**: Search first, then read targeted sections
- **Parallel reads**: Max 2–3 related files
- **Architecture exploration**: Allowed, but summarize aggressively

### Stall Protection
If progress stalls:
- STOP
- State what is known
- List uncertainties
- Propose 1–2 next actions

Do **not** continue blindly.

---

## Build / Test / Lint

### Environment Setup
```bash
source .venv/bin/activate
```
- Python commands WILL NOT work without the venv activated

### Formatting & Linting
```bash
./scripts/format.sh
./scripts/lint.sh
```

### Testing
```bash
./scripts/run_tests.sh
./scripts/run_tests.sh tests/test_file.py
./scripts/run_tests.sh tests/test_file.py::test_function
pytest -m "not slow"
```

### API Server
```bash
source .venv/bin/activate && python scripts/start_api.py
```

---

## Code Standards

### General
- Python 3.10+
- Max line length: 100
- autopep8 (aggressive)
- flake8 (E203, W503 ignored)
- Docstrings required for public APIs

### Imports
- Standard → Third-party → Local
- Use `from typing import ...`
- Blank lines between import groups
- `__all__` required in `__init__.py`

### Naming
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`
- Files: `snake_case.py`
- ### Exclusions
  - `logger` is always lower-case NOT `logger`

### Type Hints
- Required for all functions
- Use `Optional[T]` for nullable values
- Prefer `List[T]`, `Dict[K, V]`
- Use `@dataclass` for data containers

### Documentation
- Module docstring at top of file
- Class docstrings describe purpose
- Method docstrings include Args / Returns / Raises
- Use triple quotes `"""`

---

## Python Notation & Naming Rules (Agent-Enforced)

### Dot Notation
- **ALWAYS** use dot notation for object, module, and package access.
- **NEVER** use dot notation for dictionary keys or list indices.

### Underscore Rules
- `snake_case` for all identifiers created by agents
- `_single_leading` = internal/protected (avoid unless required)
- `__double_leading` = private (forbidden to access)
- `_` = ignored values only

---

## Error Handling & Logging

- Use specific exception types
- Include contextual error messages
- Log before re-raising when appropriate
- Use structured logging with directory tree mirroring

### Logging Architecture (P1.0 Compliance)

#### Complete Coverage Requirement
- ALL .py files in ./src/memevolve/ MUST have logging implemented
- ALL .py files in ./scripts/ MUST have logging implemented
- NO exceptions for utility files, test files, or configuration files

#### Exact 1:1 File-to-Log Mapping
- ALL logging MUST use LoggingManager.get_logger(__name__)
- EACH .py file MUST have corresponding .log file in ./logs/
- Directory structure MUST exactly mirror ./src/memevolve/ structure:
  - src/memevolve/components/encode/encoder.py → ./logs/components/encode/encoder.log
  - scripts/start_api.py → ./logs/scripts/start_api.log
  - memevolve/__init__.py → ./logs/memevolve.log

#### Directory Structure Mirroring
```
./logs/
├── api/
│   ├── enhanced_http_client.log
│   ├── enhanced_middleware.log
│   ├── evolution_manager.log
│   ├── routes.log
│   └── server.log
├── components/
│   ├── encode/
│   │   ├── encoder.log
│   │   └── metrics.log
│   ├── manage/
│   │   ├── base.log
│   │   └── simple_strategy.log
│   ├── retrieve/
│   │   ├── base.log
│   │   ├── hybrid_strategy.log
│   │   ├── keyword_strategy.log
│   │   ├── llm_guided_strategy.log
│   │   ├── metrics.log
│   │   └── semantic_strategy.log
│   └── store/
│       ├── base.log
│       ├── graph_store.log
│       ├── json_store.log
│       └── vector_store.log
├── evaluation/
│   ├── experiment_runner.log
│   ├── gaia_evaluator.log
│   ├── genotype_translator.log
│   ├── memory_scorer.log
│   ├── response_scorer.log
│   ├── taskcraft_evaluator.log
│   ├── token_analyzer.log
│   ├── webwalkerqa_evaluator.log
│   └── xbench_evaluator.log
├── evolution/
│   ├── diagnosis.log
│   ├── genotype.log
│   ├── mutation.log
│   └── selection.log
├── scripts/
│   ├── business_impact_analyzer.log
│   ├── generate_test_data.log
│   ├── init_memory_system.log
│   ├── memory_consolidate.log
│   ├── memory_deduplicate.log
│   ├── memory_forget.log
│   ├── memory_prune.log
│   ├── performance_analyzer.log
│   └── start_api.log
├── utils/
│   ├── config.log
│   ├── data_io.log
│   ├── debug_utils.log
│   ├── embeddings.log
│   ├── endpoint_metrics_collector.log
│   ├── logging_manager.log
│   ├── metrics.log
│   ├── mock_generators.log
│   ├── profiling.log
│   ├── quality_scorer.log
│   ├── real_data_generator.log
│   ├── streaming.log
│   └── __init__.log
├── memory_system.log
└── memevolve.log
```

#### Logger Naming Requirements
```python
# All source files MUST use this exact pattern:
logger = LoggingManager.get_logger(__name__)

# Script files MUST include path setup:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from memevolve.utils.logging_manager import LoggingManager
logger = LoggingManager.get_logger(__name__)

# NO exceptions - every .py file requires logging
```

#### Logging Features
- **Automatic directory creation** based on logger name
- **Log rotation**: 10MB max, 5 backups per file
- **UTF-8 encoding** for proper character handling
- **Console + File output** with consistent formatting
- **Complete file-level isolation** for precise debugging

#### Strict Compliance Rules
- **FORBIDDEN**: logging.getLogger() calls outside logging_manager.py
- **FORBIDDEN**: Custom file handlers outside LoggingManager
- **FORBIDDEN**: Hardcoded log paths or names
- **FORBIDDEN**: Missing logging in any .py file
- **REQUIRED**: LoggingManager.get_logger(__name__) in every .py file

---

## Architecture Overview

### Core Components
1. **Encode** – `ExperienceEncoder`
2. **Store** – JSON / Vector / Graph backends
3. **Retrieve** – Semantic / Keyword / Hybrid strategies
4. **Manage** – `MemoryManager` and management strategies

### Evolution System
- Mutation of strategy, parameters, or architecture
- Selection via multi-dimensional fitness vectors
- Diagnosis-driven mutations (no blind random tuning)

---

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

---

## Configuration Architecture Rules (CRITICAL)

### Configuration Priority Hierarchy (Highest to Lowest)
1. evolution_state.json (if enabled)
2. .env values
3. config.py defaults
4. **FORBIDDEN:** hardcoded values elsewhere

### Centralized Configuration
- ALL config via `ConfigManager`
- ZERO hardcoded values outside `config.py`
- Runtime reads must be live, not cached

---

## Final Rule

**Stability > Speed**  
**Correctness > Completeness**  
**Progress > Brute force**
