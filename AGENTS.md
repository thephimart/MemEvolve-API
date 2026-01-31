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
# All tests
./scripts/run_tests.sh

# Single file
./scripts/run_tests.sh tests/test_file.py

# Single test
./scripts/run_tests.sh tests/test_file.py::test_function

# Marker-based
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

## Error Handling & Logging

- Use specific exception types
- Include contextual error messages
- Log before re-raising when appropriate
- Use structured logging (`OperationLogger`, `StructuredLogger`)

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

## Testing Guidelines

- Use fixtures from `conftest.py`
- Test success *and* failure paths
- Prefer real dependencies when feasible (LLMs, file I/O)
- Include integration tests where meaningful

---

## Key Locations

- Source: `src/memevolve/`
- API: `src/memevolve/api/`
- Components: `src/memevolve/components/`
- Evolution: `src/memevolve/evolution/`
- Utilities: `src/memevolve/utils/`
- Scripts: `scripts/`
- Tests: `tests/`

---

## Environment Variables (Testing)

- `MEMEVOLVE_UPSTREAM_BASE_URL`
- `MEMEVOLVE_MEMORY_BASE_URL`
- `MEMEVOLVE_EMBEDDING_BASE_URL`
- `MEMEVOLVE_API_HOST`
- `MEMEVOLVE_API_PORT`
- `MEMEVOLVE_DATA_DIR`
- `MEMEVOLVE_CACHE_DIR`
- `MEMEVOLVE_LOGS_DIR`

---

## Configuration Architecture Rules (CRITICAL)

### Configuration Priority Hierarchy (Highest to Lowest)

1. **evolution_state.json values**
   - Only if: `MEMEVOLVE_EVOLUTION_ENABLED=true` in `.env` AND `evolution_state.json` exists
   - Runtime mutations override all other sources
   
2. **.env file values**
   - If: evolution is disabled OR no `evolution_state.json` exists
   - Environment variables are the **primary source of truth**
   
3. **config.py defaults**
   - Fallback when neither above provides value
   - Dataclass defaults are **fallback only**
   
4. **[FORBIDDEN]** Hardcoded values in any other files

### Decision Flow

```
Is MEMEVOLVE_EVOLUTION_ENABLED in .env?
├── YES → Does evolution_state.json exist?
│         ├── YES → Use evolution_state values (override everything)
│         └── NO  → Use .env values
└── NO  → Use .env values (ignore evolution_state even if exists)
     └── If .env missing value → Use config.py defaults
```

### Centralized Configuration
- ALL configuration lives in `src/memevolve/utils/config.py`
- **ZERO hardcoded values outside `config.py`** (tests excepted)
- Single `ConfigManager` instance shared across all components
- Runtime components must reference live config state via `ConfigManager`

### Access Pattern
```python
# CORRECT - Always read from ConfigManager live state
def get_retrieval_limit(self) -> int:
    return self.config_manager.get('retrieval.default_top_k')

# FORBIDDEN - Cached values or hardcoded fallbacks
def get_retrieval_limit(self) -> int:
    return self.config.retrieval.default_top_k if self.config else 5  # NEVER DO THIS
```

### Evolution Sync Rules
- Evolution updates `ConfigManager` first using dot notation: `config_manager.update(retrieval__default_top_k=7)`
- Runtime components must reference live config state (not cached copies)
- Config changes propagate within one evolution cycle
- Boundary validation prevents invalid ranges
- Evolution state persistence: Mutations saved to `evolution_state.json` (not `.env`)

---

## Local Model Constraints (IMPORTANT)

Development may use **slow local models with limited throughput**.

Agents should prefer:
- Precision over breadth
- Incremental changes over refactors
- Early summarization over extended reasoning
- Explicit plans over improvisation

### Codebase Scale Context
- ~52 Python files / ~19K LOC
- Architecture exploration may require multiple targeted reads
- Respect context limits without crippling effectiveness

---

## Final Rule

**Stability > Speed**  
**Correctness > Completeness**  
**Progress > Brute force**
