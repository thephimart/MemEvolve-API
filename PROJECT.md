# MemEvolve: Project Overview

## Repository

- **GitHub**: https://github.com/thephimart/memevolve
- **Branch**: master
- **License**: MIT

## Purpose

MemEvolve is a meta-evolving memory framework for agent systems that enables LLM-based agents to automatically improve their own memory architecture through dual-level evolution.

## Core Concept

The framework implements a **bilevel optimization** approach:

### Inner Loop - Experience Evolution
- Agents operate with a fixed memory architecture
- Execute tasks and accumulate experiences
- Experiences populate the memory system

### Outer Loop - Memory Architecture Evolution
- Memory system is evaluated and redesigned
- Architectural changes driven by empirical task feedback
- Results in progressively better memory systems

## Architecture

The memory system is decomposed into four orthogonal components:

```
Ω = (Encode, Store, Retrieve, Manage)
```

### Component Responsibilities

1. **Encode** - Transforms raw experience into structured representations (lessons, skills, tools, abstractions)
2. **Store** - Persists encoded information (vector databases, JSON stores, graphs, tool libraries)
3. **Retrieve** - Selects task-relevant memory (semantic, hybrid, LLM-guided strategies)
4. **Manage** - Maintains memory health (pruning, consolidation, deduplication, forgetting)

## Project Structure

```
memevolve/
 ├── docs/                  # Documentation
 ├── src/
 │   ├── components/        # Memory component implementations
 │   │   ├── encode/       # Experience encoding
 │   │   ├── store/        # Storage implementations (JSON, FAISS vector)
 │   │   ├── retrieve/     # Retrieval strategies (keyword, semantic, hybrid)
 │   │   └── manage/       # Memory management (pruning, consolidation)
 │   ├── evolution/         # Meta-evolution framework
 │   │   ├── genotype.py   # Memory architecture representation
 │   │   ├── selection.py  # Pareto-based selection
 │   │   ├── diagnosis.py  # Trajectory analysis
 │   │   └── mutation.py   # Architecture mutation
  │   ├── tests/            # Comprehensive test suite (268 tests)
  │   └── utils/            # Utility functions (config, logging, embeddings)
 ├── AGENTS.md              # Agent development guidelines
 ├── MemEvolve_systems_summary.md  # System specification
 ├── PROJECT.md             # This file
 ├── README.md              # Brief project description
 └── TODO.md                # Development roadmap
```

## Current Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| Encode | ✅ Complete | ExperienceEncoder with strategies (lesson, skill, tool, abstraction) |
| Store | ✅ Complete | JSON and FAISS-based vector storage backends |
| Retrieve | ✅ Complete | Keyword, semantic, and hybrid retrieval strategies |
| Manage | ✅ Complete | Pruning, consolidation, deduplication, forgetting |
| Evolution | ✅ Complete | Genotype representation, Pareto selection, diagnosis, mutation |

### Memory Architectures Implemented

| Architecture | Status | Key Features |
|-------------|---------|---------------|
| AgentKB | ✅ Complete | Static baseline, lesson-based, minimal overhead |
| Lightweight | ✅ Complete | Trajectory-based, JSON storage, auto-pruning |
| Riva | ✅ Complete | Agent-centric, vector storage, hybrid retrieval |
| Cerebra | ✅ Complete | Tool distillation, semantic graphs, advanced caching |

## Technology Stack

- **Language**: Python 3.12.3
- **LLM Backend**: llama.cpp (OpenAI-compatible API)
- **Vector Storage**: FAISS
- **Testing**: pytest (268 tests, pytest-timeout required)
- **Code Quality**: flake8, autopep8

## Test Coverage

- **Total Tests**: 268
- **Test Modules**: 18
- **Coverage**: All components + evolution framework + utilities
- **Test Timeout**: 600 seconds (10 minutes) required

Test breakdown:
- Evolution framework: 57 tests (genotype, selection, diagnosis, mutation)
- Memory components: 152 tests (encode, store, retrieve, manage, memory system)
- Utilities: 59 tests (config, logging, basic operations)

## Goals

1. ✅ Implement all four memory components (Encode, Store, Retrieve, Manage)
2. ✅ Support multiple storage backends (vector DB, JSON, graph)
3. ✅ Implement various retrieval strategies (semantic, hybrid, LLM-guided)
4. ✅ Build memory management operations (pruning, consolidation, deduplication)
5. ✅ Create comprehensive test suite
6. ✅ Enable meta-evolution mechanism to discover optimal memory architectures
7. ⏳ Validate on benchmarks (GAIA, WebWalkerQA, xBench, TaskCraft)

## Key Design Principles

- Agent-driven memory decisions
- Hierarchical representations
- Multi-level abstraction
- Stage-aware retrieval
- Selective forgetting

## Quick Start

```bash
# Clone the repository
git clone https://github.com/thephimart/memevolve.git
cd memevolve

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests (10 minute timeout)
pytest src/tests/ --timeout=600 -v
```

## Related Documentation

- [README.md](README.md) - Quick start and features overview
- [TODO.md](TODO.md) - Development roadmap and progress tracking
- [AGENTS.md](AGENTS.md) - Development guidelines for coding agents
- [MemEvolve_systems_summary.md](MemEvolve_systems_summary.md) - System specification
