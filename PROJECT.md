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

1. **Encode** - Transforms raw experience into structured representations (lessons, skills, abstractions)
2. **Store** - Persists encoded information (vector databases, JSON stores)
3. **Retrieve** - Selects task-relevant memory (semantic, hybrid strategies)
4. **Manage** - Maintains memory health (pruning, consolidation, deduplication, forgetting)

### API Wrapper Layer

MemEvolve provides an optional API wrapper that transparently integrates memory functionality with existing OpenAI-compatible LLM deployments:

- **Proxy Server** - FastAPI-based server that wraps any OpenAI-compatible API endpoints
- **Memory Integration** - Automatically retrieves and injects relevant context into requests
- **Experience Encoding** - Captures and stores new interactions for future retrieval
- **Management Endpoints** - Additional APIs for memory inspection and management
- **Drop-in Replacement** - Applications can use MemEvolve proxy instead of direct LLM API calls

## Project Structure

```
memevolve/
  ├── docs/                  # Documentation
  ├── src/
  │   ├── api/              # API wrapper server (FastAPI)
  │   │   ├── server.py     # Main API server
  │   │   ├── routes.py     # API endpoints
  │   │   └── middleware.py # Request/response processing
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
    │   ├── tests/            # Comprehensive test suite (393 tests)
  │   └── utils/            # Utility functions (config, logging, embeddings)
  ├── scripts/              # Deployment and utility scripts
  ├── AGENTS.md              # Agent development guidelines
  ├── MemEvolve_systems_summary.md  # System specification
  ├── PROJECT.md             # This file
  ├── README.md              # Brief project description
  └── TODO.md                # Development roadmap
```

## Current Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| Encode | ✅ Complete | ExperienceEncoder with strategies (lesson, skill, tool, abstraction) + batch processing |
| Store | ✅ Complete | JSON, FAISS-based vector, and Neo4j graph storage backends |
| Retrieve | ✅ Complete | Keyword, semantic, and hybrid retrieval strategies |
| Manage | ✅ Complete | Pruning, consolidation, deduplication, forgetting |
| Evolution | ✅ Complete | Genotype representation, Pareto selection, diagnosis, mutation |

### Memory Architectures Defined

| Architecture | Status | Key Features |
|-------------|---------|---------------|
| AgentKB | ✅ Defined | Static baseline genotype, lesson-based, minimal overhead |
| Lightweight | ✅ Defined | Trajectory-based genotype, JSON storage, auto-pruning |
| Riva | ✅ Defined | Agent-centric genotype, vector storage, hybrid retrieval |
| Cerebra | ✅ Defined | Tool distillation genotype, semantic graphs, advanced caching |

## Technology Stack

- **Language**: Python 3.12.3
- **LLM Backend**: OpenAI-compatible APIs (llama.cpp, vLLM, OpenAI, etc.)
- **API Framework**: FastAPI (for memory-enhanced proxy)
- **Vector Storage**: FAISS
- **Testing**: pytest (393 tests, pytest-timeout required)
- **Code Quality**: flake8, autopep8

## Test Coverage

- **Total Tests**: 393
- **Test Modules**: 27
- **Coverage**: All components + evolution framework + utilities + benchmark evaluation + graph storage
- **Test Timeout**: 600 seconds (10 minutes) required

Test breakdown:
- Evolution framework: ~90 tests (genotype, selection, diagnosis, mutation)
- Memory components: ~200 tests (encode, store, retrieve, manage, memory system)
- Utilities: ~70 tests (config, logging, metrics, profiling, data_io, debug_utils)

## Goals

1. ✅ Implement all four memory components (Encode, Store, Retrieve, Manage)
2. ✅ Support multiple storage backends (vector DB, JSON, graph DB)
3. ✅ Implement various retrieval strategies (semantic, hybrid)
4. ✅ Build memory management operations (pruning, consolidation, deduplication)
5. ✅ Create comprehensive test suite
6. ✅ Define reference memory architectures as genotypes
7. ✅ Enable meta-evolution mechanism to discover optimal memory architectures
8. ✅ Implement graph database backend
9. ✅ Implement LLM-guided retrieval strategy
10. ✅ Implement batch encoding optimization
11. ✅ Implement benchmark evaluation framework (GAIA, WebWalkerQA, xBench, TaskCraft)
12. ⏳ Complete empirical validation on benchmarks
13. 🚧 Create API wrapper server for seamless memory integration with OpenAI-compatible APIs
14. 🚧 Enable drop-in replacement for existing LLM API endpoints

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
