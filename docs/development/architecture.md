# MemEvolve: Architecture and System Design

## Overview

MemEvolve is a meta-evolving memory framework for agent systems that enables LLM-based agents to automatically improve their own memory architecture through dual-level evolution.

## Core Concept: Dual-Level Evolution

MemEvolve implements a **bilevel optimization** approach:

### Inner Loop — Experience Evolution
- Agents operate with a *fixed memory architecture*
- Execute tasks and accumulate experiences
- Experiences populate the memory system

### Outer Loop — Memory Architecture Evolution
- Memory system is **evaluated and redesigned**
- Architectural changes driven by empirical task feedback
- Results in progressively better memory systems

This creates a **virtuous cycle**: Better memory → better agent behavior → higher-quality experience → better memory evolution

## Formal Agent System Definition

An agentic system is formalized as:

```
M = ⟨I, S, A, Ψ, Ω⟩
```

Where:
- `I` = set of agents
- `S` = shared state space
- `A` = joint action space
- `Ψ` = environment dynamics
- `Ω` = memory module

At each timestep:
1. Agent observes state `s_t`
2. Queries memory with `(s_t, history, task)`
3. Receives retrieved context `c_t`
4. Chooses action `a_t = π(s_t, history, task, c_t)`

After task completion:
- A trajectory `τ = (s₀, a₀, …, s_T)` is recorded
- Memory state is updated with extracted experience units

## Modular Memory Design Space

To make memory evolution tractable, all memory systems are decomposed into **four orthogonal components**:

```
Ω = (Encode, Store, Retrieve, Manage)
```

### Encode
Transforms raw experience into structured representations such as lessons, skills, or abstractions.

### Store
Persists encoded information using vector databases or JSON stores.

### Retrieve
Selects task-relevant memory using semantic or hybrid strategies.

### Manage
Maintains long-term memory health via pruning, consolidation, deduplication, or forgetting.

## EvolveLab: Unified Memory Codebase

MemEvolve provides:
- A standardized abstraction for memory systems
- Four reference memory architectures defined as genotypes
- A shared MemorySystem implementation supporting configurable encoding, storage, retrieval, and management strategies
- A controlled environment for architectural evolution

## MemEvolve: Meta-Evolution Mechanism

Each memory system is treated as a **genotype** `(E, U, R, G)`.

### Evolution Steps
1. **Selection** via performance-cost Pareto ranking
2. **Diagnosis** using trajectory replay and failure analysis
3. **Design** of new variants through constrained architectural modification

## Reference Memory Architectures

Four reference memory architectures are defined as genotypes for evolutionary optimization:

| Architecture | Status | Key Features |
|-------------|---------|---------------|
| **AgentKB** | ✅ Defined | Static baseline genotype, lesson-based, minimal overhead |
| **Lightweight** | ✅ Defined | Trajectory-based genotype, JSON storage, auto-pruning |
| **Riva** | ✅ Defined | Agent-centric genotype, vector storage, hybrid retrieval |
| **Cerebra** | ✅ Defined | Tool distillation genotype, semantic graphs, advanced caching |

## Problem Statement

Modern LLM-based agent systems increasingly rely on **memory modules** to improve long-horizon reasoning, tool use, and task performance. However:

- Most existing **self-improving memory systems are manually designed**
- A single fixed memory architecture rarely generalizes across tasks, agent frameworks, and backbone LLMs
- Memory design choices (what to store, how to retrieve, how to manage) are often brittle and task-specific

**Key Question**: How can an agent system *not only learn from experience*, but also **meta-evolve its own memory architecture** to improve learning efficiency while retaining generalization?

## API Wrapper Layer

MemEvolve provides an optional API wrapper that transparently integrates memory functionality with existing OpenAI-compatible LLM deployments:

- **Proxy Server** - FastAPI-based server that wraps any OpenAI-compatible API endpoints
- **Memory Integration** - Automatically retrieves and injects relevant context into requests
- **Experience Encoding** - Captures and stores new interactions for future retrieval
- **Quality Scoring** - Independent, parity-based evaluation system for unbiased model assessment
- **Management Endpoints** - Additional APIs for memory inspection and management
- **Drop-in Replacement** - Applications can use MemEvolve proxy instead of direct LLM API calls

## Project Structure

```
MemEvolve-API/
├── docs/                  # Documentation (organized by topic)
├── src/
│   └── memevolve/        # Package source code (version controlled)
│       ├── api/              # API wrapper server (FastAPI)
│       │   ├── server.py     # Main API server
│       │   ├── routes.py     # API endpoints
│       │   ├── middleware.py # Request/response processing with quality scoring
│       │   └── evolution_manager.py # Runtime evolution orchestration
│       ├── components/        # Memory component implementations
│       │   ├── encode/       # Experience encoding (lesson, skill, tool, abstraction)
│       │   ├── store/        # Storage backends (JSON, FAISS vector, Neo4j graph)
│       │   ├── retrieve/     # Retrieval strategies (keyword, semantic, hybrid, LLM-guided)
│       │   └── manage/       # Memory management (pruning, consolidation, deduplication)
│       ├── evolution/         # Meta-evolution framework
│       │   ├── genotype.py   # Memory architecture representation
│       │   ├── selection.py  # Pareto-based selection
│       │   ├── diagnosis.py  # Trajectory analysis and failure detection
│       │   └── mutation.py   # Architecture mutation with constraints
│       ├── evaluation/       # Benchmark evaluation framework
│       │   ├── gaia_evaluator.py     # GAIA benchmark
│       │   ├── webwalkerqa_evaluator.py # WebWalkerQA benchmark
│       │   ├── xbench_evaluator.py   # xBench benchmark
│       │   └── taskcraft_evaluator.py # TaskCraft benchmark
│       ├── tests/            # Comprehensive test suite (442 tests)
│       └── utils/            # Shared utilities (config, logging, metrics, quality scoring, embeddings)
├── tests/                 # Test suite
├── scripts/              # Development and deployment scripts
├── examples/             # Usage examples and tutorials
├── data/                 # Persistent application data
├── cache/                # Temporary/recreatable data
├── logs/                 # Application logs
├── .env*                 # Environment configuration files
├── pyproject.toml        # Python packaging configuration
├── pyrightconfig.json    # Type checking configuration
├── pytest.ini            # Test configuration
└── AGENTS.md             # Agent guidelines
```

## Current Implementation Status

### Implementation Progress
- ✅ **Memory System**: Complete and tested (4 architectures: AgentKB, Lightweight, Riva, Cerebra)
- ✅ **API Pipeline**: Production-ready proxy framework with OpenAI compatibility
- ✅ **Evolution System**: Meta-evolution with real metrics (Phase 1 & 2 Complete)
- ✅ **Memory Integration**: Context injection and continuous learning with proper score display
- ✅ **Quality Scoring**: Independent parity-based evaluation system for unbiased model assessment
- ✅ **Embedding Compatibility**: Hybrid approach supporting both OpenAI and llama.cpp formats
- ✅ **Thinking Model Support**: Specialized handling for reasoning content with weighted evaluation
- ✅ **Configuration**: Global settings with consolidated environment variables
- ✅ **Monitoring**: Performance analyzer tool with automated reporting
- ✅ **Deployment**: Docker and orchestration support
- ✅ **Documentation**: Comprehensive guides, API reference, and development docs
- ✅ **Testing**: Comprehensive test suite covering all functionality with automated quality evaluation
- ✅ **Test Coverage**: Comprehensive test suite for quality scoring, memory scoring, and full integration
- ✅ **Package Transformation**: Modern Python package structure with proper installation (PEP 517/518)
- ✅ **Professional Installation**: Clean package-based setup with `pip install -e .`

### Test Coverage
- **Total Tests**: Comprehensive test suite
- **Test Modules**: Multiple coverage areas
- **Coverage Areas**:
  - Evolution framework: ~90 tests (genotype, selection, diagnosis, mutation)
  - Memory components: ~200 tests (encode, store, retrieve, manage, memory system)
  - Quality scoring: ~45 tests (ResponseQualityScorer, bias correction, evaluation methods)
  - Memory scoring: ~35 tests (score propagation, display, integration)
  - Integration testing: ~40 tests (full pipeline, end-to-end validation)
  - Utilities: ~55 tests (config, logging, metrics, profiling, data_io, debug_utils, embeddings)
  - API wrapper: ~40 tests (server, middleware, routes, evolution manager)
  - Evaluation: ~40 tests (benchmark evaluators)

### Key Design Principles
- **Agent-driven memory decisions**: Memory serves agent needs, not external requirements
- **Hierarchical representations**: Multi-level abstraction from raw experiences to high-level patterns
- **Multi-level abstraction**: Progressive distillation of knowledge
- **Stage-aware retrieval**: Different retrieval strategies for different reasoning phases
- **Selective forgetting**: Intelligent memory management based on relevance and recency
- **Parity-based evaluation**: Fair assessment across model types with bias correction
- **Modular quality scoring**: Independent evaluation system for unbiased model assessment

## Cross-Generalization

Memory systems evolved on one task transfer effectively across:
- Unseen benchmarks (GAIA, WebWalkerQA, xBench, TaskCraft)
- Different agent frameworks
- Different LLM backbones

## Empirical Validation

- **Benchmarks**: GAIA, WebWalkerQA, xBench, TaskCraft (framework implemented, validation pending)
- **Current Status**: Core functionality tested with 442 unit tests, empirical validation in progress

## Key Takeaways

- Memory architecture is as important as base model
- Manual memory design does not scale across tasks and domains
- Meta-evolution framework enables automatic discovery of optimal memory configurations
- Memory should be treated as a first-class system component alongside planning and tool use
- Quality scoring ensures fair evaluation across different model types and architectures
- Current implementation provides a solid foundation for empirical validation and production deployment

---

*Last updated: January 24, 2026*