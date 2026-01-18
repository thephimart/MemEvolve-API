# MemEvolve: Project Overview

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
│   │   ├── encode.py      # [IMPLEMENTED] Experience encoding
│   │   ├── encode/        # Additional encoding modules
│   │   ├── store/         # Storage implementations
│   │   ├── retrieve/      # Retrieval strategies
│   │   └── manage/        # Memory management
│   ├── tests/             # Test suite
│   └── utils/             # Utility functions
├── AGENTS.md              # Agent development guidelines
├── MemEvolve_systems_summary.md  # System specification
├── PROJECT.md             # This file
├── README.md              # Brief project description
└── TODO.md                # Development roadmap
```

## Current Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| Encode | ✅ Partial | ExperienceEncoder class implemented |
| Store | ❌ Not started | Storage backend implementations needed |
| Retrieve | ❌ Not started | Retrieval strategy implementations needed |
| Manage | ❌ Not started | Memory management implementations needed |

## Technology Stack

- **Language**: Python 3.12.3
- **LLM Backend**: llama.cpp
- **Vector Storage**: TBD (FAISS, ChromaDB, or similar)
- **Testing**: pytest
- **Code Quality**: flake8, autopep8

## Goals

1. Implement all four memory components (Encode, Store, Retrieve, Manage)
2. Support multiple storage backends (vector DB, JSON, graph)
3. Implement various retrieval strategies (semantic, hybrid, LLM-guided)
4. Build memory management operations (pruning, consolidation, deduplication)
5. Create comprehensive test suite
6. Enable meta-evolution mechanism to discover optimal memory architectures
7. Validate on benchmarks (GAIA, WebWalkerQA, xBench, TaskCraft)

## Key Design Principles

- Agent-driven memory decisions
- Hierarchical representations
- Multi-level abstraction
- Stage-aware retrieval
- Selective forgetting
