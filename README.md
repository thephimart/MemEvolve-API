# MemEvolve: A Meta-Evolving Memory Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-268%20passing-brightgreen.svg)](src/tests)

MemEvolve is a meta-evolving memory framework for agent systems that enables LLM-based agents to automatically improve their own memory architecture through dual-level evolution.

## 🚀 Features

- **Dual-Level Evolution**: Inner loop (experience evolution) and outer loop (architecture evolution)
- **Orthogonal Components**: Encode, Store, Retrieve, Manage - fully modular and interchangeable
- **Multiple Architectures**: AgentKB, Lightweight, Riva, Cerebra - ready-to-use memory systems
- **Flexible Storage**: JSON, FAISS-based vector, and extensible storage backends
- **Multiple Retrieval Strategies**: Keyword, semantic, and hybrid retrieval approaches
- **Pareto Optimization**: Performance-cost tradeoff analysis for architectural improvements
- **Diagnosis System**: Trajectory analysis with failure detection and improvement suggestions
- **Mutation Engine**: Random and targeted mutation strategies for architecture evolution
- **Comprehensive Testing**: 268 tests covering all components with 10-minute timeout

## 📦 Installation

### Clone Repository
```bash
git clone https://github.com/thephimart/memevolve.git
cd memevolve
```

### Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 🏗️ Architecture

MemEvolve implements a **bilevel optimization** approach:

```
Ω = (Encode, Store, Retrieve, Manage)
```

### Inner Loop - Experience Evolution
- Agents operate with a fixed memory architecture
- Execute tasks and accumulate experiences
- Experiences populate the memory system

### Outer Loop - Memory Architecture Evolution
- Memory system is evaluated and redesigned
- Architectural changes driven by empirical task feedback
- Results in progressively better memory systems

## 💾 Component Responsibilities

| Component | Responsibility | Implementation Status |
|-----------|-------------|----------------------|
| **Encode** | Transforms raw experience into structured representations (lessons, skills, tools, abstractions) | ✅ Complete |
| **Store** | Persists encoded information (JSON, vector databases) | ✅ Complete |
| **Retrieve** | Selects task-relevant memory (semantic, hybrid, LLM-guided) | ✅ Complete |
| **Manage** | Maintains memory health (pruning, consolidation, deduplication) | ✅ Complete |

## 🎯 Memory Architectures

### AgentKB (Static Baseline)
- Simple lesson-based storage
- Keyword retrieval
- Minimal management overhead

### Lightweight (Trajectory-Based)
- Lesson and skill encoding
- JSON storage with persistence
- Automatic pruning and consolidation

### Riva (Agent-Centric, Domain-Aware)
- Lesson, skill, and abstraction encoding
- Vector storage with semantic retrieval
- Hybrid retrieval with performance optimization

### Cerebra (Tool Distillation)
- Tool and abstraction focus
- Advanced vector storage with caching
- Semantic retrieval with quality filtering

## 🧪 Testing

Run the complete test suite (10 minute timeout):
```bash
pytest src/tests/ --timeout=600 -v
```

Run specific test categories:
```bash
# Evolution framework tests
pytest src/tests/test_genotype.py src/tests/test_selection.py --timeout=600 -v
pytest src/tests/test_diagnosis.py src/tests/test_mutation.py --timeout=600 -v

# Component tests
pytest src/tests/test_encode.py src/tests/test_store_base.py --timeout=600 -v
pytest src/tests/test_retrieve_base.py src/tests/test_manage_base.py --timeout=600 -v
```

Code quality checks:
```bash
# Linting
flake8 src/ --max-line-length=100

# Formatting
autopep8 --in-place --recursive src/
```

## 📊 Current Status

### Implementation Progress
- ✅ **Core Memory Components**: 100% complete
- ✅ **Integration & Testing**: 100% complete
- ✅ **Meta-Evolution Mechanism**: 100% complete
- ✅ **Utilities & Tooling**: 50% complete (config, logging, embeddings done; missing: metrics, profiling, dev scripts)
- ⏳ **Documentation**: 30% complete
- ⏳ **Validation & Benchmarks**: 0% complete

### Test Coverage
- **Total Tests**: 268
- **Test Modules**: 18
- **Components Tested**: All four memory components + evolution framework + utilities
- **Test Timeout**: 600 seconds (10 minutes) required for pytest-timeout plugin

## 📖 Documentation

- [**PROJECT.md**](PROJECT.md) - Comprehensive project overview and architecture
- [**TODO.md**](TODO.md) - Development roadmap and progress tracking
- [**AGENTS.md**](AGENTS.md) - Development guidelines for coding agents
- [**MemEvolve_systems_summary.md**](MemEvolve_systems_summary.md) - System specification and design principles

## 🛠️ Development

### Project Structure
```
memevolve/
├── src/
│   ├── components/        # Memory component implementations
│   │   ├── encode/      # Experience encoding
│   │   ├── store/       # Storage backends (JSON, vector)
│   │   ├── retrieve/    # Retrieval strategies
│   │   └── manage/      # Memory management
│   ├── evolution/        # Meta-evolution framework
│   │   ├── genotype.py  # Memory architecture representation
│   │   ├── selection.py # Pareto-based selection
│   │   ├── diagnosis.py # Trajectory analysis
│   │   └── mutation.py  # Architecture mutation
│   └── tests/           # Comprehensive test suite
├── docs/                # Additional documentation
└── examples/            # Usage examples
```

### Key Design Principles
- Agent-driven memory decisions
- Hierarchical representations
- Multi-level abstraction
- Stage-aware retrieval
- Selective forgetting

## 🤝 Contributing

This is a private repository. For development:
1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run tests: `pytest src/tests/ --timeout=600 -v`
4. Commit with descriptive messages
5. Push to branch: `git push origin feature/your-feature`

## 📝 License

MIT License - See LICENSE file for details

## 📧 Contact

- **Repository**: https://github.com/thephimart/memevolve
- **Issues**: https://github.com/thephimart/memevolve/issues

## 🔗 Related Resources

- [PROJECT.md](PROJECT.md) - Detailed architecture and implementation status
- [TODO.md](TODO.md) - Development roadmap
- [AGENTS.md](AGENTS.md) - Development guidelines

---

**Built with ❤️ for meta-evolving AI systems**
