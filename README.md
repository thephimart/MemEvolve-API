# MemEvolve: A Meta-Evolving Memory Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-362%20passing-brightgreen.svg)](src/tests)

MemEvolve is a meta-evolving memory framework for agent systems that enables LLM-based agents to automatically improve their own memory architecture through dual-level evolution.

## 🔬 Research Background

This implementation is based on the concepts introduced in the paper:

**MemEvolve: Meta-Evolution of Agent Memory Systems**  
📄 [arXiv:2506.10055](https://arxiv.org/abs/2506.10055)  
👥 Authors: Guibin Zhang, Haotian Ren, Chong Zhan, Zhenhong Zhou, Junhao Wang, He Zhu, Wangchunshu Zhou, Shuicheng Yan

## 🚀 Features

## 🧪 Benchmark Evaluation Framework

MemEvolve includes a comprehensive evaluation framework for validating meta-evolving memory architectures across multiple AI agent benchmarks:

### Supported Benchmarks
- **GAIA**: General AI Assistant evaluation (450+ questions, 3 difficulty levels)
- **WebWalkerQA**: Web traversal and information extraction (680 questions across websites)
- **xBench**: Profession-aligned evaluation (recruitment, marketing domains)
- **TaskCraft**: Agentic task completion (36k+ synthetic tasks with tool use)

### Storage Backends
- **JSON Storage**: Simple file-based storage for development
- **FAISS Vector Storage**: High-performance similarity search for embeddings
- **Neo4j Graph Storage**: Relationship-aware storage with graph traversal capabilities

### Evaluation Capabilities
- **Automated Experiment Runner**: Compare all reference architectures across all benchmarks
- **Structured Metrics**: Performance scores, execution timing, memory utilization tracking
- **Statistical Analysis**: Comparative analysis with confidence intervals and significance testing
- **Result Persistence**: JSON reports and human-readable summaries
- **Cross-Validation**: Test generalization across different LLM backbones and task domains

### Quick Evaluation
```bash
# Run baseline experiments across all architectures and benchmarks
python -m src.evaluation.experiment_runner --experiment-type baseline --max-samples 10

# Run specific architecture on specific benchmark
python -m src.evaluation.experiment_runner --experiment-type single --architecture AgentKB --benchmark GAIA
```

- **Dual-Level Evolution**: Inner loop (experience evolution) and outer loop (architecture evolution)
- **Orthogonal Components**: Encode, Store, Retrieve, Manage - fully modular and interchangeable
- **Reference Architectures**: AgentKB, Lightweight, Riva, Cerebra - defined as configurable genotypes
- **Flexible Storage**: JSON, FAISS-based vector, Neo4j graph, and extensible storage backends
- **Multiple Retrieval Strategies**: Keyword, semantic, hybrid, and LLM-guided retrieval approaches
- **Pareto Optimization**: Performance-cost tradeoff analysis for architectural improvements
- **Diagnosis System**: Trajectory analysis with failure detection and improvement suggestions
- **Mutation Engine**: Random and targeted mutation strategies for architecture evolution
- **Comprehensive Testing**: 393 tests covering all components with 10-minute timeout

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

### Environment Configuration

MemEvolve uses environment variables for configuration. Copy `.env.example` to `.env` and update the values for your environment:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

**Key Environment Variables:**
- `MEMEVOLVE_LLM_BASE_URL`: Base URL for LLM API (default: http://localhost:11434/v1)
- `MEMEVOLVE_LLM_API_KEY`: API key for LLM service (leave empty for local services)
- `MEMEVOLVE_EMBEDDING_BASE_URL`: Base URL for embedding API
- `MEMEVOLVE_STORAGE_PATH`: Path for storing memory data
- `MEMEVOLVE_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

The `.env` file is automatically loaded and should not be committed to version control.

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

The following reference architectures are defined as genotypes in the evolution framework:

### AgentKB (Static Baseline)
- Simple lesson-based storage configuration
- Keyword retrieval strategy
- Minimal management overhead

### Lightweight (Trajectory-Based)
- Lesson and skill encoding configuration
- JSON storage with persistence
- Automatic pruning and consolidation

### Riva (Agent-Centric, Domain-Aware)
- Lesson, skill, and abstraction encoding configuration
- Vector storage with semantic retrieval
- Hybrid retrieval with performance optimization

### Cerebra (Tool Distillation)
- Tool and abstraction encoding configuration
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
- ✅ **Reference Architectures**: 100% complete (defined as genotypes)
- ✅ **Utilities & Tooling**: 90% complete (config, logging, embeddings, metrics, profiling, data_io, debug_utils done)
- ✅ **LLM-Guided Retrieval**: 100% complete
- ✅ **Batch Encoding Optimization**: 100% complete
- ✅ **Benchmark Evaluation Framework**: 100% complete (GAIA, WebWalkerQA, xBench, TaskCraft)
- ✅ **Graph Database Backend**: 100% complete (Neo4j with NetworkX fallback)
- ✅ **Documentation**: 85% complete (tutorials, guides, examples)
- ⏳ **Full Benchmark Validation**: 80% complete (infrastructure ready, empirical validation pending)

### Test Coverage
- **Total Tests**: 393
- **Test Modules**: 27
- **Components Tested**: All four memory components + evolution framework + utilities + benchmark evaluation + graph storage
- **Test Timeout**: 600 seconds (10 minutes) required for pytest-timeout plugin

## 📚 Documentation

### Getting Started
- **[Developer Onboarding](docs/developer_onboarding.md)**: Complete setup guide and development workflow
- **[Quick Start Tutorial](docs/tutorials/quick_start.md)**: Step-by-step introduction to MemEvolve
- **[Configuration Guide](docs/configuration_guide.md)**: Detailed configuration options and best practices

### Advanced Usage
- **[Advanced Patterns Tutorial](docs/tutorials/advanced_patterns.md)**: Complex use cases and optimizations
- **[Troubleshooting Guide](docs/troubleshooting.md)**: Common issues and solutions

### Examples
- **[Basic Usage Example](examples/basic_usage.py)**: Core functionality demonstration
- **[Graph Storage Example](examples/graph_store_example.py)**: Relationship-aware memory operations

### API Reference
- **[MemorySystem](src/memory_system.py)**: Main memory system class with comprehensive docstrings
- **[Component APIs](src/components/)**: Detailed documentation for all MemEvolve components

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
