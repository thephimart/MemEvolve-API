# MemEvolve-API: Self-Evolving Memory API Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-479+%20passing-brightgreen.svg)](tests)

**This is an API pipeline framework that proxies API requests to OpenAI compatible endpoints providing memory, memory management, and evolves the memory implementation thru mutations to enhance the memory system overtime.**

A production-ready proxy that automatically adds persistent memory to any OpenAI-compatible LLM API. Unlike other memory systems, MemEvolve continuously evolves its own architecture to optimize performance over time.

## 🎯 **Key Differentiators**

- **API Pipeline Framework**: Transparent proxy for any OpenAI-compatible LLM
- **Self-Evolving Memory**: Memory architectures that evolve through mutations
- **Zero Code Changes**: Drop-in replacement requiring no application modifications
- **Research Grounded**: Implementation based on arXiv:2512.18746 paper
- **Production Hardened**: Docker deployment, monitoring, enterprise reliability

## 🔬 Research Background

This implementation is based on the concepts introduced in the paper:

**MemEvolve: Meta-Evolution of Agent Memory Systems**  
📄 [arXiv:2512.18746](https://arxiv.org/abs/2512.18746)  
👥 Authors: Guibin Zhang, Haotian Ren, Chong Zhan, Zhenhong Zhou, Junhao Wang, He Zhu, Wangchunshu Zhou, Shuicheng Yan

If you use MemEvolve in your research, please cite:

```bibtex
@misc{zhang2025memevolvemetaevolutionagentmemory,
      title={MemEvolve: Meta-Evolution of Agent Memory Systems},
      author={Guibin Zhang and Haotian Ren and Chong Zhan and Zhenhong Zhou and Junhao Wang and He Zhu and Wangchunshu Zhou and Shuicheng Yan},
      year={2025},
      eprint={2512.18746},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.18746},
      }
```

## 🚀 Features

- **API Pipeline Framework**: Transparent proxy that intercepts and enhances OpenAI-compatible API requests
- **Self-Evolving Memory**: Memory system that evolves through mutations to continuously improve performance
- **Zero Code Changes**: Drop-in replacement for existing LLM APIs - just change the endpoint URL
- **Automatic Memory Injection**: Retrieves and injects relevant context into every API request
- **Continuous Learning**: Every interaction trains and improves memory system over time
- **Universal Compatibility**: Works with any OpenAI-compatible service (llama.cpp, vLLM, OpenAI, Anthropic, etc.)
- **Production Hardened**: Docker deployment, health monitoring, circuit breakers, and enterprise reliability
- **Memory Management APIs**: Full REST API for inspecting, searching, and managing stored memories
- **Real-time Health Dashboard**: Web-based monitoring interface at `/dashboard` with live metrics and performance analytics
- **Quality Scoring System**: Independent, parity-based evaluation ensuring fair assessment across all model types
- **Thinking Model Support**: Specialized handling for models with reasoning content and thinking processes

## 🏆 **Recent Accomplishments**

### Phase 1: Critical Evolution Fixes ✅ **COMPLETE**
- **Evolution System**: Fixed encoding strategies, response timing, quality scoring, and fitness evaluation
- **Real Metrics**: Implemented actual performance tracking instead of placeholder values
- **Meta-Evolution**: System now performs meaningful optimization of memory architectures
- **Production Status**: API pipeline fully functional with active evolution capabilities

### Repository & Documentation Updates
- **Repository Renamed**: `MemEvolve-API` for clear API focus differentiation
- **Documentation Reorganized**: Comprehensive docs with clear navigation and user guides
- **Data Organization**: Centralized directory structure for backup and maintenance
- **Configuration Simplified**: Streamlined environment variables and settings

### Phase 2: Production Polish ✅ **COMPLETE**
- **Configuration Consolidation**: Global max_retries and top_k settings for all APIs
- **Performance Analyzer**: Comprehensive monitoring and automated reporting tool
- **Server Stability**: Clean startup with all components properly initialized
- **Enterprise Ready**: Production-grade reliability and monitoring capabilities

### Phase 3: Quality & Reliability ✅ **COMPLETE**
- **Memory Scoring Fix**: Resolved "score: N/A" issue - retrieved memories now display actual relevance scores
- **Parity-Based Quality Scoring**: Independent, unbiased evaluation system for fair model assessment
- **Enhanced Embedding Compatibility**: Improved llama.cpp integration with automatic format detection
- **Thinking Model Support**: Proper handling of reasoning content with quality-weighted evaluation

## 🌟 How It Works

### API Pipeline Flow
1. **Intercept Requests**: MemEvolve proxy receives API calls intended for your LLM service
2. **Memory Retrieval**: Queries the evolving memory system for relevant context
3. **Context Injection**: Enhances prompts with retrieved memories before forwarding to LLM
4. **Response Processing**: Captures responses and extracts new experiences for memory storage
5. **Continuous Evolution**: Background evolution system optimizes memory architecture over time

### Self-Evolution Process
- **Inner Loop**: Memory system operates and accumulates experiences from API traffic
- **Outer Loop**: Evolution framework mutates memory architectures and selects optimal configurations
- **Continuous Improvement**: System gets better at context retrieval and response enhancement automatically

## 📊 **Monitoring & Analysis Tools**

### Performance Analyzer
MemEvolve includes comprehensive monitoring tools for production deployments:

```bash
# Analyze last 24 hours of system performance
python scripts/performance_analyzer.py --days 1

# Generate detailed performance reports with actionable insights
# - API performance metrics and bottlenecks
# - Memory system efficiency and utilization
# - Evolution progress and optimization trends
# - Quality scoring analysis and recommendations
```

**Key Features:**
- ✅ **No LLM Dependencies** - Pure Python analysis of local logs and data
- ✅ **Actionable Insights** - Identifies performance bottlenecks and optimization opportunities
- ✅ **Historical Analysis** - Analyze any time period from system logs
- ✅ **Automated Reporting** - Generate reports for monitoring dashboards and alerts

**See [Performance Analyzer Documentation](docs/tools/performance_analyzer.md)** for detailed usage and examples.

## 📊 Example Enhancement

**Before (Direct LLM):**
```json
{"messages": [{"role": "user", "content": "How do I debug Python memory leaks?"}]}
```

**After (With MemEvolve):**
```json
{
  "messages": [
    {"role": "system", "content": "Relevant past experiences:\n• Memory profiling with tracemalloc (relevance: 0.89)\n• GC monitoring techniques (relevance: 0.76)"},
    {"role": "user", "content": "How do I debug Python memory leaks?"}
  ]
}
```

## 🚀 Quick Start

See the [Getting Started Guide](docs/user-guide/getting-started.md) for detailed setup instructions.

**TL;DR:**
```bash
git clone https://github.com/thephimart/MemEvolve-API.git
cd MemEvolve-API
pip install -e .
cp .env.example .env  # Configure your LLM endpoint
python scripts/start_api.py
# Point your apps to http://localhost:11436/v1
```



## 📦 Installation (Detailed)

### Prerequisites
- **Python**: 3.10+ (developed on 3.12+, tested on 3.12+ and 3.10+; compatible with 3.7+ untested)
- **LLM API**: Access to any OpenAI-compatible API (vLLM, Ollama, OpenAI, etc.) with embedding support
- **API Endpoints**: 1-3 endpoints (can be the same service or separate):
  - **Minimum: 1 endpoint** (must support both chat completions and embeddings)
  - **Recommended: 3 separate endpoints** for optimal performance:
    - **Upstream API**: Primary LLM service for chat completions and user interactions
    - **LLM API**: Dedicated LLM service for memory encoding and processing (can reuse upstream)
    - **Embedding API**: Service for creating vector embeddings of memories (can reuse upstream)

**Why separate endpoints?** Using dedicated services prevents distracting your main LLM with embedding and memory management tasks, while lightweight task-focused models improve efficiency and reduce latency.

### Standard Port Assignments

For consistency in examples and documentation, MemEvolve uses these standard port assignments:

| Service | Port | Environment Variable | Purpose |
|---------|------|---------------------|---------|
| **MemEvolve API** | `11436` | - | Main API proxy server |
| **Upstream LLM** | `11434` | `MEMEVOLVE_UPSTREAM_BASE_URL` | Primary chat completions |
| **Memory LLM** | `11433` | `MEMEVOLVE_MEMORY_BASE_URL` | Memory encoding/processing |
| **Embedding API** | `11435` | `MEMEVOLVE_EMBEDDING_BASE_URL` | Vector embeddings |

**Example:** `http://localhost:11434/v1` for upstream, `http://localhost:11433/v1` for memory LLM.

### Tested and Working Configurations

MemEvolve has been tested with the following model configurations:

**Upstream LLM** (primary chat completions):
- **llama.cpp** with GPT-OSS-20B (GGUF, MXFP4) ✅ Tested and working
- **llama.cpp** with GLM-4.6V-Flash (GGUF, Q5_K_M) ✅ Tested and working
- **llama.cpp** with Falcon-H1R-7B (GGUF, Q5_K_M) ✅ Tested and working
- **llama.cpp** with Qwen3-VL-30B-A3B-Thinking (GGUF, BF16) ✅ Tested and working
- **llama.cpp** with LFM-2.5-1.2B-Instruct (GGUF, BF16) ✅ Tested and working

**Memory LLM** (encoding and processing - configured via `MEMEVOLVE_MEMORY_*` variables):
- **llama.cpp** with LFM-2.5-1.2B-Instruct (GGUF, BF16) ✅ Tested and working

**Embedding API** (vector embeddings):
- **llama.cpp** with nomic-embed-text-v2-moe (GGUF, Q5_K_M) ✅ Tested and working

*Note: The current running configuration demonstrates optimal separation of concerns with specialized models for each function: large model for chat completions, efficient model for memory processing, and dedicated embedding model.*

**Thinking/Reasoning Models**: Models with thinking/reasoning capabilities are fully supported. MemEvolve properly handles `reasoning_content` and `content` separation for memory encoding with parity-based quality scoring (70% answer + 30% reasoning evaluation).

### Setup
```bash
git clone https://github.com/thephimart/MemEvolve-API.git
cd MemEvolve-API
pip install -e .
cp .env.example .env
# Edit .env with your API endpoints:
# - MEMEVOLVE_UPSTREAM_BASE_URL (required)
# - MEMEVOLVE_EMBEDDING_BASE_URL (auto-detected for common setups)
```



## 🏗️ How It Works

MemEvolve consists of four core memory components working together:

### Memory Pipeline
```
User Request → Memory Retrieval → LLM Processing → Response + Learning → Memory Storage
```

### Components
- **Encode**: Transforms conversations into structured memories (lessons, skills, insights)
- **Store**: Persists memories using vector databases for fast similarity search
- **Retrieve**: Finds relevant memories based on conversation context
- **Manage**: Maintains memory health through pruning and consolidation

### Evolution System (Production Ready)

**Current Status**:
- ✅ **Fully Operational**: Meta-evolution with real performance optimization
- ✅ **Evolution Cycles**: Active genotype evaluation and selection
- ✅ **Fitness Evaluation**: Meaningful differentiation between architectures
- ✅ **Performance Metrics**: Real response timing and quality scoring
- ✅ **Production Safe**: Circuit breakers, monitoring, and rollback capabilities
MemEvolve includes a functional meta-evolution framework that automatically optimizes memory architectures:
- **Implemented**: Component hot-swapping, genotype application, fitness evaluation
- **In Progress**: Safe evolution cycles (shadow mode, circuit breakers, staged rollout)
- **Architecture-Level Evolution**: Optimizes memory strategies, storage backends, retrieval methods
- **Model Constraints**: Respects fixed embedding dimensions and context windows
- **Inner Loop**: Agent operates with fixed memory system, accumulates experiences
- **Outer Loop**: Memory system architecture evolves based on empirical performance feedback
- **Result**: Self-improving memory systems that adapt to your specific use case

### API Requirements
MemEvolve needs AI services for:
- **Upstream API**: Chat completions and user interactions (e.g., llama.cpp, vLLM, OpenAI)
- **LLM API**: Encoding experiences (defaults to Upstream API if not specified)
- **Embedding API**: Vectorizing memories for semantic search (defaults to Upstream API if not specified, Upstream API must have embedding capabilities)

**Configuration Options:**
- **Single endpoint**: Use one service for everything (simplest setup)
- **Dual endpoints**: Separate chat vs memory processing (better performance)
- **Triple endpoints**: Fully dedicated services (optimal performance and specialization)

### Smart Integration
- **Context Injection**: Relevant memories added to system prompts
- **Continuous Learning**: Every interaction improves future responses
- **Automatic Management**: Memory stays optimized without manual intervention

## 💾 Component Responsibilities

| Component | Responsibility | Implementation Status |
|-----------|-------------|----------------------|
| **Encode** | Transforms raw experience into structured representations (lessons, skills, tools, abstractions) | ✅ Complete |
| **Store** | Persists encoded information (JSON, vector databases) | ✅ Complete |
| **Retrieve** | Selects task-relevant memory (semantic, hybrid, LLM-guided) | ✅ Complete |
| **Manage** | Maintains memory health (pruning, consolidation, deduplication) | ✅ Complete |



## 🧪 Testing

Run the API wrapper test suite:
```bash
pytest tests/test_api_server.py -v
```

Run all tests:
```bash
pytest tests/ -v
```

Code quality:
```bash
flake8 src/memevolve/ --max-line-length=100 --extend-ignore=E203,W503
```

Format code:
```bash
autopep8 --in-place --recursive --max-line-length=100 --aggressive --aggressive src/memevolve/
```

## 📊 Current Status

### Implementation Progress
- ✅ **Memory System**: Complete and tested (4 architectures: AgentKB, Lightweight, Riva, Cerebra)
- ✅ **API Pipeline**: Production-ready proxy framework with OpenAI compatibility
- ✅ **Evolution System**: Meta-evolution with real metrics (Phase 1 & 2 Complete)
- ✅ **Memory Integration**: Context injection and continuous learning with proper score display
- ✅ **Quality Scoring**: Independent parity-based evaluation system for unbiased model assessment
- ✅ **Configuration**: Global settings with consolidated environment variables
- ✅ **Monitoring**: Performance analyzer tool with automated reporting
- ✅ **Deployment**: Docker and orchestration support
- ✅ **Documentation**: Comprehensive guides, API reference, and development docs
- ✅ **Testing**: 479+ tests covering all functionality with automated quality evaluation

### Test Coverage & Performance
- **Total Tests**: 479+ comprehensive test suite
- **API Tests**: 10+ integration tests covering full pipeline
- **Evolution Tests**: 6+ tests for multi-architecture fitness evaluation
- **Memory Tests**: 15+ test files covering components and 4 reference architectures
- **Performance**: Real-world API performance with detailed timing analysis
- **Evolution**: Active meta-evolution with meaningful fitness optimization
- **Monitoring**: Automated performance reports and bottleneck detection

### Key Accomplishments
- **Phase 1 Complete**: All critical evolution fixes implemented
- **Phase 2 Complete**: Production polish with monitoring and configuration consolidation
- **Phase 3 Complete**: Memory scoring fixes and quality scoring system implementation
- **Real Metrics**: Performance timing, quality scoring, utilization tracking
- **Production Ready**: Enterprise-grade API proxy with comprehensive monitoring
- **Performance Analyzed**: 200-run testing with detailed timing breakdown
- **Quality System**: Parity-based scoring ensuring fair evaluation across model types
- **Research Grounded**: Implementation based on arXiv:2512.18746

## 📚 Documentation

Complete documentation is organized by topic in the [`docs/`](docs/index.md) directory:

### 🚀 User Guides
- **[Getting Started](docs/user-guide/getting-started.md)** - Quick setup and first steps
- **[Configuration](docs/user-guide/configuration.md)** - Environment setup and options
- **[Deployment](docs/user-guide/deployment.md)** - Docker and production deployment

### 🔧 Technical Reference
- **[API Reference](docs/api/api-reference.md)** - All endpoints and configuration options
- **[Troubleshooting](docs/api/troubleshooting.md)** - Common issues and solutions

### 🛠️ Development
- **[Architecture](docs/development/architecture.md)** - System design and implementation
- **[Evolution System](docs/development/evolution.md)** - Meta-evolution framework details
- **[Roadmap](docs/development/roadmap.md)** - Development priorities and progress
- **[Scripts](docs/development/scripts.md)** - Build and maintenance tools
- **[Agent Guidelines](AGENTS.md)** - Development guidelines

### 📖 Tutorials
- **[Advanced Patterns](docs/tutorials/advanced-patterns.md)** - Complex memory architectures

## 🛠️ Development

### Project Structure
```
MemEvolve-API/
  ├── src/
  │   └── memevolve/        # Package source code (version controlled)
  │       ├── api/             # API pipeline framework
  │       │   ├── server.py    # FastAPI proxy server with OpenAI compatibility
  │       │   ├── routes.py    # Memory management and evolution endpoints
  │       │   └── middleware.py # Memory integration and quality evaluation
  │       ├── components/        # Memory component implementations
  │       │   ├── encode/      # Experience encoding
  │       │   ├── store/       # Storage backends (JSON, vector)
  │       │   ├── retrieve/    # Retrieval strategies
  │       │   └── manage/      # Memory management
  │       ├── evolution/        # Meta-evolution framework
  │       │   ├── genotype.py  # Memory architecture representation
  │       │   ├── selection.py # Pareto-based selection
  │       │   ├── diagnosis.py # Trajectory analysis
  │       │   └── mutation.py  # Architecture mutation
  │       ├── utils/           # Configuration, logging, embeddings
  │       └── evaluation/       # Benchmark evaluation framework
  ├── tests/                # Test suite
  ├── scripts/              # Startup and deployment scripts
  ├── docs/                 # Documentation (organized by topic)
  ├── examples/             # Usage examples
  ├── data/                 # Persistent application data
  ├── cache/                # Temporary/recreatable data
  ├── logs/                 # Application logs
  ├── .env*                 # Environment configuration files
  ├── pyproject.toml        # Python packaging configuration
  ├── pyrightconfig.json    # Type checking configuration
  ├── pytest.ini            # Test configuration
  └── AGENTS.md             # Agent guidelines
```

### Key Design Principles
- Agent-driven memory decisions
- Hierarchical representations
- Multi-level abstraction
- Stage-aware retrieval
- Selective forgetting

## 🤝 Contributing

This is a public repository. For development:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Commit with descriptive messages
6. Push to your fork and create a pull request

## 📝 License

MIT License - See LICENSE file for details

## 📧 Contact

- **Repository**: https://github.com/thephimart/MemEvolve-API
- **Issues**: https://github.com/thephimart/MemEvolve-API/issues

## 🔗 Related Resources

- [Documentation Index](docs/index.md) - Complete documentation overview
- [Development Roadmap](docs/development/roadmap.md) - Current priorities and progress
- [Architecture Overview](docs/development/architecture.md) - System design details
- [Evolution System](docs/development/evolution.md) - Meta-evolution framework

---

*Last updated: January 25, 2026*