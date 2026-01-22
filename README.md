# MemEvolve: Memory-Enhanced LLM API Proxy

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-400+%20passing-brightgreen.svg)](src/tests)

MemEvolve adds persistent memory capabilities to any OpenAI-compatible LLM API. Drop-in memory functionality for existing LLM deployments - no code changes required.

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

- **Drop-in Memory**: Add persistent memory to any OpenAI-compatible LLM API without code changes
- **Transparent Proxy**: Your existing applications work unchanged - just change the API URL
- **Smart Context**: Automatically retrieves and injects relevant memories into conversations
- **Learning System**: Captures insights from every interaction to improve future responses
- **Universal Compatibility**: Works with llama.cpp, vLLM, OpenAI API, Anthropic, and any OpenAI-compatible service
- **Production Ready**: Docker deployment, health monitoring, and enterprise-grade reliability
- **Memory Management**: Full API for inspecting, searching, and managing stored memories

## 🌟 How It Works

1. **Proxy Requests**: MemEvolve sits between your application and your LLM API
2. **Add Context**: Before sending to LLM, retrieves relevant memories from past conversations
3. **Enhanced Responses**: LLM receives conversation history + relevant context
4. **Learn Continuously**: After response, extracts and stores new insights for future use

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

See the [Getting Started Guide](docs/getting-started.md) for detailed setup instructions.

**TL;DR:**
```bash
git clone https://github.com/thephimart/memevolve.git
cd memevolve
pip install -r requirements.txt
cp .env.example .env  # Configure your LLM endpoint
python scripts/start_api.py
# Point your apps to http://localhost:11436/v1
```



## 📦 Installation (Detailed)

### Prerequisites
- **Python**: 3.12 or higher
- **LLM API**: Access to any OpenAI-compatible API (vLLM, Ollama, OpenAI, etc.) with embedding support
- **API Endpoints**: 1-3 endpoints (can be the same service or separate):
  - **Minimum: 1 endpoint** (must support both chat completions and embeddings)
  - **Recommended: 3 separate endpoints** for optimal performance:
    - **Upstream API**: Primary LLM service for chat completions and user interactions
    - **LLM API**: Dedicated LLM service for memory encoding and processing (can reuse upstream)
    - **Embedding API**: Service for creating vector embeddings of memories (can reuse upstream)

**Why separate endpoints?** Using dedicated services prevents distracting your main LLM with embedding and memory management tasks, while lightweight task-focused models improve efficiency and reduce latency.

### Tested and Working Configurations

MemEvolve has been tested with the following model configurations:

**Upstream LLM** (primary chat completions):
- **llama.cpp** with GPT-OSS-20B (GGUF, MXFP4) ✅ Tested and working
- **llama.cpp** with GLM-4.6V-Flash (GGUF, Q5_K_M) ✅ Tested and working
- **llama.cpp** with Falcon-H1R-7B (GGUF, Q5_K_M) ✅ Tested and working
- **llama.cpp** with Qwen3-VL-30B-A3B-Thinking (GGUF, BF16) ✅ Tested and working
- **llama.cpp** with LFM-2.5-1.2B-Instruct (GGUF, BF16) ✅ Tested and working

**Memory LLM** (encoding and processing):
- **llama.cpp** with LFM-2.5-1.2B-Instruct (GGUF, BF16) ✅ Tested and working

**Embedding API** (vector embeddings):
- **llama.cpp** with nomic-embed-text-v2-moe (GGUF, Q5_K_M) ✅ Tested and working

*Note: The current running configuration demonstrates optimal separation of concerns with specialized models for each function: large model for chat completions, efficient model for memory processing, and dedicated embedding model.*

**Thinking/Reasoning Models**: Models with thinking/reasoning capabilities are fully supported. MemEvolve properly handles `reasoning_content` and `content` separation for memory encoding.

### Setup
```bash
git clone https://github.com/thephimart/memevolve.git
cd memevolve
pip install -r requirements.txt
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

### Evolution System (In Testing)

**Current Status**:
- ✅ Evolution cycles functional with component hot-swapping
- ✅ Genotype application logic implemented
- ✅ Fitness evaluation with rolling windows
- ✅ Core memory functionality working (injection + encoding)
- ✅ Hybrid streaming mode operational
- 🔄 **In testing phase**: System undergoing active development and evaluation
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
pytest src/tests/test_api_server.py -v
```

Run all tests:
```bash
pytest src/tests/ -v
```

Code quality:
```bash
flake8 src/ --max-line-length=100
```

## 📊 Current Status

### Implementation Progress
- ✅ **Memory System**: Complete and tested
- ✅ **API Wrapper**: Production-ready proxy server
- ✅ **Memory Integration**: Context injection and learning
- ✅ **Configuration**: Simple .env-based setup
- ✅ **Deployment**: Docker and orchestration support
- ✅ **Documentation**: API wrapper guides and examples
- ✅ **Testing**: 400+ tests covering all functionality

### Test Coverage
- **Total Tests**: 442
- **API Tests**: 9 comprehensive integration tests
- **Memory Tests**: Full component coverage
- **Performance**: <200ms latency overhead verified

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
memevolve/
  ├── src/
  │   ├── api/             # API wrapper server
  │   │   ├── server.py    # FastAPI server with proxy endpoints
  │   │   ├── routes.py    # Memory management endpoints
  │   │   └── middleware.py # Memory integration middleware
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
  │   ├── tests/           # Comprehensive test suite
  │   └── utils/           # Configuration, logging, embeddings
  ├── scripts/             # Startup and deployment scripts
  ├── docs/                # Comprehensive documentation
  └── examples/            # Usage examples
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
4. Run tests: `pytest src/tests/ -v`
5. Commit with descriptive messages
6. Push to your fork and create a pull request

## 📝 License

MIT License - See LICENSE file for details

## 📧 Contact

- **Repository**: https://github.com/thephimart/memevolve
- **Issues**: https://github.com/thephimart/memevolve/issues

## 🔗 Related Resources

- [Documentation Index](docs/index.md) - Complete documentation overview
- [Development Roadmap](docs/development/roadmap.md) - Current priorities and progress
- [Architecture Overview](docs/development/architecture.md) - System design details
- [Evolution System](docs/development/evolution.md) - Meta-evolution framework

---

*Last updated: January 22, 2026*