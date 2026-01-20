# MemEvolve: Memory-Enhanced LLM API Proxy

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-400+%20passing-brightgreen.svg)](src/tests)

MemEvolve adds persistent memory capabilities to any OpenAI-compatible LLM API. Drop-in memory functionality for existing LLM deployments - no code changes required.

## 🔬 Research Background

This implementation is based on the concepts introduced in the paper:

**MemEvolve: Meta-Evolution of Agent Memory Systems**  
📄 [arXiv:2506.10055](https://arxiv.org/abs/2506.10055)  
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

## 🚀 Quick Start (5 minutes)

### 1. Install & Configure

```bash
# Clone and setup
git clone https://github.com/thephimart/memevolve.git
cd memevolve
pip install -r requirements.txt

# Configure your LLM API
cp .env.example .env
# Edit .env - set MEMEVOLVE_UPSTREAM_BASE_URL (embeddings default to same endpoint)
```

### 2. Start MemEvolve Proxy

```bash
# Start the memory-enhanced proxy (auto-reload disabled by default)
python scripts/start_api.py

# For development with auto-reload (shows file change notifications)
python scripts/start_api.py --reload
```

### 3. Point Your Apps to MemEvolve

```python
# Change your existing OpenAI client:
client = OpenAI(
    base_url="http://localhost:11436/v1",  # Was: your-llm-url/v1
    api_key="dummy"  # API key handled by proxy
)
```

**That's it!** MemEvolve automatically adds memory to all your LLM interactions.

### 🎨 Try the Web Interface

For an immediate demo, use the included Streamlit web interface:

```bash
# In another terminal (MemEvolve server must be running)
cd webui
pip install -r requirements.txt
streamlit run main.py
```

Open `http://localhost:11437` for a chat interface that automatically uses MemEvolve's memory features!

## 📦 Installation (Detailed)

### Prerequisites
- **Python**: 3.12 or higher
- **LLM API**: Access to any OpenAI-compatible API (vLLM, Ollama, OpenAI, etc.) with embedding support
- **Three API Endpoints** (can be the same service or separate):
  - **Upstream API**: Primary LLM service for chat completions and user interactions
  - **LLM API**: Dedicated LLM service for memory encoding and processing (can reuse upstream)
  - **Embedding API**: Service for creating vector embeddings of memories (can reuse upstream)

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

### API Requirements
MemEvolve needs AI services for:
- **LLM API**: Chat completions and encoding experiences (e.g., llama.cpp, vLLM, OpenAI)
- **Embedding API**: Vectorizing memories for semantic search (defaults to same as LLM endpoint)

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
- **Total Tests**: 400+
- **API Tests**: 9 comprehensive integration tests
- **Memory Tests**: Full component coverage
- **Performance**: <200ms latency overhead verified

## 📚 Documentation

### Getting Started
- **[Getting Started Guide](docs/getting-started.md)**: Complete setup and usage guide
- **[API Reference](docs/api-reference.md)**: All endpoints and configuration options
- **[Deployment Guide](docs/deployment.md)**: Docker and production deployment

### Configuration & Troubleshooting
- **[Configuration Guide](docs/configuration.md)**: Environment setup and options
- **[Troubleshooting Guide](docs/troubleshooting.md)**: Common issues and solutions

### Advanced Topics
- **[Advanced Patterns](docs/tutorials/advanced_patterns.md)**: Complex memory architectures

## 📖 Technical Documentation

- [**PROJECT.md**](PROJECT.md) - Technical architecture and implementation
- [**TODO.md**](TODO.md) - Development roadmap
- [**AGENTS.md**](AGENTS.md) - Development guidelines

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