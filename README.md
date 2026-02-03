# MemEvolve-API v2.0.0: Self-Evolving Memory API Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version 2.0](https://img.shields.io/badge/version-2.0--development-orange.svg)](https://github.com/thephimart/MemEvolve-API/tree/master)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests)

## 🚨 **IMPORTANT: v2.0.0 Development Status**

This is **v2.0.0 on the master branch in active development**. The core systems are functional with working evolution and memory capabilities.

### **✅ Fully Functional Core Systems**
- **OpenAI-Compatible API**: Chat completions endpoint fully functional for production use
- **Memory Retrieval & Injection**: Automatic context enhancement with 356+ stored experiences
- **Experience Encoding**: Memory creation and storage operational with semantic scoring
- **Evolution System**: Working fitness calculations and boundary-compliant mutations
- **Core API Proxy**: Drop-in replacement for any OpenAI-compatible LLM service

### **🔧 Management & Analytics (In Development)**
- **Management API Endpoints**: Partial functionality, some endpoints incomplete
- **Dashboard Features**: Basic endpoints working, advanced analytics in development
- **Business Analytics**: Framework in place, ROI tracking being enhanced

### **⚠️ Minor Issues Being Refined**
- **Memory Encoding**: Content quality improvements in progress
- **Token Efficiency**: Analytics calculations being refined
- **Code Quality**: Line length violations (non-blocking)

> **Production Status**: Core memory and evolution systems ready for use. Management endpoints are functional but incomplete.

---

An API pipeline framework that proxies requests to OpenAI-compatible endpoints, providing persistent memory and continuous architectural evolution through mutations.

**Key capabilities**: Transparent API proxy, self-evolving memory systems, zero-code integration, research-based implementation (arXiv:2512.18746), and development deployment.

## 🔬 Research Background

Based on **MemEvolve: Meta-Evolution of Agent Memory Systems** (arXiv:2512.18746). See [complete research details](docs/development/evolution.md) for implementation specifics and citation information.

## 🚀 Features

- **API Proxy**: Transparent interception of OpenAI-compatible requests
- **Self-Evolving Memory**: Working architectural optimization through mutations with fitness evaluation
- **Auto-Evolution**: Request-based and periodic evolution triggers with boundary validation
- **Memory Management**: 356+ stored experiences with semantic retrieval and relevance filtering
- **Zero Integration**: Drop-in replacement - just change endpoint URL
- **Memory Injection**: Automatic context enhancement for all requests
- **Universal Compatibility**: Works with any OpenAI-compatible service
- **Quality Scoring**: Working relevance and quality evaluation system

For detailed feature documentation, see the [complete feature list](docs/index.md#key-features).

## 📝 Centralized Logging

MemEvolve features a comprehensive logging system with component-specific event routing for enhanced observability. The system routes events to dedicated log files with fine-grained control.

**Quick Overview:**
- **API Server**: HTTP requests → `logs/api-server/api_server.log`
- **Middleware**: Request processing → `logs/middleware/enhanced_middleware.log`  
- **Memory**: Core operations → `logs/memory/memory.log`
- **Evolution**: Parameter tracking → `logs/evolution/evolution.log`
- **System**: Application events → `logs/memevolve.log`

For complete configuration, usage, and troubleshooting details, see [Centralized Logging Guide](docs/user-guide/centralized-logging.md).

## 📊 Implementation Status

**Current Version**: v2.0.0 Development - Master branch in active development

### **✅ Fully Functional Core Systems**
- **OpenAI-Compatible API**: Chat completions endpoint fully operational
- **Memory System**: 356+ experiences stored with semantic retrieval and relevance filtering
- **Evolution System**: Working fitness calculations and boundary-compliant mutations
- **Quality Scoring**: Functional relevance and quality evaluation
- **Configuration System**: 137 environment variables with centralized management and component-specific logging
- **API Proxy Framework**: Transparent request/response processing

### **🔧 Management & Analytics (In Development)**
- **Management API Endpoints**: Basic functionality, advanced features incomplete
- **Dashboard Features**: Framework in place, enhanced analytics being developed
- **Business Analytics**: ROI tracking structure, real-time metrics in progress

### **📋 Current Focus Areas**
1. Complete management API endpoint functionality
2. Enhance memory encoding content quality
3. Develop advanced dashboard analytics
4. Refine business impact scoring

For detailed implementation progress, see [development roadmap](docs/development/roadmap.md) and [known issues](docs/api/troubleshooting.md#known-issues-in-v20).

## 🌟 How It Works

**API Pipeline**: Request interception → Memory retrieval → Context injection → LLM processing → Response learning → Continuous evolution

**Self-Evolution**: Inner loop (memory operation) + Outer loop (architectural optimization) = Continuous performance improvement

For detailed architecture and evolution mechanics, see [system architecture](docs/development/architecture.md) and [evolution framework](docs/development/evolution.md).

## 📊 Monitoring & Analytics

MemEvolve provides comprehensive monitoring tools:

- **Business Impact Analyzer**: Executive-level ROI validation and business intelligence
- **Performance Analyzer**: System monitoring and bottleneck identification  
- **Real-time Dashboard**: `/dashboard-data` endpoint with live metrics

See [monitoring documentation](docs/tools/) for detailed usage guides.

## 📄 Example

**Before (Direct LLM)**:
```json
{"messages": [{"role": "user", "content": "How do I debug Python memory leaks?"}]}
```

**After (With MemEvolve)**:
```json
{
  "messages": [
    {"role": "system", "content": "Relevant past experiences:\n• Memory profiling with tracemalloc (relevance: 0.89)\n• GC monitoring techniques (relevance: 0.76)"},
    {"role": "user", "content": "How do I debug Python memory leaks?"}
  ]
}
```

For more examples and advanced patterns, see [tutorials](docs/tutorials/).

## 🚀 Quick Start

**5-Minute Setup:**
```bash
git clone https://github.com/thephimart/MemEvolve-API.git
cd MemEvolve-API
pip install -e .
cp .env.example .env
# IMPORTANT: Edit .env with your API endpoint (required):
# MEMEVOLVE_UPSTREAM_BASE_URL=https://your-llm-provider.com/v1
# MEMEVOLVE_UPSTREAM_API_KEY=your-api-key
python scripts/start_api.py
# Point your apps to http://localhost:11436/v1
```

**Prerequisites:**
- **Python**: 3.10+ (developed on 3.12+)
- **LLM API**: Any OpenAI-compatible service with embedding support
- **API Endpoints**: 1 endpoint (chat + embeddings) or 3 separate for optimal performance

For detailed installation instructions, port assignments, and tested configurations, see [Getting Started Guide](docs/user-guide/getting-started.md).



## 📦 Installation (Development)

### Prerequisites (Development)

### 🐍 Python & Dependencies

- **Python**: 3.10+ (developed on 3.12+, tested on 3.12+ and 3.10+; compatible with 3.7+ untested)
- **LLM API**: Access to any OpenAI-compatible API (vLLM, Ollama, OpenAI, etc.) with embedding support
- **API Endpoints**: 1-3 endpoints (can be the same service or separate) - Development endpoints only:
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
- **llama.cpp** with LFM-2.5-1.2B-Thinking (GGUF, BF16) ✅ Tested and working
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



## 🏗️ Architecture

**Memory Components**: Encode → Store → Retrieve → Manage (working in pipeline)

**Evolution System**: Multi-trigger automatic optimization with real performance metrics

**API Requirements**: 
- Upstream API (chat completions)
- Memory LLM (encoding, optional)
- Embedding API (vector search, optional)

For complete architecture details, see [system design](docs/development/architecture.md).

## 💾 Components

| Component | Function |
|-----------|----------|
| **Encode** | Experience transformation into structured memories |
| **Store** | Memory persistence (JSON, vector, graph backends) |
| **Retrieve** | Context-relevant memory selection |
| **Manage** | Memory health optimization |

For detailed component documentation, see [architecture guide](docs/development/architecture.md).



## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Code quality checks
./scripts/lint.sh

# Code formatting  
./scripts/format.sh
```

For detailed testing guidelines, see [contributing guide](docs/development/contributing.md).

## 📊 Current Status

**Version**: v2.0.0 Active Development - Master Branch

### **✅ Core Systems Fully Functional**
- **OpenAI-Compatible API**: Chat completions endpoint fully operational
- **Memory System**: Four-component architecture with 356+ stored experiences
- **Evolution System**: Working fitness calculation and boundary-compliant mutations
- **Quality Scoring**: Functional relevance and quality evaluation system
- **API Proxy**: Transparent request/response processing ready

### **🔧 Management & Analytics (In Development)**
- **Management API Endpoints**: Basic functionality implemented
- **Dashboard Features**: Framework operational, advanced features incomplete
- **Business Analytics**: ROI structure in place, real-time metrics in development

### **📋 Minor Refinements in Progress**
- **Memory Encoding**: Content quality improvements being refined
- **Analytics**: Token efficiency and business impact calculations enhanced
- **Code Quality**: Non-blocking style improvements (line length violations)

For detailed progress tracking, see [development roadmap](docs/development/roadmap.md) and [known issues](docs/api/troubleshooting.md#known-issues-in-v20).

## 📚 Documentation

**Complete documentation**: [docs/index.md](docs/index.md)

**Key Guides**:
- [Getting Started](docs/user-guide/getting-started.md) - Quick setup
- [Configuration](docs/user-guide/configuration.md) - 137 environment variables with centralized logging
- [API Reference](docs/api/api-reference.md) - Endpoints and options
- [Architecture](docs/development/architecture.md) - System design
- [Development](docs/development/roadmap.md) - Contributing guidelines

## 🛠️ Development

**Structure**: API proxy, memory components, evolution framework, utilities, and comprehensive testing

**Development Guidelines**: See [AGENTS.md](AGENTS.md) for coding standards and [contributing guide](docs/development/contributing.md) for workflow.

For complete project structure and design principles, see [architecture documentation](docs/development/architecture.md).

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes and run tests
4. Submit pull request

See [contributing guide](docs/development/contributing.md) for detailed guidelines.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

## 🔗 Resources

- **Repository**: https://github.com/thephimart/MemEvolve-API
- **Issues**: https://github.com/thephimart/MemEvolve-API/issues
- **Documentation**: [docs/index.md](docs/index.md)

---

---

**⚠️ Version 2.0.0 Development Notice**: This is the master branch in active development. The main API pipeline is fully functional and ready for use. Management endpoints and evolution/scoring systems are in testing and may not function as expected. See [Known Issues](docs/api/troubleshooting.md#known-issues-in-v20) for detailed status.

*Last updated: February 3, 2026*
