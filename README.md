# MemEvolve-API v2.1.0: Self-Evolving Memory API Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version 2.1](https://img.shields.io/badge/version-2.1--development-yellow.svg)](https://github.com/thephimart/MemEvolve-API/tree/master)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests)

## 🚨 **IMPORTANT: v2.1.0 Development Status - NOT PRODUCTION READY**

This is **v2.1.0 on the master branch in active development**. The core memory system is functional with 75%+ storage success rate. The evolution system requires analysis and implementation.

### **✅ Functional Core Systems**
- **OpenAI-Compatible API**: Chat completions endpoint functional for development
- **Memory Retrieval & Injection**: Context enhancement with growing database of experiences
- **Experience Encoding**: Memory creation and storage operational with schema transformation (75%+ success)
- **Centralized Configuration**: Unified logging and configuration management implemented
- **Core API Proxy**: Drop-in replacement for any OpenAI-compatible LLM service

### **⚠️ Systems Requiring Analysis**
- **Evolution System**: Current state unknown, needs investigation and fixes
- **Management & Analytics**: Framework in place, development pending

### **🔧 Recent Major Improvements**
- **Schema Transformation**: Fixed 100% storage verification failures
- **JSON Repair System**: 9-level fallback for robust LLM response handling
- **Logging Optimization**: 75% log volume reduction, proper level hierarchy
- **Centralized Configuration**: Unified token limits and service management

> **Development Status**: Core memory system functional, evolution system pending analysis. NOT PRODUCTION READY.

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

**Current Version**: v2.1.0 Development - Master branch

### **✅ Functional Core Systems**
- **OpenAI-Compatible API**: Chat completions endpoint operational
- **Memory System**: Growing database with 75%+ storage success rate, semantic retrieval functional
- **Schema Transformation**: Fixed storage verification failures through LLM output conversion
- **JSON Repair System**: 9-level fallback ensures robust response parsing
- **Configuration System**: Centralized logging and unified token management
- **API Proxy Framework**: Transparent request/response processing

### **⚠️ Systems Pending Analysis**
- **Evolution System**: Requires investigation to determine current state and implement fixes
- **Management & Analytics**: Framework exists, development pending after core systems verified

### **📋 Completed Major Improvements (v2.1.0)**
1. ✅ Schema transformation eliminates storage verification failures
2. ✅ 9-level JSON repair system for robust LLM response handling  
3. ✅ Centralized logging with 75% volume reduction and proper level hierarchy
4. ✅ Unified configuration management with service validation
5. ✅ Enhanced error handling and atomic storage operations

For detailed implementation progress, see [development roadmap](docs/development/roadmap.md) and [dev_tasks.md](dev_tasks.md) for current priorities.

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

**Version**: v2.1.0 Active Development - master Branch

### **✅ Core Systems Functional**
- **OpenAI-Compatible API**: Chat completions endpoint operational
- **Memory System**: Four-component architecture with 75%+ storage success rate
- **Schema & JSON Handling**: Robust transformation and repair systems implemented
- **Logging & Configuration**: Centralized management with optimized output
- **API Proxy**: Transparent request/response processing

### **⚠️ Systems Pending Analysis**
- **Evolution System**: Current state unknown, next priority for investigation
- **Management & Analytics**: Framework in place, development pending

### **📋 v2.1.0 Key Improvements Completed**
- **Memory Pipeline**: Fixed 100% storage failures through schema transformation
- **Error Resilience**: 9-level JSON repair system ensures robust LLM response handling
- **Observability**: 75% log volume reduction with proper level hierarchy
- **Configuration**: Unified token limits and service validation system
- **Architecture**: Isolated HTTP session management prevents race conditions

For detailed progress tracking, see [dev_tasks.md](dev_tasks.md) for current priorities and completed work.

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

**⚠️ Version 2.1.0 Development Notice**: This is the master branch in active development. Core memory system is functional (75%+ success rate) with robust error handling and logging. Evolution system requires analysis and implementation. NOT PRODUCTION READY. See [dev_tasks.md](dev_tasks.md) for current priorities and status.

*Last updated: February 10, 2026*
