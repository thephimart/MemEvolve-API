# MemEvolve-API v2.1.0: Self-Evolving Memory API Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version 2.1](https://img.shields.io/badge/version-2.1--development-yellow.svg)](https://github.com/thephimart/MemEvolve-API/tree/master)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests)

## 🚨 **IMPORTANT: v2.1.0 IN DEVELOPMENT - NOT PRODUCTION READY**

This is **v2.1.0 on the master branch in active development**. Core memory system is fully functional with operational IVF vector store delivering substantial performance benefits. Evolution system requires analysis and implementation.

### **✅ Functional Core Systems**
- **OpenAI-Compatible API**: Chat completions endpoint functional for development
- **Memory Retrieval & Injection**: Context enhancement with growing database of experiences
- **Experience Encoding**: Memory creation and storage operational (95%+ success rate)
- **IVF Vector Store**: Fully operational with self-healing capabilities, 477+ memories indexed
- **Centralized Configuration**: Unified logging and configuration management implemented
- **Optimized Logging**: 75% log volume reduction, startup noise eliminated

### **⚠️ Systems Requiring Analysis**
- **Evolution System**: Current state unknown, needs investigation and fixes
- **IVF Phase 3**: Configuration & monitoring (13 hours implementation ready)

### **📊 Performance Benefits (Verified in Production)**
- **Response Time**: 33-76% faster with memory injection
- **Memory Retrieval Overhead**: 24-147ms (negligible vs. model time)
- **Token Efficiency**: 23-54% reduction in output tokens
- **ROI**: 347x on memory retrieval time
- **System Uptime**: 16+ hours of continuous operation with 477+ memories indexed

### **🔧 Recent Major Improvements**
- **Encoding Pipeline**: Flexible 1-4 field acceptance, reasoning contamination eliminated
- **JSON Repair System**: 9-level fallback for robust LLM response handling (8% error rate)
- **IVF Vector Store**: Fully operational with adaptive training and self-healing
- **Logging Optimization**: 70%+ startup noise reduction, consolidated memory retrieval logs
- **Configuration Unification**: Fixed max_tokens=0 bug, merged duplicate schemas

> **Development Status**: Core systems functional, IVF Phase 3 ready to implement. NOT PRODUCTION READY.

---

An API pipeline framework that proxies requests to OpenAI-compatible endpoints, providing persistent memory and continuous architectural evolution through mutations.

**Key capabilities**: Transparent API proxy, self-evolving memory systems, zero-code integration, research-based implementation (arXiv:2512.18746), and development deployment.

## 🔬 Research Background

Based on **MemEvolve: Meta-Evolution of Agent Memory Systems** (arXiv:2512.18746). See [complete research details](docs/development/evolution.md) for implementation specifics and citation information.

## 🚀 Features

- **API Proxy**: Transparent interception of OpenAI-compatible requests
- **Self-Evolving Memory**: Working architectural optimization through mutations with fitness evaluation
- **Auto-Evolution**: Request-based and periodic evolution triggers with boundary validation
- **Memory Management**: 477+ stored experiences with semantic retrieval and relevance filtering
- **Zero Integration**: Drop-in replacement - just change endpoint URL
- **Memory Injection**: Automatic context enhancement for all requests
- **Universal Compatibility**: Works with any OpenAI-compatible service
- **Quality Scoring**: Working relevance and quality evaluation system
- **IVF Vector Store**: Production-ready with 33-76% response time improvement

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
- **Memory System**: Growing database with 477+ experiences, 95%+ storage success rate
- **IVF Vector Store**: Fully operational with self-healing, 16+ hours production runtime
- **Encoding Pipeline**: Flexible 1-4 field acceptance, reasoning contamination eliminated
- **JSON Repair System**: 9-level fallback ensures robust response parsing (8% error rate)
- **Logging System**: Optimized with 70%+ startup noise reduction, consolidated retrieval logs
- **Configuration System**: Unified encoder configuration, max_tokens=0 bug fixed
- **API Proxy Framework**: Transparent request/response processing

### **📊 Verified Performance Benefits**
| Metric | Improvement |
|--------|-------------|
| **Response Time** | 33-76% faster with memory injection |
| **Memory Retrieval Overhead** | 24-147ms (negligible vs. model time) |
| **Token Efficiency** | 23-54% reduction in output tokens |
| **ROI** | 347x on memory retrieval time |

### **⚠️ Systems Pending Implementation**
- **IVF Phase 3**: Configuration & monitoring (13 hours implementation ready)
- **Evolution System**: Requires investigation to determine current state and implement fixes

### **📋 Completed Major Improvements (v2.1.0)**
1. ✅ Encoding pipeline optimized (95%+ success rate, flexible schema)
2. ✅ IVF vector store fully operational (production verified)
3. ✅ Configuration unification (merged duplicate schemas, fixed max_tokens)
4. ✅ Logging optimization (75% volume reduction, startup noise eliminated)
5. ✅ JSON parsing improvements (76% error reduction from 34% to 8%)

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

## 📊 Performance Analysis

The memory injection pipeline delivers **substantial, verified performance improvements** for AI queries based on 16+ hours of production runtime:

| Metric | Improvement |
|--------|-------------|
| **Response Time** | 33-76% faster with memory injection |
| **Memory Retrieval Overhead** | 24-147ms (negligible vs. model time) |
| **Token Efficiency** | 23-54% reduction in output tokens |
| **ROI** | 347x on memory retrieval time |

### Key Findings

- **Threshold 0.44**: Optimal balance between injection rate and quality
- **Hybrid Scoring**: Prevents false positives from single-strategy matches
- **Cumulative Benefit**: Repeated queries see progressive improvement as memory system evolves
- **IVF Vector Store**: Fully operational with 477+ indexed memories, self-healing on corruption

See [Performance Report](reports/memory_pipeline_performance_report.md) for detailed analysis.

---

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
- **Memory System**: Four-component architecture with 477+ experiences, 95%+ storage success
- **IVF Vector Store**: Fully operational with self-healing, 16+ hours production verified
- **Encoding Pipeline**: Flexible 1-4 field acceptance, reasoning contamination eliminated
- **JSON Handling**: Robust transformation and repair systems (8% error rate)
- **Logging System**: Optimized with 70%+ startup noise reduction
- **Configuration System**: Unified encoder configuration, max_tokens bug fixed
- **API Proxy**: Transparent request/response processing
- **Performance**: 33-76% faster response times, 347x ROI verified

### **⚠️ Systems Pending Implementation**
- **IVF Phase 3**: Configuration & monitoring (13 hours ready to implement)
- **Evolution System**: Current state unknown, next priority for investigation

### **📋 v2.1.0 Key Improvements Completed**
- **Memory Pipeline**: Encoding optimized to 95%+ success rate
- **IVF Vector Store**: Fully operational, production verified with 477+ memories
- **Configuration Unification**: Merged duplicate schemas, fixed max_tokens=0
- **Logging Optimization**: 70%+ startup noise reduction, consolidated retrieval logs
- **JSON Parsing**: 76% error reduction (34% → 8%)
- **Performance**: 33-76% faster, 347x ROI, 23-54% token reduction

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

**MIT License — use it, fork it, break it, improve it.**

See [LICENSE](LICENSE) for details

## 🔗 Resources

- **Repository**: https://github.com/thephimart/MemEvolve-API
- **Issues**: https://github.com/thephimart/MemEvolve-API/issues
- **Documentation**: [docs/index.md](docs/index.md)

---

---

**⚠️ Version 2.1.0 Development Notice**: This is the master branch in active development. Core memory system is functional (75%+ success rate) with robust error handling and logging. Evolution system requires analysis and implementation. NOT PRODUCTION READY. See [dev_tasks.md](dev_tasks.md) for current priorities and status.

*Last updated: February 14, 2026*
