# MemEvolve Getting Started Guide

Welcome to MemEvolve v2.1.0! **This is an API pipeline framework that proxies API requests to OpenAI compatible endpoints providing memory, memory management, and evolves the memory implementation through mutations to enhance the memory system over time.**

## 🚨 v2.1.0 IN DEVELOPMENT - NOT PRODUCTION READY

**IMPORTANT**: You are using MemEvolve-API v2.1.0 on the master branch in active development. Core memory system is functional (95%+ success rate). IVF vector store corruption is fixed. Evolution system requires analysis and implementation.

### **✅ Functional Core Systems**
- **OpenAI-Compatible API**: Chat completions endpoint operational for development
- **Memory Retrieval & Injection**: Context enhancement with growing database
- **Experience Encoding**: Memory creation with flexible 1-4 field acceptance (95%+ success)
- **IVF Vector Store**: Corruption eliminated with self-healing capabilities
- **Schema & JSON Handling**: Robust transformation and repair systems (8% error rate)
- **Centralized Configuration**: Unified encoder configuration, logging optimization
- **Optimized Logging**: 70%+ startup noise reduction, consolidated retrieval logs

### **⚠️ Systems Pending Implementation**
- **IVF Phase 3**: Configuration & monitoring (13 hours implementation ready)
- **Evolution System**: Current state unknown, next priority for investigation

### **v2.1.0 Key Improvements**
- **Memory Pipeline**: Encoding optimized to 95%+ success rate with flexible schema
- **IVF Vector Store**: Fixed 45% corruption with adaptive training
- **Configuration Unification**: Merged duplicate schemas, fixed max_tokens=0 bug
- **Logging Optimization**: 70%+ startup noise reduction, eliminated duplicate retrieval logs
- **JSON Parsing**: 76% error reduction (34% → 8%)

### **Development Status**
- ✅ **Use for Development**: Core memory system functional for testing
- ⚠️ **NOT PRODUCTION READY**: Evolution system pending analysis
- 📋 **Track Progress**: See [dev_tasks.md](../../dev_tasks.md) for current priorities

---

This guide will help you get started with MemEvolve v2.1.0 - a self-evolving API proxy that adds persistent memory capabilities to any OpenAI-compatible LLM service without requiring code changes.

## 🚀 Quick Start (5 minutes)

### Prerequisites
- **Python**: 3.10+ (developed on 3.12+, tested on 3.12+ and 3.10+; should be compatible with 3.7+ untested)
- **LLM API**: Access to any OpenAI-compatible API (vLLM, Ollama, OpenAI, etc.) with embedding support
- **API Endpoint**: Your LLM service endpoint (embeddings can use the same endpoint)

### Installation

```bash
# Clone repository
git clone https://github.com/thephimart/MemEvolve-API.git
cd MemEvolve-API

# Option 1: One-click interactive setup (⚠️ UNTESTED - use at your own risk)
./scripts/setup.sh

# Option 2: Manual installation (RECOMMENDED)
pip install -e .
```

## 🎯 Using MemEvolve API Pipeline

If you have existing applications using OpenAI-compatible APIs, you can add memory with just configuration changes:

#### Quick API Setup

```bash
# 1. Configure environment (add to .env file)
MEMEVOLVE_UPSTREAM_BASE_URL=http://localhost:11434/v1
MEMEVOLVE_UPSTREAM_API_KEY=your-llm-key
# MEMEVOLVE_EMBEDDING_BASE_URL=http://localhost:11435/v1  # Optional: separate embedding service

# 2. Start MemEvolve proxy
python scripts/start_api.py

# For development with auto-reload (shows file change notifications)
# python scripts/start_api.py --reload
```

**Note:** MemEvolve uses your LLM endpoint for both chat completions and embeddings by default. Only configure separate embedding endpoints if required.

**Port Convention:** Documentation examples use standard ports (11434 for upstream, 11433 for memory, 11435 for embeddings, 11436 for MemEvolve API).

#### Example: Existing OpenAI App

```python
import openai

# Your existing code (no changes needed!)
client = openai.OpenAI(
    base_url="http://localhost:11436/v1",  # Changed to MemEvolve proxy
    api_key="dummy"  # API key not used by proxy
)

response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "How do I optimize database queries?"}]
)

# MemEvolve automatically:
# - Retrieves relevant database optimization memories
# - Injects them into your prompt for better responses
# - Evaluates response quality with fair, parity-based scoring
# - Learns from this interaction for future queries
```

#### Test Your Setup
**Note:** The main API endpoint (chat completions) is fully functional. Management endpoints provide basic functionality with advanced features in development.

```bash
# Check server health (management endpoint - basic functionality)
curl http://localhost:11436/health

# Access real-time dashboard (basic functionality)
open http://localhost:11436/dashboard

# View memory statistics (management endpoint - basic functionality)
curl http://localhost:11436/memory/stats

# Search memory (management endpoint - basic functionality)
curl -X POST http://localhost:11436/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "database optimization", "top_k": 3}'

# Check quality scoring metrics (basic functionality)
curl http://localhost:11436/quality/metrics
```



## 🔧 Configuration

### Basic Configuration

Create a `.env` file in your project root:

```bash
# =============================================================================
# API ENDPOINTS - Three potential endpoints (can be same or separate)
# =============================================================================

# UPSTREAM API: Primary LLM for chat completions (required)
# This is the main LLM service that handles user conversations
# Model name will be auto-resolved on startup if MEMEVOLVE_UPSTREAM_AUTO_RESOLVE_MODELS=true
MEMEVOLVE_UPSTREAM_BASE_URL=http://localhost:11434/v1
MEMEVOLVE_UPSTREAM_API_KEY=your-llm-api-key

# Memory API: Dedicated memory encoding service (optional, defaults to upstream)
# Model name will be auto-resolved on startup if MEMEVOLVE_MEMORY_AUTO_RESOLVE_MODELS=true
# Used for processing and encoding memories - can be same as upstream for simplicity
MEMEVOLVE_MEMORY_BASE_URL=http://localhost:11433/v1
MEMEVOLVE_MEMORY_API_KEY=your-llm-api-key

# EMBEDDING API: Service for vector embeddings (optional, defaults to upstream)
# Creates vector representations of memories for semantic search
# Model name will be auto-resolved on startup if MEMEVOLVE_EMBEDDING_AUTO_RESOLVE_MODELS=true
MEMEVOLVE_EMBEDDING_BASE_URL=http://localhost:11435/v1
MEMEVOLVE_EMBEDDING_API_KEY=your-llm-api-key

# =============================================================================
# MEMORY SYSTEM CONFIGURATION
# =============================================================================

# Storage settings
MEMEVOLVE_STORAGE_PATH=./data/memory.json
MEMEVOLVE_DEFAULT_TOP_K=5
MEMEVOLVE_MANAGEMENT_ENABLE_AUTO_MANAGEMENT=true
```

### Advanced Configuration

```bash
# Memory architecture
MEMEVOLVE_RETRIEVAL_STRATEGY_TYPE=hybrid  # semantic, keyword, or hybrid
MEMEVOLVE_RETRIEVAL_SEMANTIC_WEIGHT=0.7   # Balance between semantic and keyword
MEMEVOLVE_RETRIEVAL_KEYWORD_WEIGHT=0.3

# Auto-management
MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD=1000
MEMEVOLVE_MANAGEMENT_DEDUPLICATE_THRESHOLD=0.9
MEMEVOLVE_MANAGEMENT_MAX_MEMORY_AGE_DAYS=365

# API server (for proxy mode)
MEMEVOLVE_API_HOST=127.0.0.1
MEMEVOLVE_API_PORT=11436
MEMEVOLVE_API_MEMORY_INTEGRATION=true
```

## 🧪 Testing Your Setup

### Run the Test Suite

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests (includes quality scoring tests)
pytest tests/ -v

# Run specific component tests
pytest tests/test_memory_system.py -v
pytest tests/test_semantic_strategy.py -v
pytest tests/test_quality_scorer.py -v
pytest tests/test_memory_scoring.py -v
pytest tests/test_full_pipeline_integration.py -v
pytest tests/test_quality_scorer.py -v
```

### Manual Testing

```python
#!/usr/bin/env python3
"""MemEvolve Diagnostic Script"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from memevolve.memory_system import MemorySystem
from memevolve.utils.config import MemEvolveConfig

def test_memory_system():
    """Test basic memory system functionality."""
    print("🧠 Testing MemEvolve Memory System")

    # Configure (using environment variables)
    config = MemorySystemConfig()

    # Create memory system
    memory = MemorySystem(config)

    # Test basic operations
    test_unit = {
        "content": "Python list comprehensions are efficient for filtering data.",
        "type": "lesson",
        "tags": ["python", "performance"]
    }

    # Store
    unit_id = memory.store(test_unit)
    print(f"✅ Stored unit with ID: {unit_id}")

    # Retrieve
    retrieved = memory.retrieve(unit_id)
    print(f"✅ Retrieved unit: {retrieved['content'][:50]}...")

    # Search
    results = memory.retrieve("python filtering", top_k=3)
    print(f"✅ Found {len(results)} relevant memories")

    # Health check
    health = memory.get_health_metrics()
    print(f"✅ Memory system healthy: {health.total_units} units stored")

    print("🎉 MemEvolve is working correctly!")

if __name__ == "__main__":
    test_memory_system()
```

## 📚 Next Steps

- **API Reference**: Learn about all available endpoints and parameters
- **Quality Scoring**: Understand and configure response quality evaluation
- **Advanced Patterns**: Discover complex memory architectures and optimization techniques
- **Deployment**: Set up production deployments with monitoring
- **Troubleshooting**: Common issues and their solutions

## 🆘 Getting Help

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Check the full guides in `/docs`
- **Quality Scoring Guide**: [Learn about response quality evaluation](quality-scoring.md)
- **Community**: Join discussions and share your use cases

---

**⚠️ Version 2.1.0 Development Notice**: This is the master branch in active development. Core memory system is functional (75%+ success rate) with robust error handling. Evolution system requires analysis and implementation. NOT PRODUCTION READY. See [dev_tasks.md](../../dev_tasks.md) for current priorities.

Happy memory-augmented development with v2.1.0! 🎯

*Last updated: February 10, 2026*