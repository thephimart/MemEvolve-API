# MemEvolve Getting Started Guide

Welcome to MemEvolve! This guide will help you get started with the MemEvolve API wrapper - the easiest way to add persistent memory to any OpenAI-compatible LLM service.

## 🚀 Quick Start (5 minutes)

### Prerequisites

- **Python**: 3.12 or higher
- **LLM API**: Access to any OpenAI-compatible API (vLLM, Ollama, OpenAI, etc.) with embedding support
- **API Endpoint**: Your LLM service endpoint (embeddings can use the same endpoint)

### Installation

```bash
# Clone the repository
git clone https://github.com/thephimart/memevolve.git
cd memevolve

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Using MemEvolve API Wrapper

If you have existing applications using OpenAI-compatible APIs, you can add memory with just configuration changes:

#### Quick API Setup

```bash
# 1. Configure environment (add to .env file)
MEMEVOLVE_UPSTREAM_BASE_URL=http://localhost:8000/v1
MEMEVOLVE_UPSTREAM_API_KEY=your-llm-key
# MEMEVOLVE_EMBEDDING_BASE_URL=http://different-endpoint:8001/v1  # Optional: separate embedding service

# 2. Start MemEvolve proxy
python scripts/start_api.py

# For development with auto-reload (shows file change notifications)
# python scripts/start_api.py --reload
```

**Note:** MemEvolve uses your LLM endpoint for both chat completions and embeddings by default. Only configure separate embedding endpoints if required.

#### Example: Existing OpenAI App

```python
import openai

# Your existing code (no changes needed!)
client = openai.OpenAI(
    base_url="http://localhost:8001/v1",  # Changed to MemEvolve proxy
    api_key="dummy"  # API key not used by proxy
)

response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "How do I optimize database queries?"}]
)

# MemEvolve automatically:
# - Retrieves relevant database optimization memories
# - Injects them into your prompt for better responses
# - Learns from this interaction for future queries
```

#### Test Your Setup

```bash
# Check server health
curl http://localhost:11436/health

# View memory statistics
curl http://localhost:11436/memory/stats

# Search memory
curl -X POST http://localhost:11436/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "database optimization", "top_k": 3}'
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
MEMEVOLVE_UPSTREAM_BASE_URL=http://localhost:8000/v1
MEMEVOLVE_UPSTREAM_API_KEY=your-llm-api-key

# LLM API: Dedicated LLM for memory encoding (optional, defaults to upstream)
# Used for processing and encoding memories - can be same as upstream for simplicity
MEMEVOLVE_LLM_BASE_URL=http://localhost:8000/v1
MEMEVOLVE_LLM_API_KEY=your-llm-api-key

# EMBEDDING API: Service for vector embeddings (optional, defaults to upstream)
# Creates vector representations of memories for semantic search
MEMEVOLVE_EMBEDDING_BASE_URL=http://localhost:8000/v1
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

# Run all tests
pytest src/tests/ -v

# Run specific component tests
pytest src/tests/test_memory_system.py -v
pytest src/tests/test_semantic_strategy.py -v
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

from memory_system import MemorySystem, MemorySystemConfig

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
- **Advanced Patterns**: Discover complex memory architectures and optimization techniques
- **Deployment**: Set up production deployments with Docker and monitoring
- **Troubleshooting**: Common issues and their solutions

## 🆘 Getting Help

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Check the full guides in `/docs`
- **Community**: Join discussions and share your use cases

Happy memory-augmented building! 🚀