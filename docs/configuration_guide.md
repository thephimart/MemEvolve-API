# MemEvolve Configuration Guide

This guide covers configuration options, best practices, and optimization strategies for MemEvolve deployments.

## Configuration Overview

MemEvolve can be configured through three methods (in order of precedence):

1. **Programmatic Configuration**: Direct `MemorySystemConfig` object
2. **Environment Variables**: System-wide settings
3. **Default Values**: Sensible fallbacks

## 🔧 Core Configuration

### API Wrapper Configuration (Easiest)

For the API wrapper approach, add these settings to your `.env` file:

```bash
# API Server Configuration
MEMEVOLVE_API_ENABLE=true
MEMEVOLVE_API_HOST=0.0.0.0
MEMEVOLVE_API_PORT=8001

# LLM API (required for chat completions and memory encoding)
MEMEVOLVE_UPSTREAM_BASE_URL=https://your-llm-service.com/v1
MEMEVOLVE_UPSTREAM_API_KEY=your-production-key

# Embedding API (optional - defaults to same as upstream)
# Only needed if embeddings are served from a different endpoint
# If not set, MemEvolve will fall back to using MEMEVOLVE_UPSTREAM_BASE_URL for embeddings
# MEMEVOLVE_EMBEDDING_BASE_URL=https://your-embedding-service.com/v1
# MEMEVOLVE_EMBEDDING_API_KEY=your-embedding-key

# Memory System Settings
MEMEVOLVE_API_MEMORY_INTEGRATION=true
MEMEVOLVE_STORAGE_PATH=/app/data/memory.json
MEMEVOLVE_RETRIEVAL_TOP_K=5
MEMEVOLVE_MANAGEMENT_ENABLE_AUTO_MANAGEMENT=true
MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD=1000
```

**Smart Defaults:** The memory system automatically inherits LLM settings from your upstream API configuration. Only override `MEMEVOLVE_LLM_*` variables if you want different models for memory encoding vs chat responses.

### MemorySystemConfig (Library Usage)

For programmatic/library usage:

```python
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory_system import MemorySystemConfig

config = MemorySystemConfig(
    # LLM Configuration
    llm_base_url="http://localhost:8080/v1",
    llm_api_key="your-api-key",
    llm_model="llama-2-7b-chat",  # Optional, auto-detected if not set

    # Retrieval Settings
    default_retrieval_top_k=5,
    retrieval_strategy=None,  # Uses default if None

    # Storage Settings
    storage_backend=None,  # Uses JSONFileStore if None

    # Management Settings
    enable_auto_management=True,
    auto_prune_threshold=1000,
    management_strategy=None,  # Uses SimpleManagementStrategy if None

    # System Settings
    log_level="INFO",

    # Callbacks
    on_encode_complete=None,
    on_retrieve_complete=None,
    on_manage_complete=None
)
```

### Environment Variables (.env file)

Create a `.env` file in your project root with the following configuration:

```bash
# API Server Configuration
MEMEVOLVE_API_ENABLE=true
MEMEVOLVE_API_HOST=0.0.0.0
MEMEVOLVE_API_PORT=8001
MEMEVOLVE_UPSTREAM_BASE_URL=https://your-llm-service.com/v1
MEMEVOLVE_UPSTREAM_API_KEY=your-production-key
MEMEVOLVE_API_MEMORY_INTEGRATION=true

# Memory System Configuration
MEMEVOLVE_LLM_BASE_URL=https://your-llm-service.com/v1
MEMEVOLVE_LLM_API_KEY=your-production-key
MEMEVOLVE_STORAGE_PATH=/data/memory.db
MEMEVOLVE_RETRIEVAL_TOP_K=10
MEMEVOLVE_MANAGEMENT_ENABLE_AUTO_MANAGEMENT=true
MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD=10000

# System Configuration
MEMEVOLVE_LOG_LEVEL=WARNING
MEMEVOLVE_STORAGE_BACKEND_TYPE=vector
MEMEVOLVE_RETRIEVAL_STRATEGY_TYPE=hybrid
```

Copy `.env.example` to `.env` and modify the values for your environment.

## 🎯 Component-Specific Configuration

### Storage Backend Selection

Choose the right storage backend based on your use case:

## Why JSON is the Default

JSON storage is the default choice because:
- **Zero external dependencies** - No need to install FAISS or set up Neo4j
- **Human-readable** - Easy to inspect and debug memory contents
- **Simple setup** - Just works out of the box
- **Good for development** - Perfect for testing and small-scale use

Switch to Vector or Graph storage when you need better performance or advanced features.

### Environment Variable Configuration

#### JSON Storage (Development/Default)
```bash
# Basic JSON configuration (default settings)
MEMEVOLVE_STORAGE_BACKEND_TYPE=json
MEMEVOLVE_STORAGE_PATH=./data/memory.json
```

**Best for**: Development, small datasets (< 10K memories), debugging
**Performance**: Fast writes, linear search reads, basic keyword matching
**Storage**: Human-readable JSON files

#### Vector Storage (Production/Semantic Search)
```bash
# Vector storage configuration
MEMEVOLVE_STORAGE_BACKEND_TYPE=vector
MEMEVOLVE_STORAGE_PATH=./data/memory_vectors
MEMEVOLVE_STORAGE_VECTOR_DIM=768        # Must match your embedding model
MEMEVOLVE_STORAGE_INDEX_TYPE=IndexIVFFlat  # FAISS index type
```

**Best for**: Large datasets, semantic search, production use
**Requirements**: `pip install faiss-cpu`
**Performance**: Fast similarity search (~100x faster than JSON), higher memory usage
**Storage**: Binary FAISS index files

#### Graph Storage (Relationships/Knowledge Graphs)
```bash
# Graph storage configuration
MEMEVOLVE_STORAGE_BACKEND_TYPE=graph
MEMEVOLVE_STORAGE_PATH=bolt://localhost:7687
```

**Best for**: Complex relationships, graph queries, knowledge networks
**Requirements**: Neo4j database, `pip install neo4j`
**Performance**: Relationship-aware queries, graph traversal

## Storage Backend Trade-offs

| Feature | JSON | Vector | Graph |
|---------|------|--------|-------|
| **Setup Complexity** | 🟢 Simple | 🟡 Medium | 🔴 Complex |
| **External Dependencies** | 🟢 None | 🟡 FAISS | 🔴 Neo4j |
| **Memory Capacity** | 🟡 <10K | 🟢 >100K | 🟢 >100K |
| **Query Speed** | 🔴 Slow | 🟢 Fast | 🟡 Medium |
| **Semantic Search** | 🔴 Basic | 🟢 Excellent | 🟡 Good |
| **Relationship Queries** | 🔴 None | 🔴 None | 🟢 Advanced |
| **Debugging** | 🟢 Easy | 🟡 Medium | 🔴 Hard |
| **Production Ready** | 🟡 Small scale | 🟢 Yes | 🟢 Yes |

### When to Choose Each Backend

#### Use JSON Storage When:
- You're just getting started with MemEvolve
- You have < 1,000 memories
- You need to inspect/debug memory contents easily
- You're developing and don't want external dependencies
- Performance isn't critical yet

#### Use Vector Storage When:
- You have > 1,000 memories
- You need fast semantic search
- You're building a production application
- You want the best performance for similarity queries

#### Use Graph Storage When:
- Your memories have complex relationships
- You need to query connections between memories
- You're building knowledge graphs or recommendation systems
- You want advanced relationship analysis

### Default Settings Rationale

The `.env.example` defaults are chosen for the best out-of-the-box experience:

- **JSON storage**: No dependencies, easy to understand and debug
- **768 dimensions**: Common embedding size (matches most models)
- **IndexFlatIP**: Simple, reliable FAISS index for development
- **Hybrid retrieval**: Balances keyword and semantic search
- **Auto-management**: Keeps memory size reasonable automatically

### Retrieval Strategy Configuration

#### Hybrid Retrieval (Recommended)
```python
from components.retrieve import HybridRetrievalStrategy

config.retrieval_strategy = HybridRetrievalStrategy(
    keyword_weight=0.3,     # Weight for keyword matching
    semantic_weight=0.7,    # Weight for semantic similarity
    combine_method="weighted_sum"  # How to combine scores
)
```

#### LLM-Guided Retrieval
```python
from components.retrieve import LLMGuidedRetrievalStrategy, SemanticRetrievalStrategy

def llm_caller(prompt: str) -> str:
    """Your LLM API call function"""
    # Implement your LLM call here
    return response

config.retrieval_strategy = LLMGuidedRetrievalStrategy(
    llm_client_callable=llm_caller,
    base_strategy=SemanticRetrievalStrategy(),
    reasoning_temperature=0.3,
    max_reasoning_tokens=256
)
```

#### Custom Retrieval Strategy
```python
from components.retrieve.base import RetrievalStrategy

class CustomRetrievalStrategy(RetrievalStrategy):
    def retrieve(self, query, storage_backend, top_k=5, filters=None):
        # Your custom retrieval logic
        results = []
        # ... implementation ...
        return results
```

### Management Strategy Configuration

#### Time-Based Decay
```python
from components.manage import TimeDecayManagementStrategy

config.management_strategy = TimeDecayManagementStrategy(
    half_life_days=30,      # Memories decay over 30 days
    min_importance=0.1      # Minimum importance threshold
)
```

#### Size-Based Pruning
```python
from components.manage import SizeBasedManagementStrategy

config.management_strategy = SizeBasedManagementStrategy(
    max_memories=5000,      # Maximum number of memories
    pruning_strategy="oldest_first"  # Which memories to remove
)
```

## 🚀 Performance Optimization

### Memory System Optimization

#### Batch Processing
```python
# Use batch operations for better performance
memory.add_trajectory_batch(experiences, use_parallel=True)

# Configure batch sizes
config.batch_size = 50
config.num_workers = 4
```

#### Memory Pool Configuration
```python
# For high-throughput applications
config.memory_pool_size = 1000
config.preload_embeddings = True
config.cache_embeddings = True
```

### LLM Optimization

#### Connection Pooling
```python
config.llm_connection_pool_size = 10
config.llm_timeout = 30
config.llm_retry_attempts = 3
```

#### Model-Specific Settings
```python
# For different model types
if "gpt" in config.llm_model:
    config.llm_temperature = 0.7
    config.llm_max_tokens = 1024
elif "llama" in config.llm_model:
    config.llm_temperature = 0.1
    config.llm_max_tokens = 512
```

### Storage Optimization

#### Vector Storage Tuning
```python
# For large datasets
vector_config = {
    "index_type": "IndexIVFFlat",  # Faster for large datasets
    "nlist": 100,                  # Number of clusters
    "nprobe": 10                   # Search quality vs speed tradeoff
}
config.storage_backend = VectorStore(**vector_config)
```

#### Graph Storage Tuning
```python
# For complex knowledge graphs
graph_config = {
    "relationship_types": ["SIMILAR_TO", "RELATED_TO", "CAUSES"],
    "max_relationships_per_node": 50,
    "relationship_similarity_threshold": 0.7
}
config.storage_backend = GraphStorageBackend(**graph_config)
```

## 🔧 Best Practices

### Development Environment

```python
def create_development_config():
    return MemorySystemConfig(
        log_level="DEBUG",
        enable_auto_management=False,  # Manual control for debugging
        storage_backend=JSONFileStore(),  # Easy to inspect
        default_retrieval_top_k=10  # More results for testing
    )
```

### Production Environment

```python
def create_production_config():
    return MemorySystemConfig(
        log_level="WARNING",
        enable_auto_management=True,
        auto_prune_threshold=50000,
        storage_backend=VectorStore(dimension=768),
        retrieval_strategy=HybridRetrievalStrategy(),
        # Add monitoring callbacks
        on_encode_complete=log_memory_operation,
        on_retrieve_complete=log_retrieval_operation
    )
```

### Testing Configuration

```python
def create_test_config():
    return MemorySystemConfig(
        llm_base_url="http://mock-llm:8080",  # Mock LLM for testing
        storage_backend=JSONFileStore(file_path=":memory:"),  # In-memory
        enable_auto_management=False,
        log_level="ERROR"
    )
```

## 📊 Monitoring and Observability

### Callback-Based Monitoring

```python
def memory_operation_logger(operation_type):
    def logger(mem_id=None, memory=None, results=None):
        import logging
        logging.info(f"{operation_type}: {mem_id or len(results or [])} items")
    return logger

config.on_encode_complete = memory_operation_logger("ENCODE")
config.on_retrieve_complete = lambda results: print(f"Retrieved {len(results)} memories")
config.on_manage_complete = memory_operation_logger("MANAGE")
```

### Performance Metrics

```python
import time
from functools import wraps

def performance_monitor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print(f"{func.__name__} took {duration:.3f} seconds")
        return result
    return wrapper

# Apply to key methods
memory.add_experience = performance_monitor(memory.add_experience)
memory.query_memory = performance_monitor(memory.query_memory)
```

### Health Checks

```python
def health_check(memory_system):
    """Comprehensive health check for memory system"""
    health = {
        "status": "healthy",
        "checks": {}
    }

    try:
        # Component availability
        health["checks"]["encoder"] = hasattr(memory_system, 'encoder')
        health["checks"]["storage"] = hasattr(memory_system, 'storage')
        health["checks"]["retrieval"] = hasattr(memory_system, 'retrieval_context')

        # Storage functionality
        count = memory_system.storage.count()
        health["checks"]["storage_functional"] = count >= 0

        # Basic operations
        test_exp = {"action": "test", "result": "ok"}
        mem_id = memory_system.add_experience(test_exp)
        results = memory_system.query_memory("test")
        health["checks"]["operations_functional"] = len(results) > 0

        # Clean up test data
        memory_system.storage.delete(mem_id)

    except Exception as e:
        health["status"] = "unhealthy"
        health["error"] = str(e)

    return health
```

## 🔄 Configuration Management

### Configuration Validation

```python
from typing import List
from components.retrieve import LLMGuidedRetrievalStrategy

def validate_config(config: MemorySystemConfig) -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []

    # Required settings
    if not config.llm_base_url:
        issues.append("llm_base_url is required")
    if not config.llm_api_key:
        issues.append("llm_api_key is required")

    # Storage compatibility
    if hasattr(config.storage_backend, 'dimension'):
        # Vector storage requires embeddings
        if not config.embedding_function:
            issues.append("Vector storage requires embedding_function")

    # Retrieval compatibility
    if isinstance(config.retrieval_strategy, LLMGuidedRetrievalStrategy):
        if not callable(getattr(config, 'llm_client_callable', None)):
            issues.append("LLMGuidedRetrieval requires llm_client_callable")

    return issues

# Usage
issues = validate_config(config)
if issues:
    print("Configuration issues:")
    for issue in issues:
        print(f"  - {issue}")
```

### Configuration Profiles

```python
CONFIG_PROFILES = {
    "development": {
        "log_level": "DEBUG",
        "storage_backend": JSONFileStore(),
        "enable_auto_management": False
    },
    "production": {
        "log_level": "WARNING",
        "storage_backend": VectorStore(dimension=768),
        "enable_auto_management": True,
        "auto_prune_threshold": 10000
    },
    "testing": {
        "log_level": "ERROR",
        "storage_backend": JSONFileStore(file_path=":memory:"),
        "enable_auto_management": False
    }
}

def create_config_from_profile(profile_name: str, **overrides) -> MemorySystemConfig:
    """Create configuration from predefined profiles"""
    if profile_name not in CONFIG_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}")

    profile_config = CONFIG_PROFILES[profile_name].copy()
    profile_config.update(overrides)

    return MemorySystemConfig(**profile_config)

# Usage
config = create_config_from_profile("production", llm_base_url="custom-url")
```

## 🚨 Common Configuration Pitfalls

### 1. LLM Connection Issues
```python
# ❌ Wrong: Missing protocol
config.llm_base_url = "localhost:8080"

# ✅ Correct: Include protocol
config.llm_base_url = "http://localhost:8080"
```

### 2. Storage Backend Mismatches
```python
# ❌ Wrong: Vector storage without proper dimensions
config.storage_backend = VectorStore(dimension=512)  # But using 768-dim embeddings

# ✅ Correct: Match embedding dimensions
config.storage_backend = VectorStore(dimension=768)
```

### 3. Memory Leak Prevention
```python
# ❌ Wrong: No management in production
config.enable_auto_management = False  # Memory will grow indefinitely

# ✅ Correct: Enable management
config.enable_auto_management = True
config.auto_prune_threshold = 5000
```

### 4. Retrieval Strategy Conflicts
```python
# ❌ Wrong: Incompatible strategy/backend combination
config.storage_backend = JSONFileStore()  # No vector search
config.retrieval_strategy = SemanticRetrievalStrategy()  # Requires vectors

# ✅ Correct: Compatible combinations
config.storage_backend = VectorStore()
config.retrieval_strategy = HybridRetrievalStrategy()
```

This configuration guide should help you optimize MemEvolve for your specific use case and deployment environment.