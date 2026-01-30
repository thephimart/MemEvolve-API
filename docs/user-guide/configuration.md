# MemEvolve Configuration Guide

Configuration options, best practices, and optimization strategies for MemEvolve deployments. **78 configurable environment variables** provide complete system control including auto-evolution and business analytics.

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
MEMEVOLVE_API_PORT=11436

# LLM API (required for chat completions and memory encoding)
MEMEVOLVE_UPSTREAM_BASE_URL=https://your-llm-service.com/v1
MEMEVOLVE_UPSTREAM_API_KEY=your-production-key

# Embedding API (optional - defaults to same as upstream)
# Only needed if embeddings are served from a different endpoint
# If not set, MemEvolve will fall back to using MEMEVOLVE_UPSTREAM_BASE_URL for embeddings
MEMEVOLVE_EMBEDDING_BASE_URL=
MEMEVOLVE_EMBEDDING_API_KEY=
MEMEVOLVE_EMBEDDING_MODEL=

# Memory System Settings
MEMEVOLVE_API_MEMORY_INTEGRATION=true
MEMEVOLVE_STORAGE_BACKEND_TYPE=json
MEMEVOLVE_DEFAULT_TOP_K=3
MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD=1024
```

**Smart Defaults:** The memory system automatically inherits LLM settings from your upstream API configuration. Only override `MEMEVOLVE_MEMORY_*` variables if you want different models for memory encoding vs chat responses.

### MemorySystemConfig (Library Usage)

For programmatic/library usage:

```python
import sys
from pathlib import Path

# Import from installed package (no path manipulation needed)
from memevolve.memory_system import MemorySystemConfig

# Most configuration is now handled via environment variables (.env file)
# Programmatic configuration is reserved for advanced use cases

# Example for advanced programmatic setup:
config = MemorySystemConfig(
    # Note: Most of these are now configured via environment variables
    # See .env.example for complete configuration options
)
```

### Environment Variables (.env file)

Create a `.env` file in your project root with the following configuration:

```bash
# API Server Configuration
MEMEVOLVE_API_ENABLE=true
MEMEVOLVE_API_HOST=0.0.0.0
MEMEVOLVE_API_PORT=11436
MEMEVOLVE_UPSTREAM_BASE_URL=https://your-llm-service.com/v1
MEMEVOLVE_UPSTREAM_API_KEY=your-production-key
MEMEVOLVE_API_MEMORY_INTEGRATION=true

# LLM Services Configuration
MEMEVOLVE_MEMORY_BASE_URL=
MEMEVOLVE_MEMORY_API_KEY=
MEMEVOLVE_MEMORY_MODEL=
MEMEVOLVE_EMBEDDING_BASE_URL=
MEMEVOLVE_EMBEDDING_API_KEY=
MEMEVOLVE_EMBEDDING_MODEL=

# Data Directories
MEMEVOLVE_DATA_DIR=./data
MEMEVOLVE_CACHE_DIR=./cache
MEMEVOLVE_LOGS_DIR=./logs

# Storage & Retrieval
MEMEVOLVE_STORAGE_BACKEND_TYPE=json
MEMEVOLVE_STORAGE_INDEX_TYPE=flat
MEMEVOLVE_STORAGE_PATH=./data/memory.json
MEMEVOLVE_RETRIEVAL_STRATEGY_TYPE=hybrid
MEMEVOLVE_RETRIEVAL_TOP_K=3
MEMEVOLVE_RETRIEVAL_SEMANTIC_WEIGHT=0.7
MEMEVOLVE_RETRIEVAL_KEYWORD_WEIGHT=0.3

# Neo4j Configuration (when using graph storage)
MEMEVOLVE_NEO4J_URI=bolt://localhost:7687
MEMEVOLVE_NEO4J_USER=neo4j
MEMEVOLVE_NEO4J_PASSWORD=password
MEMEVOLVE_NEO4J_TIMEOUT=30
MEMEVOLVE_NEO4J_MAX_RETRIES=3

# Memory Management
MEMEVOLVE_MANAGEMENT_ENABLE_AUTO_MANAGEMENT=true
MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD=1024
MEMEVOLVE_MANAGEMENT_FORGETTING_STRATEGY=lru

# System Configuration
MEMEVOLVE_LOG_LEVEL=INFO

# Evolution System Configuration (advanced feature)
MEMEVOLVE_ENABLE_EVOLUTION=true
MEMEVOLVE_EVOLUTION_POPULATION_SIZE=10
MEMEVOLVE_EVOLUTION_GENERATIONS=20
MEMEVOLVE_EVOLUTION_MUTATION_RATE=0.1
MEMEVOLVE_EVOLUTION_CROSSOVER_RATE=0.5
MEMEVOLVE_EVOLUTION_SELECTION_METHOD=pareto
MEMEVOLVE_EVOLUTION_TOURNAMENT_SIZE=3
MEMEVOLVE_EVOLUTION_CYCLE_SECONDS=60
```

Copy `.env.example` to `.env` and modify the values for your environment.

### Evolution System Configuration

The evolution system automatically optimizes memory system performance through genetic algorithms. Enable with `MEMEVOLVE_ENABLE_EVOLUTION=true`.

#### Evolution Parameters
- **Population Size** (`MEMEVOLVE_EVOLUTION_POPULATION_SIZE`): Genotypes per generation (default: 10)
  - Higher values: Better optimization, slower evolution
  - Lower values: Faster evolution, potentially suboptimal

- **Generations** (`MEMEVOLVE_EVOLUTION_GENERATIONS`): Evolution cycles (default: 20)
  - More generations: Thorough optimization, longer convergence
  - Fewer generations: Faster results, may not find optimal settings

- **Mutation Rate** (`MEMEVOLVE_EVOLUTION_MUTATION_RATE`): Parameter change probability (default: 0.1)
  - 0.1 = 10% chance of mutations
  - Higher: More exploration, more variability
  - Lower: More stable evolution, slower adaptation

- **Crossover Rate** (`MEMEVOLVE_EVOLUTION_CROSSOVER_RATE`): Genetic mixing rate (default: 0.5)
  - 0.5 = 50% of offspring from crossover
  - Higher: Faster convergence through genetic mixing

- **Selection Method** (`MEMEVOLVE_EVOLUTION_SELECTION_METHOD`): Optimization strategy (default: pareto)
  - Pareto: Multi-objective optimization balancing competing goals

- **Tournament Size** (`MEMEVOLVE_EVOLUTION_TOURNAMENT_SIZE`): Selection pressure (default: 3)
  - Larger: Stronger selection, faster convergence

- **Evolution Cycle Rate** (`MEMEVOLVE_EVOLUTION_CYCLE_SECONDS`): Seconds between evolution cycles (default: 60)
  - Controls how frequently evolution optimization runs

#### Auto-Evolution Triggers
- **Auto-Evolution Enabled** (`MEMEVOLVE_AUTO_EVOLUTION_ENABLED`): Enable intelligent auto-evolution (default: true)
  - Automatic evolution based on performance triggers
  - More responsive than manual evolution triggers

- **Request Count Trigger** (`MEMEVOLVE_AUTO_EVOLUTION_REQUESTS`): Start evolution after N requests (default: 500)
  - Evolution begins automatically after processing this many requests
  - Ensures sufficient data for meaningful optimization

- **Performance Degradation Trigger** (`MEMEVOLVE_AUTO_EVOLUTION_DEGRADATION`): Start evolution if performance degrades by X% (default: 0.2)
  - Detects when memory system performance is declining
  - Automatic recovery through evolution when quality drops

- **Fitness Plateau Trigger** (`MEMEVOLVE_AUTO_EVOLUTION_PLATEAU`): Start evolution if fitness is stable for N generations (default: 5)
  - Detects when optimization has reached local maximum
  - Continues searching for better configurations

- **Time-Based Trigger** (`MEMEVOLVE_AUTO_EVOLUTION_HOURS`): Periodic evolution every N hours (default: 24)
  - Ensures regular optimization regardless of other triggers
  - Useful for long-running systems with stable performance
  - Higher values: Less frequent evolution, more stable performance
  - Lower values: More frequent evolution, faster adaptation

#### Evolution Process
1. **Initial Population**: Random genotypes created
2. **Fitness Evaluation**: Each genotype tested on real API traffic
3. **Selection**: Best performers selected for reproduction
4. **Genetic Operations**: Crossover and mutation create new genotypes
5. **Iteration**: Process repeats for specified generations

#### What Gets Optimized
- Retrieval strategy weights (semantic vs keyword)
- Memory management thresholds
- Encoding parameters
- Embedding dimensions (within model capabilities)
- Storage-specific parameters (when applicable)

#### Performance Expectations
- **Early evolution** (first 1000 requests): Frequent parameter changes, performance fluctuations
- **Mid evolution** (1000-10000 requests): Settling on effective configurations
- **Late evolution** (10000+ requests): Fine-tuning for optimal performance
- **Convergence**: Typically 20-50 generations, depending on population size

## 📝 Logging Configuration

MemEvolve supports configurable logging for different system components. By default, all logging is disabled to reduce noise, but can be enabled per component for debugging and monitoring.

### Individual System Logging

Each major component can have its logging independently enabled:

```bash
# API Server Logging (FastAPI proxy operations)
MEMEVOLVE_LOG_API_SERVER_ENABLE=false
MEMEVOLVE_LOG_API_SERVER_DIR=./logs

# Middleware Logging (Memory integration processing)
MEMEVOLVE_LOG_MIDDLEWARE_ENABLE=false
MEMEVOLVE_LOG_MIDDLEWARE_DIR=./logs

# Memory System Logging (Core memory operations)
MEMEVOLVE_LOG_MEMORY_ENABLE=false
MEMEVOLVE_LOG_MEMORY_DIR=./logs

# Experiment Logging (Benchmark and evaluation runs)
MEMEVOLVE_LOG_EXPERIMENT_ENABLE=false
MEMEVOLVE_LOG_EXPERIMENT_DIR=./logs
```

### When to Enable Logging

- **Development**: Enable component logging to debug issues
- **Production Monitoring**: Enable API server logging for request tracking
- **Memory Debugging**: Enable memory logging to track memory operations
- **Benchmarking**: Enable experiment logging for performance analysis

### Log File Locations

Logs are written to separate files in the configured directory:
- `api-server.log` - API server operations and requests
- `middleware.log` - Memory integration middleware activities
- `memory.log` - Core memory system operations
- `experiment.log` - Experiment and evaluation runs

### Performance Considerations

- Logging adds I/O overhead, especially for high-throughput applications
- File logging is buffered but can impact performance under heavy load
- Consider log rotation for long-running deployments
- Use structured logging for better analysis and monitoring

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
MEMEVOLVE_STORAGE_INDEX_TYPE=ivf  # FAISS index type: flat/ivf/hnsw

# Embedding dimension (optional - auto-detected by default)
# MEMEVOLVE_EMBEDDING_DIMENSION=768  # Only override if needed
```

**Note:** `MEMEVOLVE_STORAGE_INDEX_TYPE` only affects the **vector backend**. JSON and graph backends ignore this setting.

**IVF Index Training:** IVF indexes require training before use. The system automatically trains indexes on first use, but this may cause a brief delay on initial memory storage operations.

**Best for**: Large datasets, semantic search, production use
**Requirements**: `pip install faiss-cpu`
**Performance**: Fast similarity search (~100x faster than JSON), higher memory usage
**Storage**: Binary FAISS index files

**Note**: Embedding dimensions are automatically detected from your embedding model's metadata. Set `MEMEVOLVE_EMBEDDING_DIMENSION` only if auto-detection fails or you need to force a specific dimension.

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
| **Query Speed** | 🔴 Slow | 🟢 Fast* | 🟡 Medium |
| **Semantic Search** | 🔴 Basic | 🟢 Excellent | 🟡 Good |
| **Relationship Queries** | 🔴 None | 🔴 None | 🟢 Advanced |
| **Debugging** | 🟢 Easy | 🟡 Medium | 🔴 Hard |
| **Development Ready** | 🟡 Small scale | 🟢 Yes | 🟢 Yes |

\* *Vector performance depends on index type: flat (fastest), ivf (balanced, requires training), hnsw (speed, higher memory)*

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
- **ivf**: Balanced speed-accuracy FAISS index for general use
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
from components.retrieve import APIGuidedRetrievalStrategy, SemanticRetrievalStrategy

def llm_caller(prompt: str) -> str:
    """Your LLM API call function"""
    # Implement your LLM call here
    return response

config.retrieval_strategy = APIGuidedRetrievalStrategy(
    api_client_callable=llm_caller,
    base_strategy=SemanticRetrievalStrategy(),
    reasoning_temperature=0.3,
    max_reasoning_tokens=256
)
```

**Note**: LLM-guided retrieval works with models that support structured reasoning output, including thinking models like GLM-4.6V-Flash and other reasoning-enhanced LLMs.

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
config.api_max_retries = 3
config.memory.timeout = 30
```

#### Model-Specific Settings
```python
# For different model types
if "gpt" in config.memory.model:
    # Model-specific parameters would be set in genotype evolution
    pass
elif "llama" in config.memory.model:
    # Model-specific parameters would be set in genotype evolution
    pass
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
    # Create embedding function with auto-detected dimensions
    from utils.embeddings import create_embedding_function
    embedding_func = create_embedding_function(
        base_url="https://your-embedding-service.com/v1",
        api_key="your-embedding-key"
    )
    
    # Dimensions auto-detected from embedding model metadata
    # or set via MEMEVOLVE_EMBEDDING_DIMENSION environment variable
    
    return MemorySystemConfig(
        log_level="WARNING",
        enable_auto_management=True,
        auto_prune_threshold=50000,
        storage_backend=VectorStore(
            index_file="./data/vectors",
            embedding_function=embedding_func,
            embedding_dim=embedding_func.get_embedding_dim()
        ),
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
        memory_base_url="http://mock-llm:11433",  # Mock LLM for testing
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
from components.retrieve import APIGuidedRetrievalStrategy

def validate_config(config: MemorySystemConfig) -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []

    # Required settings
    if not config.memory.base_url:
        issues.append("memory.base_url is required")
    if not config.memory.api_key:
        issues.append("memory.api_key is required")

    # Storage compatibility
    if hasattr(config.storage_backend, 'dimension'):
        # Vector storage requires embeddings
        if not config.embedding_function:
            issues.append("Vector storage requires embedding_function")

    # Retrieval compatibility
    if isinstance(config.retrieval_strategy, APIGuidedRetrievalStrategy):
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
        # VectorStore created with auto-detected embedding dimensions
        "storage_backend": lambda: VectorStore(
            index_file="./data/production_vectors",
            embedding_function=create_embedding_function(),
            embedding_dim=create_embedding_function().get_embedding_dim()
        ),
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
config = create_config_from_profile("production", memory_base_url="custom-url")
```

## 🚨 Common Configuration Pitfalls

### 1. LLM Connection Issues
```python
# ❌ Wrong: Missing protocol
config.memory.base_url = "localhost:11433"

# ✅ Correct: Include protocol
config.memory.base_url = "http://localhost:11433"
```

### 2. Storage Backend Mismatches
```python
# ❌ Wrong: Hardcoded dimension that doesn't match your embedding model
embedding_func = create_embedding_function()  # Returns 768-dim embeddings
config.storage_backend = VectorStore(
    index_file="./data/vectors",
    embedding_function=embedding_func,
    embedding_dim=512  # Wrong! Mismatch with embedding function
)

# ✅ Correct: Use auto-detected dimension from embedding function
embedding_func = create_embedding_function()
config.storage_backend = VectorStore(
    index_file="./data/vectors",
    embedding_function=embedding_func,
    embedding_dim=embedding_func.get_embedding_dim()  # Auto-detects 768
)

# ✅ Also correct: Use MEMEVOLVE_EMBEDDING_DIMENSION to override
# Set in .env: MEMEVOLVE_EMBEDDING_DIMENSION=1024
embedding_func = create_embedding_function()
config.storage_backend = VectorStore(
    index_file="./data/vectors",
    embedding_function=embedding_func,
    embedding_dim=embedding_func.get_embedding_dim()  # Uses 1024 from env
)
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