# MemEvolve Developer Onboarding Guide

Welcome to MemEvolve! This guide will help you get started quickly with developing memory-augmented applications. Whether you're building AI agents, recommendation systems, or knowledge management tools, MemEvolve provides the foundation for persistent, evolving memory capabilities.

## 🚀 Quick Start (5 minutes)

### Prerequisites

- **Python**: 3.8 or higher
- **LLM API**: Access to any OpenAI-compatible API (vLLM, Ollama, OpenAI, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/thephimart/memevolve.git
cd memevole

# Install core dependencies
pip install -r requirements.txt

# Optional: Install extended dependencies
pip install neo4j networkx faiss-cpu  # For different storage backends
pip install datasets                  # For benchmark evaluation
```

### Your First Memory System

```python
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory_system import MemorySystem, MemorySystemConfig

# Configure with your LLM
config = MemorySystemConfig(
    llm_base_url="http://localhost:8080/v1",  # Your LLM API endpoint
    llm_api_key="your-api-key"
)

# Create memory system
memory = MemorySystem(config)

# Add an experience
memory.add_experience({
    "action": "debug code",
    "result": "found the bug in 5 minutes",
    "context": "Python web application"
})

# Query memories
results = memory.query_memory("debugging techniques")
print(f"Found {len(results)} relevant memories")
```

## 🏗️ Architecture Overview

MemEvolve follows a modular architecture with four core components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    EXPERIENCE   │ -> │   ENCODE        │ -> │   STORE         │ -> │   RETRIEVE      │
│    (Raw Data)   │    │ (Transform)     │    │ (Persist)       │    │ (Query)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                                       │
                                                                       v
                                                            ┌─────────────────┐
                                                            │   MANAGE        │
                                                            │ (Lifecycle)     │
                                                            └─────────────────┘
```

### Components

1. **Encode**: Transforms raw experiences into structured memories
2. **Store**: Persists memories using various backends (JSON, Vector, Graph)
3. **Retrieve**: Finds relevant memories using different strategies (keyword, semantic, hybrid, LLM-guided)
4. **Manage**: Handles memory lifecycle (pruning, consolidation, decay)

## ⚙️ Configuration Guide

### Environment Variables

MemEvolve can be configured entirely through environment variables:

```bash
# Required: LLM API access
export MEMEVOLVE_LLM_BASE_URL="http://localhost:8080/v1"
export MEMEVOLVE_LLM_API_KEY="your-api-key"
export MEMEVOLVE_LLM_MODEL="your-model-name"  # Optional

# Optional: System behavior
export MEMEVOLVE_LOG_LEVEL="INFO"                    # DEBUG, INFO, WARNING, ERROR
export MEMEVOLVE_DEFAULT_TOP_K="5"                   # Default retrieval count
export MEMEVOLVE_AUTO_MANAGE="true"                  # Enable automatic management
```

### Programmatic Configuration

For more control, use the `MemorySystemConfig` class:

```python
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory_system import MemorySystem, MemorySystemConfig
from components.store import GraphStorageBackend
from components.retrieve import LLMGuidedRetrievalStrategy

# Advanced configuration
config = MemorySystemConfig(
    # LLM settings
    llm_base_url="http://localhost:8080/v1",
    llm_api_key="your-key",
    llm_model="llama-2-7b",

    # Retrieval settings
    default_retrieval_top_k=10,

    # Management settings
    enable_auto_management=True,
    auto_prune_threshold=1000,

    # Custom components
    storage_backend=GraphStorageBackend(create_relationships=True),
    # retrieval_strategy=LLMGuidedRetrievalStrategy(llm_callable),

    # Callbacks
    on_encode_complete=lambda mem_id, mem: print(f"Encoded: {mem_id}"),
    on_retrieve_complete=lambda results: print(f"Retrieved: {len(results)} results")
)

memory = MemorySystem(config)
```

## 🗂️ Storage Backends

MemEvolve supports multiple storage backends for different use cases:

### JSON Storage (Default)
- **Best for**: Development, simple applications
- **Features**: File-based, human-readable, easy debugging
- **Limitations**: No semantic search, basic querying

```python
from components.store import JSONFileStore
config.storage_backend = JSONFileStore()
```

### FAISS Vector Storage
- **Best for**: Semantic search, large datasets
- **Features**: High-performance similarity search, scalable
- **Requirements**: `pip install faiss-cpu`

```python
from components.store import VectorStore
config.storage_backend = VectorStore(dimension=768)  # Embedding dimension
```

### Neo4j Graph Storage
- **Best for**: Relationship-aware queries, complex knowledge graphs
- **Features**: Graph traversal, relationship queries, automatic linking
- **Requirements**: Neo4j database, `pip install neo4j`

```python
from components.store import GraphStorageBackend
config.storage_backend = GraphStorageBackend(
    uri="bolt://localhost:7687",
    create_relationships=True  # Auto-link similar memories
)
```

## 🔍 Retrieval Strategies

Choose the right retrieval strategy for your use case:

### Keyword Retrieval
- **Best for**: Exact matches, structured data
- **Method**: TF-IDF based text matching

### Semantic Retrieval
- **Best for**: Meaning-based search, general queries
- **Method**: Vector similarity with embeddings

### Hybrid Retrieval
- **Best for**: Balanced approach, most applications
- **Method**: Combines keyword and semantic scoring

### LLM-Guided Retrieval
- **Best for**: Complex reasoning, context-aware search
- **Method**: Uses LLM to enhance and rerank results

```python
from components.retrieve import (
    KeywordRetrievalStrategy,
    SemanticRetrievalStrategy,
    HybridRetrievalStrategy,
    LLMGuidedRetrievalStrategy
)

# Different strategies for different needs
config.retrieval_strategy = HybridRetrievalStrategy()  # Most common

# Or create custom LLM-guided retrieval
def llm_caller(prompt):
    # Your LLM API call here
    return "LLM response"

llm_guided = LLMGuidedRetrievalStrategy(
    llm_client_callable=llm_caller,
    base_strategy=SemanticRetrievalStrategy()
)
config.retrieval_strategy = llm_guided
```

## 📊 Development Workflow

### 1. Local Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Run tests
python -m pytest src/tests/ -v

# Run examples
python examples/basic_usage.py
```

### 2. Testing Your Changes

```bash
# Run specific test module
python -m pytest src/tests/test_memory_system.py -v

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Test examples
python examples/graph_store_example.py
```

### 3. Adding New Features

MemEvolve is designed to be extensible. Here's how to add new components:

#### Custom Storage Backend

```python
from components.store.base import StorageBackend

class CustomStorageBackend(StorageBackend):
    def store(self, unit):
        # Your storage logic
        return unit_id

    def retrieve(self, unit_id):
        # Your retrieval logic
        return unit

    # Implement other required methods...
```

#### Custom Retrieval Strategy

```python
from components.retrieve.base import RetrievalStrategy

class CustomRetrievalStrategy(RetrievalStrategy):
    def retrieve(self, query, storage_backend, top_k=5, filters=None):
        # Your retrieval logic
        return results

    # Implement other required methods...
```

#### Custom Encoder

```python
from components.encode.encoder import ExperienceEncoder

class CustomEncoder(ExperienceEncoder):
    def encode_experience(self, experience):
        # Your encoding logic
        return encoded_unit
```

### 4. Benchmarking Your Changes

Use the built-in evaluation framework to test your changes:

```bash
# Run baseline experiments
python -m src.evaluation.experiment_runner --experiment-type baseline --max-samples 5

# Test specific components
python -m src.evaluation.experiment_runner --experiment-type single \
    --architecture AgentKB --benchmark GAIA --max-samples 10
```

## 🐛 Troubleshooting

### Common Issues

#### "LLM client not initialized"
- **Cause**: Missing or incorrect LLM API configuration
- **Solution**: Check `MEMEVOLVE_LLM_BASE_URL` and `MEMEVOLVE_LLM_API_KEY`

#### "ImportError: No module named 'faiss'"
- **Cause**: Missing optional dependency
- **Solution**: `pip install faiss-cpu` or use different storage backend

#### Slow performance
- **Cause**: Large dataset with inefficient retrieval
- **Solution**: Use VectorStore or GraphStorageBackend, adjust top_k

#### Memory growing indefinitely
- **Cause**: No automatic management enabled
- **Solution**: Set `enable_auto_management=True` or call `memory.manage_memory()` manually

### Getting Help

1. **Documentation**: Check the `docs/` directory for detailed guides
2. **Examples**: Run examples in the `examples/` directory
3. **Tests**: Look at existing tests for usage patterns
4. **Issues**: Open GitHub issues for bugs or feature requests

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY examples/ ./examples/

EXPOSE 8000
CMD ["python", "examples/basic_usage.py"]
```

### Environment-Specific Configuration

```python
import os

def create_production_config():
    env = os.getenv("ENVIRONMENT", "development")

    if env == "production":
        return MemorySystemConfig(
            llm_base_url=os.getenv("PROD_LLM_URL"),
            llm_api_key=os.getenv("PROD_LLM_KEY"),
            storage_backend=GraphStorageBackend(
                uri=os.getenv("NEO4J_URI"),
                user=os.getenv("NEO4J_USER"),
                password=os.getenv("NEO4J_PASSWORD")
            ),
            enable_auto_management=True,
            log_level="WARNING"
        )
    else:
        # Development config
        return MemorySystemConfig()
```

## 📚 Learning Resources

1. **[Quick Start Tutorial](tutorials/quick_start.md)**: Basic usage and concepts
2. **[Advanced Patterns](tutorials/advanced_patterns.md)**: Complex use cases and optimizations
3. **[API Reference](api/)**: Detailed component documentation
4. **[Examples](examples/)**: Working code samples
5. **[Benchmarks](evaluation/)**: Performance evaluation and comparison

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Development workflow

---

**Happy coding with MemEvolve!** 🧠✨

Need help? Check the [troubleshooting guide](troubleshooting.md) or open an issue on GitHub.