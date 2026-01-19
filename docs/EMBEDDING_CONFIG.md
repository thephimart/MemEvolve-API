# Model Configuration Summary

## LLM Model (Chat Completions)

**Endpoint:** `http://192.168.1.61:11434/v1`

**Model:** `ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf`

**Specifications:**
- Context window: 131,072 tokens
- Vocabulary size: 201,088
- Embedding dimension (internal): 2880
- Parameters: 20,914,757,184 (~20B)
- Quantization: mxfp4

## Embedding Model (Vector Search)

**Endpoint:** `http://192.168.1.61:11435/v1`

**Model:** `nomic-ai_nomic-embed-text-v2-moe-GGUF_nomic-embed-text-v2-moe.Q5_K_M.gguf`

**Specifications:**
- Embedding dimension: 768
- Context window: 512 tokens
- Vocabulary size: 250,048
- Parameters: 475,288,320
- Quantization: Q5_K_M

## Changes Made

### 1. LLM Model Auto-Detection
- `ExperienceEncoder` now accepts `model: Optional[str] = None`
- Auto-fetches model info from `/v1/models` endpoint
- Model ID is automatically detected and used for chat completions
- Graceful fallback if endpoint doesn't support `/v1/models`

### 2. Embedding Model Auto-Detection
- `OpenAIEmbeddingProvider` now accepts `model: Optional[str] = None`
- Auto-fetches model info from `/v1/models` endpoint
- Model ID is automatically detected and used for embedding requests
- Graceful fallback if endpoint doesn't support `/v1/models`

### 3. MemorySystemConfig Updates
- Added `llm_model` parameter for optional explicit model specification
- Updated `_initialize_encoder()` to pass model to encoder

### 4. Configuration Examples
```python
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# MemorySystem with auto-detected models
from memory_system import MemorySystem, MemorySystemConfig
from utils import create_embedding_function
from components.retrieve import SemanticRetrievalStrategy
from components.store import VectorStore

embedding_fn = create_embedding_function(
    provider="openai",
    base_url="http://192.168.1.61:11435/v1"
)

vector_store = VectorStore(
    index_file="data/memory_vectors",
    embedding_function=embedding_fn,
    embedding_dim=768
)

semantic_strategy = SemanticRetrievalStrategy(
    embedding_function=embedding_fn,
    similarity_threshold=0.7
)

config = MemorySystemConfig(
    llm_base_url="http://192.168.1.61:11434/v1",
    storage_backend=vector_store,
    retrieval_strategy=semantic_strategy
)

# Models auto-detected:
# - LLM: ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf
# - Embedding: nomic-ai_nomic-embed-text-v2-moe-GGUF_nomic-embed-text-v2-moe.Q5_K_M.gguf

memory_system = MemorySystem(config=config)
```

### 5. Explicit Model Specification (Optional)
```python
# Still supports explicit model if needed
config = MemorySystemConfig(
    llm_base_url="http://192.168.1.61:11434/v1",
    llm_model="specific-model-name"  # Optional override
)

embedding_fn = create_embedding_function(
    provider="openai",
    base_url="http://192.168.1.61:11435/v1",
    model="specific-embedding-model"  # Optional override
)
```

## Test Results

All tests pass:
- ✅ 17 semantic strategy tests
- ✅ 16 encoding metrics tests
- ✅ 4 encoder tests
- ✅ 1 memory system initialization test
- ✅ 38 total tests passing

## Semantic Similarity Validation

Tested embedding quality with similarity comparisons:

| Similar Pair | Similarity | Dissimilar Pair | Similarity | Difference |
|---------------|--------------|-------------------|--------------|-------------|
| machine learning vs deep learning | 0.7094 | vs cooking recipes | 0.1373 | 5.2x |
| python programming vs javascript coding | 0.3958 | vs guitar lessons | 0.2150 | 1.8x |
| database vs data storage | 0.4727 | vs flower gardening | 0.1972 | 2.4x |

Embeddings capture semantic meaning effectively.

## Configuration Summary

- **LLM chat (encoding):** `http://192.168.1.61:11434/v1` ✅
- **Embeddings (vector search):** `http://192.168.1.61:11435/v1` ✅
- **LLM Model:** ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf (auto-detected) ✅
- **Embedding Model:** nomic-ai_nomic-embed-text-v2-moe-GGUF_nomic-embed-text-v2-moe.Q5_K_M.gguf (auto-detected) ✅
- **Text-only embeddings:** As specified ✅
- **No model parameter required:** Both endpoints auto-detect models ✅
