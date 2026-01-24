"""
Example configuration for MemEvolve with llama.cpp embedding server.

This demonstrates how to configure MemEvolve to use your embedding endpoint
at http://192.168.1.61:11435/v1.
"""

from memevolve.utils import create_embedding_function
from memevolve.components.retrieve import SemanticRetrievalStrategy
from memevolve.components.store import VectorStore
from memevolve.components.manage import SimpleManagementStrategy
from memevolve.memory_system import MemorySystem, MemorySystemConfig


def example_vector_store():
    """Configure VectorStore with embedding endpoint."""

    embedding_fn = create_embedding_function(
        provider="openai",
        base_url="http://192.168.1.61:11435/v1"
    )

    vector_store = VectorStore(
        index_file="data/vectors",
        embedding_function=embedding_fn,
        embedding_dim=768
    )

    return vector_store


def example_semantic_retrieval():
    """Configure SemanticRetrievalStrategy with embedding endpoint."""

    embedding_fn = create_embedding_function(
        provider="openai",
        base_url="http://192.168.1.61:11435/v1"
    )

    strategy = SemanticRetrievalStrategy(
        embedding_function=embedding_fn,
        similarity_threshold=0.7
    )

    return strategy


def example_memory_system():
    """Configure MemorySystem with vector storage and semantic retrieval."""

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

    management_strategy = SimpleManagementStrategy()

    config = MemorySystemConfig(
        llm_base_url="http://192.168.1.61:11434/v1",
        storage_backend=vector_store,
        retrieval_strategy=semantic_strategy,
        management_strategy=management_strategy
    )

    memory_system = MemorySystem(config=config)

    return memory_system


if __name__ == "__main__":
    print("Embedding endpoint configured: http://192.168.1.61:11435/v1")
    print("LLM chat endpoint configured: http://192.168.1.61:11434/v1")

    memory_system = example_memory_system()

    print("\nMemorySystem configured with:")
    print("  - VectorStore backend")
    print("  - SemanticRetrievalStrategy")
    print("  - SimpleManagementStrategy")
