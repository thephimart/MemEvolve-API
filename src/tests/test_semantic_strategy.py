import os
import tempfile
from utils import create_embedding_function
from components.store import JSONFileStore
from components.retrieve import (
    SemanticRetrievalStrategy,
    RetrievalResult
)
import numpy as np
import pytest
import sys

sys.path.insert(0, 'src')


def dummy_embedding(text: str) -> np.ndarray:
    """Simple dummy embedding for testing."""
    np.random.seed(hash(text) % 2147483647)
    embedding = np.random.randn(10)
    return embedding / np.linalg.norm(embedding)


def get_test_embedding_function():
    """Get embedding function for testing using actual config from .env."""
    from dotenv import load_dotenv
    import os
    load_dotenv()  # Load .env file

    from utils import create_embedding_function

    # Override with correct test endpoints if .env has localhost
    if os.getenv("MEMEVOLVE_EMBEDDING_BASE_URL", "").startswith("http://localhost"):
        os.environ["MEMEVOLVE_EMBEDDING_BASE_URL"] = "http://192.168.1.61:11435/v1"
    if os.getenv("MEMEVOLVE_UPSTREAM_BASE_URL", "").startswith("http://localhost"):
        os.environ["MEMEVOLVE_UPSTREAM_BASE_URL"] = "http://192.168.1.61:11434/v1"

    # Use actual configuration with fallback hierarchy: EMBEDDING_BASE_URL -> UPSTREAM_BASE_URL
    # No skipping - tests should fail if real endpoints aren't available
    embedding_fn = create_embedding_function("openai")
    return embedding_fn


@pytest.fixture
def semantic_strategy():
    """Create a semantic retrieval strategy for testing."""
    embedding_fn = get_test_embedding_function()
    return SemanticRetrievalStrategy(
        embedding_function=embedding_fn,
        similarity_threshold=0.0
    )


@pytest.fixture
def temp_store(tmp_path):
    """Create a temporary JSON file store."""
    filepath = tmp_path / "test_store.json"
    return JSONFileStore(str(filepath))


@pytest.fixture
def populated_store(temp_store):
    """Populate store with test data."""
    units = [
        {
            "id": "unit1",
            "type": "lesson",
            "content": "machine learning is a field of AI",
            "tags": ["ai", "ml"]
        },
        {
            "id": "unit2",
            "type": "skill",
            "content": "python programming for data science",
            "tags": ["python", "data"]
        },
        {
            "id": "unit3",
            "type": "lesson",
            "content": "web development with javascript",
            "tags": ["web", "javascript"]
        },
        {
            "id": "unit4",
            "type": "abstraction",
            "content": "artificial intelligence and neural networks",
            "tags": ["ai", "neural"]
        }
    ]
    temp_store.store_batch(units)
    return temp_store


def test_strategy_initialization(semantic_strategy):
    """Test semantic strategy initialization."""
    assert semantic_strategy.similarity_threshold == 0.0
    assert len(semantic_strategy._cache) == 0
    assert callable(semantic_strategy.embedding_function)


def test_retrieve_with_query(semantic_strategy, populated_store):
    """Test retrieving with a query."""
    results = semantic_strategy.retrieve(
        "machine learning",
        populated_store,
        top_k=3
    )

    assert len(results) <= 3
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(r.score >= 0 for r in results)
    assert all(r.score <= 1 for r in results)


def test_retrieve_with_top_k(semantic_strategy, populated_store):
    """Test retrieving with specific top_k."""
    results = semantic_strategy.retrieve(
        "programming",
        populated_store,
        top_k=2
    )

    assert len(results) <= 2


def test_retrieve_with_similarity_threshold(tmp_path):
    """Test retrieving with similarity threshold."""
    embedding_fn = get_test_embedding_function()
    strategy = SemanticRetrievalStrategy(
        embedding_function=embedding_fn,
        similarity_threshold=0.5
    )

    filepath = tmp_path / "test_store.json"
    store = JSONFileStore(str(filepath))
    units = [
        {
            "id": "unit1",
            "type": "lesson",
            "content": "test content",
            "tags": []
        }
    ]
    store.store_batch(units)

    results = strategy.retrieve(
        "test",
        store,
        top_k=5
    )

    assert all(r.score >= 0.5 for r in results)


def test_retrieve_with_no_matches(semantic_strategy, tmp_path):
    """Test retrieving when no matches found."""
    filepath = tmp_path / "test_store.json"
    store = JSONFileStore(str(filepath))

    results = semantic_strategy.retrieve(
        "nonexistent query",
        store,
        top_k=5
    )

    assert len(results) == 0


def test_retrieve_by_ids(semantic_strategy, populated_store):
    """Test retrieving by specific IDs."""
    results = semantic_strategy.retrieve_by_ids(
        ["unit1", "unit2"],
        populated_store
    )

    assert len(results) == 2
    assert results[0].unit_id == "unit1"
    assert results[1].unit_id == "unit2"
    assert all(r.score == 1.0 for r in results)


def test_retrieve_by_ids_nonexistent(semantic_strategy, populated_store):
    """Test retrieving by IDs with some nonexistent IDs."""
    results = semantic_strategy.retrieve_by_ids(
        ["unit1", "nonexistent", "unit2"],
        populated_store
    )

    assert len(results) == 2
    assert results[0].unit_id == "unit1"
    assert results[1].unit_id == "unit2"


def test_search(semantic_strategy, populated_store):
    """Test searching for units."""
    results = semantic_strategy.search(
        "web development",
        populated_store,
        top_k=3
    )

    assert all(isinstance(r, RetrievalResult) for r in results)


def test_count_relevant(semantic_strategy, populated_store):
    """Test counting relevant units."""
    count = semantic_strategy.count_relevant(
        "AI",
        populated_store
    )

    assert isinstance(count, int)
    assert count >= 0


def test_retrieve_with_filters(semantic_strategy, populated_store):
    """Test retrieving with filters."""
    filters = {"type": "lesson"}
    results = semantic_strategy.retrieve(
        "content",
        populated_store,
        filters=filters
    )

    assert all(r.unit.get("type") == "lesson" for r in results)


def test_scoring(semantic_strategy, populated_store):
    """Test that results are properly scored and sorted."""
    results = semantic_strategy.retrieve(
        "test query",
        populated_store,
        top_k=10
    )

    if len(results) > 1:
        for i in range(len(results) - 1):
            assert results[i].score >= results[i+1].score


def test_embedding_caching(semantic_strategy):
    """Test that embeddings are cached."""
    text = "test text for caching"
    embedding1 = semantic_strategy._get_embedding(text)
    embedding2 = semantic_strategy._get_embedding(text)

    np.testing.assert_array_equal(embedding1, embedding2)
    assert text in semantic_strategy._cache


def test_clear_cache(semantic_strategy):
    """Test clearing the embedding cache."""
    semantic_strategy._get_embedding("test text")
    assert len(semantic_strategy._cache) > 0

    semantic_strategy.clear_cache()
    assert len(semantic_strategy._cache) == 0


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    embedding_fn = get_test_embedding_function()
    strategy = SemanticRetrievalStrategy(
        embedding_function=embedding_fn
    )

    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    vec3 = np.array([1.0, 0.0, 0.0])

    sim12 = strategy._cosine_similarity(vec1, vec2)
    sim13 = strategy._cosine_similarity(vec1, vec3)

    assert abs(sim12) < 0.01
    assert abs(sim13 - 1.0) < 0.01


def test_unit_to_text(semantic_strategy):
    """Test converting unit to text."""
    unit = {
        "id": "test",
        "type": "lesson",
        "content": "test content",
        "tags": ["tag1", "tag2"]
    }

    text = semantic_strategy._unit_to_text(unit)
    assert "test content" in text
    assert "tag1" in text
    assert "tag2" in text
    assert "lesson" in text


def test_create_embedding_function():
    """Test creating embedding function from utility."""
    embedding_fn = create_embedding_function(provider="dummy")

    assert callable(embedding_fn)

    embedding = embedding_fn("test text")
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == 768


def test_create_embedding_function_with_custom_dim():
    """Test creating embedding function with custom dimension."""
    embedding_fn = create_embedding_function(
        provider="dummy",
        embedding_dim=128
    )

    embedding = embedding_fn("test text")
    assert len(embedding) == 128
