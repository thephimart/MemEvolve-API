import numpy as np
import pytest
from components.store import JSONFileStore
from components.retrieve import (
    HybridRetrievalStrategy,
    RetrievalResult
)
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
    load_dotenv()  # Load .env file

    from utils import create_embedding_function

    # Use actual configuration with fallback hierarchy: EMBEDDING_BASE_URL -> UPSTREAM_BASE_URL
    # No skipping - tests should fail if real endpoints aren't available
    embedding_fn = create_embedding_function("openai")
    return embedding_fn


@pytest.fixture
def hybrid_strategy():
    """Create a hybrid retrieval strategy for testing."""
    embedding_fn = get_test_embedding_function()
    return HybridRetrievalStrategy(
        embedding_function=embedding_fn,
        semantic_weight=0.7,
        keyword_weight=0.3
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
            "content": "machine learning algorithms and techniques",
            "tags": ["ai", "ml", "algorithms"]
        },
        {
            "id": "unit2",
            "type": "skill",
            "content": "python programming for developers",
            "tags": ["python", "programming", "dev"]
        },
        {
            "id": "unit3",
            "type": "lesson",
            "content": "web development using javascript frameworks",
            "tags": ["web", "javascript", "frameworks"]
        },
        {
            "id": "unit4",
            "type": "abstraction",
            "content": "artificial intelligence and deep learning models",
            "tags": ["ai", "deep-learning", "neural-networks"]
        }
    ]
    temp_store.store_batch(units)
    return temp_store


def test_hybrid_initialization(hybrid_strategy):
    """Test hybrid strategy initialization."""
    assert hybrid_strategy.semantic_weight == 0.7
    assert hybrid_strategy.keyword_weight == 0.3
    assert hybrid_strategy.semantic_strategy is not None
    assert hybrid_strategy.keyword_strategy is not None


def test_retrieve_with_query(hybrid_strategy, populated_store):
    """Test retrieving with a query."""
    results = hybrid_strategy.retrieve(
        "machine learning",
        populated_store,
        top_k=3
    )

    assert len(results) <= 3
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(r.score >= 0 for r in results)
    assert all("metadata" in r.__dict__ for r in results)


def test_retrieve_with_top_k(hybrid_strategy, populated_store):
    """Test retrieving with specific top_k."""
    results = hybrid_strategy.retrieve(
        "programming",
        populated_store,
        top_k=2
    )

    assert len(results) <= 2


def test_retrieve_with_different_weights():
    """Test retrieving with different weight configurations."""

    embedding_fn = get_test_embedding_function()
    strategy_equal = HybridRetrievalStrategy(
        embedding_function=embedding_fn,
        semantic_weight=0.5,
        keyword_weight=0.5
    )

    assert strategy_equal.semantic_weight == 0.5
    assert strategy_equal.keyword_weight == 0.5

    strategy_semantic_heavy = HybridRetrievalStrategy(
        embedding_function=embedding_fn,
        semantic_weight=0.9,
        keyword_weight=0.1
    )

    assert strategy_semantic_heavy.semantic_weight == 0.9
    assert strategy_semantic_heavy.keyword_weight == 0.1


def test_set_weights(hybrid_strategy):
    """Test updating weights dynamically."""
    hybrid_strategy.set_weights(semantic_weight=0.8, keyword_weight=0.2)

    assert hybrid_strategy.semantic_weight == 0.8
    assert hybrid_strategy.keyword_weight == 0.2


def test_set_weights_normalization(hybrid_strategy):
    """Test that weights are normalized."""
    hybrid_strategy.set_weights(semantic_weight=10, keyword_weight=90)

    assert abs(hybrid_strategy.semantic_weight - 0.1) < 0.001
    assert abs(hybrid_strategy.keyword_weight - 0.9) < 0.001


def test_set_weights_zero_total(hybrid_strategy):
    """Test weight normalization with zero total."""
    hybrid_strategy.set_weights(semantic_weight=0, keyword_weight=0)

    assert hybrid_strategy.semantic_weight == 0.5
    assert hybrid_strategy.keyword_weight == 0.5


def test_retrieve_by_ids(hybrid_strategy, populated_store):
    """Test retrieving by specific IDs."""
    results = hybrid_strategy.retrieve_by_ids(
        ["unit1", "unit2"],
        populated_store
    )

    assert len(results) == 2
    assert results[0].unit_id == "unit1"
    assert results[1].unit_id == "unit2"


def test_search(hybrid_strategy, populated_store):
    """Test searching for units."""
    results = hybrid_strategy.search(
        "web development",
        populated_store,
        top_k=3
    )

    assert all(isinstance(r, RetrievalResult) for r in results)


def test_count_relevant(hybrid_strategy, populated_store):
    """Test counting relevant units."""
    count = hybrid_strategy.count_relevant(
        "AI",
        populated_store
    )

    assert isinstance(count, int)
    assert count >= 0


def test_retrieve_with_filters(hybrid_strategy, populated_store):
    """Test retrieving with filters."""
    filters = {"type": "lesson"}
    results = hybrid_strategy.retrieve(
        "content",
        populated_store,
        filters=filters
    )

    assert all(r.unit.get("type") == "lesson" for r in results)


def test_hybrid_scoring(hybrid_strategy, populated_store):
    """Test that results have hybrid scoring metadata."""
    results = hybrid_strategy.retrieve(
        "test query",
        populated_store,
        top_k=10
    )

    for result in results:
        assert "metadata" in result.__dict__
        if result.metadata:
            assert "semantic_score" in result.metadata
            assert "keyword_score" in result.metadata


def test_combine_results_unique():
    """Test combining results from both strategies."""
    from components.retrieve import HybridRetrievalStrategy

    embedding_fn = get_test_embedding_function()
    strategy = HybridRetrievalStrategy(
        embedding_function=embedding_fn
    )

    import sys
    sys.path.insert(0, 'src')
    from components.retrieve.base import RetrievalResult

    semantic_results = [
        RetrievalResult(
            unit_id="u1",
            unit={"id": "u1", "content": "semantic match"},
            score=0.9
        ),
        RetrievalResult(
            unit_id="u2",
            unit={"id": "u2", "content": "semantic match 2"},
            score=0.7
        )
    ]

    keyword_results = [
        RetrievalResult(
            unit_id="u1",
            unit={"id": "u1", "content": "keyword match"},
            score=0.8
        ),
        RetrievalResult(
            unit_id="u3",
            unit={"id": "u3", "content": "keyword only"},
            score=0.6
        )
    ]

    combined = strategy._combine_results(
        semantic_results,
        keyword_results,
        "test"
    )

    assert len(combined) == 3

    u1_result = next((r for r in combined if r.unit_id == "u1"), None)
    assert u1_result is not None
    expected_score = 0.7 * 0.9 + 0.3 * 0.8
    assert abs(u1_result.score - expected_score) < 0.01


def test_combine_results_semantic_only():
    """Test combining with only semantic results."""
    from components.retrieve import HybridRetrievalStrategy

    embedding_fn = get_test_embedding_function()
    strategy = HybridRetrievalStrategy(
        embedding_function=embedding_fn
    )

    import sys
    sys.path.insert(0, 'src')
    from components.retrieve.base import RetrievalResult

    semantic_results = [
        RetrievalResult(
            unit_id="u1",
            unit={"id": "u1", "content": "semantic match"},
            score=0.9
        )
    ]

    keyword_results = []

    combined = strategy._combine_results(
        semantic_results,
        keyword_results,
        "test"
    )

    assert len(combined) == 1
    assert combined[0].score == 0.9
    assert combined[0].metadata["semantic_score"] == 0.9
    assert combined[0].metadata["keyword_score"] == 0
    assert combined[0].metadata["keyword_rank"] is None


def test_combine_results_keyword_only():
    """Test combining with only keyword results."""
    from components.retrieve import HybridRetrievalStrategy

    embedding_fn = get_test_embedding_function()
    strategy = HybridRetrievalStrategy(
        embedding_function=embedding_fn
    )

    import sys
    sys.path.insert(0, 'src')
    from components.retrieve.base import RetrievalResult

    semantic_results = []

    keyword_results = [
        RetrievalResult(
            unit_id="u1",
            unit={"id": "u1", "content": "keyword match"},
            score=0.8
        )
    ]

    combined = strategy._combine_results(
        semantic_results,
        keyword_results,
        "test"
    )

    assert len(combined) == 1
    assert combined[0].score == 0.8
    assert combined[0].metadata["semantic_score"] == 0
    assert combined[0].metadata["keyword_score"] == 0.8
    assert combined[0].metadata["semantic_rank"] is None
