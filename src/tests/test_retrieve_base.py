import pytest
from components.retrieve import (
    RetrievalStrategy,
    RetrievalContext,
    RetrievalResult
)
import sys

sys.path.insert(0, 'src')


class MockRetrievalStrategy(RetrievalStrategy):
    """Mock retrieval strategy for testing."""

    def retrieve(
        self,
        query: str,
        storage_backend,
        top_k: int = 5,
        filters=None
    ) -> list:
        all_units = storage_backend.retrieve_all()
        results = []
        for i, unit in enumerate(all_units[:top_k]):
            results.append(RetrievalResult(
                unit_id=unit.get("id", f"unit_{i}"),
                unit=unit,
                score=1.0 - (i * 0.1)
            ))
        return results

    def retrieve_by_ids(
        self,
        unit_ids: list,
        storage_backend
    ) -> list:
        results = []
        for unit_id in unit_ids:
            unit = storage_backend.retrieve(unit_id)
            if unit:
                results.append(RetrievalResult(
                    unit_id=unit_id,
                    unit=unit,
                    score=1.0
                ))
        return results

    def search(
        self,
        query: str,
        storage_backend,
        top_k: int = 5,
        filters=None
    ) -> list:
        return self.retrieve(query, storage_backend, top_k, filters)

    def count_relevant(
        self,
        query: str,
        storage_backend,
        filters=None
    ) -> int:
        all_units = storage_backend.retrieve_all()
        return len(all_units)


class MockStorage:
    """Mock storage backend for testing."""

    def __init__(self):
        self.data = {}

    def store(self, unit: dict) -> str:
        unit_id = f"unit_{len(self.data)}"
        unit["id"] = unit_id
        self.data[unit_id] = unit
        return unit_id

    def retrieve(self, unit_id: str):
        return self.data.get(unit_id)

    def retrieve_all(self) -> list:
        return list(self.data.values())


@pytest.fixture
def mock_storage():
    storage = MockStorage()
    storage.store({
        "type": "lesson",
        "content": "Test lesson about Python programming",
        "tags": ["python", "programming"]
    })
    storage.store({
        "type": "skill",
        "content": "Python debugging techniques",
        "tags": ["python", "debugging"]
    })
    storage.store({
        "type": "abstraction",
        "content": "General programming concepts",
        "tags": ["programming", "concepts"]
    })
    return storage


@pytest.fixture
def mock_strategy():
    return MockRetrievalStrategy()


@pytest.fixture
def retrieval_context(mock_strategy):
    return RetrievalContext(strategy=mock_strategy, default_top_k=2)


def test_retrieval_result_creation():
    result = RetrievalResult(
        unit_id="test_id",
        unit={"content": "test"},
        score=0.9
    )
    assert result.unit_id == "test_id"
    assert result.score == 0.9
    assert result.metadata is None


def test_retrieval_context_initialization(retrieval_context):
    assert retrieval_context.strategy is not None
    assert retrieval_context.default_top_k == 2
    assert len(retrieval_context.retrieval_history) == 0


def test_retrieve_with_context(retrieval_context, mock_storage):
    results = retrieval_context.retrieve("python", mock_storage)
    assert len(results) == 2
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert len(retrieval_context.retrieval_history) == 1


def test_retrieve_with_custom_top_k(retrieval_context, mock_storage):
    results = retrieval_context.retrieve("test", mock_storage, top_k=1)
    assert len(results) == 1


def test_retrieve_by_ids(retrieval_context, mock_storage):
    results = retrieval_context.retrieve_by_ids(["unit_0"], mock_storage)
    assert len(results) == 1
    assert results[0].unit_id == "unit_0"


def test_search(retrieval_context, mock_storage):
    results = retrieval_context.search("python", mock_storage)
    assert isinstance(results, list)


def test_count_relevant(retrieval_context, mock_storage):
    count = retrieval_context.count_relevant("test", mock_storage)
    assert count == 3


def test_retrieval_history(retrieval_context, mock_storage):
    retrieval_context.retrieve("query1", mock_storage)
    retrieval_context.retrieve("query2", mock_storage)

    history = retrieval_context.get_retrieval_history()
    assert len(history) == 2
    assert history[0]["query"] == "query1"
    assert history[1]["query"] == "query2"
    assert "timestamp" in history[0]
    assert "results_count" in history[0]


def test_clear_history(retrieval_context, mock_storage):
    retrieval_context.retrieve("test", mock_storage)
    assert len(retrieval_context.get_retrieval_history()) == 1

    retrieval_context.clear_history()
    assert len(retrieval_context.get_retrieval_history()) == 0


def test_set_strategy(retrieval_context):
    new_strategy = MockRetrievalStrategy()
    retrieval_context.set_strategy(new_strategy)
    assert retrieval_context.strategy == new_strategy
