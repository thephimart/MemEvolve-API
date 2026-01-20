import pytest
from components.retrieve import KeywordRetrievalStrategy, RetrievalResult
import sys

sys.path.insert(0, 'src')


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
        "content": "Python programming language basics",
        "tags": ["python", "programming", "basics"]
    })
    storage.store({
        "type": "skill",
        "content": "Java development techniques",
        "tags": ["java", "development"]
    })
    storage.store({
        "type": "abstraction",
        "content": "General programming concepts and patterns",
        "tags": ["programming", "concepts", "patterns"]
    })
    storage.store({
        "type": "lesson",
        "content": "Python debugging methods",
        "tags": ["python", "debugging"]
    })
    return storage


@pytest.fixture
def keyword_strategy():
    return KeywordRetrievalStrategy(case_sensitive=False)


def test_strategy_initialization():
    strategy = KeywordRetrievalStrategy(case_sensitive=True)
    assert strategy.case_sensitive is True


def test_retrieve_with_query(keyword_strategy, mock_storage):
    results = keyword_strategy.retrieve("python", mock_storage, top_k=3)
    assert len(results) > 0
    assert all(isinstance(r, RetrievalResult) for r in results)

    python_results = [
        r for r in results if "python" in r.unit["content"].lower()]
    assert len(python_results) >= 2


def test_retrieve_with_top_k(keyword_strategy, mock_storage):
    results = keyword_strategy.retrieve("programming", mock_storage, top_k=2)
    assert len(results) <= 2


def test_retrieve_with_no_matches(keyword_strategy, mock_storage):
    results = keyword_strategy.retrieve("nonexistent", mock_storage)
    assert len(results) == 0


def test_retrieve_by_ids(keyword_strategy, mock_storage):
    unit_ids = ["unit_0", "unit_1"]
    results = keyword_strategy.retrieve_by_ids(unit_ids, mock_storage)
    assert len(results) == 2
    assert results[0].unit_id == "unit_0"
    assert results[1].unit_id == "unit_1"
    assert all(r.score == 1.0 for r in results)


def test_retrieve_by_ids_nonexistent(keyword_strategy, mock_storage):
    results = keyword_strategy.retrieve_by_ids(["nonexistent"], mock_storage)
    assert len(results) == 0


def test_search(keyword_strategy, mock_storage):
    results = keyword_strategy.search("java", mock_storage, top_k=5)
    assert len(results) > 0
    java_results = [r for r in results if "java" in r.unit["content"].lower()]
    assert len(java_results) >= 1


def test_count_relevant(keyword_strategy, mock_storage):
    count = keyword_strategy.count_relevant("python", mock_storage)
    assert count >= 2


def test_count_relevant_no_matches(keyword_strategy, mock_storage):
    count = keyword_strategy.count_relevant("nonexistent", mock_storage)
    assert count == 0


def test_retrieve_with_filters(keyword_strategy, mock_storage):
    filters = {"type": "lesson"}
    results = keyword_strategy.retrieve(
        "test", mock_storage, top_k=10, filters=filters)
    assert all(r.unit["type"] == "lesson" for r in results)


def test_case_sensitive_retrieval():
    strategy = KeywordRetrievalStrategy(case_sensitive=True)
    storage = MockStorage()
    storage.store({
        "id": "unit_0",
        "content": "Python",
        "type": "test"
    })

    results_upper = strategy.retrieve("Python", storage)
    results_lower = strategy.retrieve("python", storage)

    assert len(results_upper) >= 1
    assert len(results_lower) == 0


def test_scoring(keyword_strategy, mock_storage):
    results = keyword_strategy.retrieve(
        "python programming", mock_storage, top_k=5)
    assert all(r.score > 0 for r in results)
    assert results[0].score >= results[-1].score


def test_tag_matching(keyword_strategy, mock_storage):
    results = keyword_strategy.retrieve("debugging", mock_storage)
    assert len(results) >= 1
    assert "debugging" in results[0].unit["tags"]


def test_metadata_in_results(keyword_strategy, mock_storage):
    results = keyword_strategy.retrieve("python", mock_storage)
    assert len(results) > 0
    assert "matching_terms" in results[0].metadata


def test_multiple_keywords(keyword_strategy, mock_storage):
    results = keyword_strategy.retrieve("python programming", mock_storage)
    assert len(results) > 0


def test_empty_query(keyword_strategy, mock_storage):
    results = keyword_strategy.retrieve("", mock_storage)
    assert len(results) == 0


def test_short_terms_filtered(keyword_strategy, mock_storage):
    results = keyword_strategy.retrieve("a b c d e", mock_storage)
    assert len(results) == 0
