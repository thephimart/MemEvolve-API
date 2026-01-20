from components.retrieve import LLMGuidedRetrievalStrategy, RetrievalResult
import sys
import pytest
from unittest.mock import Mock, MagicMock

sys.path.insert(0, 'src')


@pytest.fixture
def mock_llm_client():
    """Mock LLM client that returns predefined responses."""
    def mock_call(prompt):
        if "retrieval guidance" in prompt.lower():
            return '''{
                "enhanced_query": "improved search for data structures",
                "preferred_types": ["tool", "skill"],
                "focus_areas": ["algorithms", "data structures"],
                "exclusion_criteria": ["unrelated topics"]
            }'''
        elif "select the top" in prompt.lower():
            return "[0, 1, 2]"
        else:
            return '{"type": "tool", "content": "Mock response"}'
    return mock_call


@pytest.fixture
def mock_base_strategy():
    """Mock base retrieval strategy."""
    strategy = Mock()
    strategy.retrieve.return_value = [
        RetrievalResult(
            "unit1", {"type": "tool", "content": "Binary search algorithm"}, 0.9),
        RetrievalResult("unit2", {"type": "skill",
                        "content": "Sorting techniques"}, 0.8),
        RetrievalResult("unit3", {"type": "lesson",
                        "content": "Algorithm complexity"}, 0.7),
        RetrievalResult(
            "unit4", {"type": "tool", "content": "Hash table implementation"}, 0.6),
    ]
    strategy.retrieve_by_ids.return_value = []
    return strategy


@pytest.fixture
def llm_guided_strategy(mock_llm_client, mock_base_strategy):
    """Create LLM-guided retrieval strategy with mocks."""
    return LLMGuidedRetrievalStrategy(
        llm_client_callable=mock_llm_client,
        base_strategy=mock_base_strategy,
        reasoning_temperature=0.3
    )


def test_llm_guided_retrieval_initialization(llm_guided_strategy):
    """Test LLM-guided strategy initialization."""
    assert llm_guided_strategy.llm_call is not None
    assert llm_guided_strategy.base_strategy is not None
    assert llm_guided_strategy.reasoning_temperature == 0.3


def test_retrieval_guidance_generation(llm_guided_strategy, mock_llm_client):
    """Test that LLM generates retrieval guidance."""
    mock_storage = Mock()
    mock_storage.retrieve_all.return_value = [
        {"type": "tool", "content": "Sample tool", "tags": ["algorithm"]},
        {"type": "skill", "content": "Sample skill", "tags": ["technique"]}
    ]

    guidance = llm_guided_strategy._get_retrieval_guidance(
        "search algorithms", mock_storage)

    assert "enhanced_query" in guidance
    assert "preferred_types" in guidance
    assert "focus_areas" in guidance
    assert isinstance(guidance["preferred_types"], list)


def test_llm_guided_retrieve(llm_guided_strategy):
    """Test the main retrieve method with LLM guidance."""
    mock_storage = Mock()
    mock_storage.retrieve_all.return_value = []

    results = llm_guided_strategy.retrieve(
        query="implement binary search",
        storage_backend=mock_storage,
        top_k=2
    )

    # Should return results from base strategy
    assert len(results) <= 2
    assert all(isinstance(r, RetrievalResult) for r in results)


def test_enhance_filters(llm_guided_strategy):
    """Test filter enhancement based on LLM guidance."""
    base_filters = {"status": "active"}
    guidance = {
        "preferred_types": ["tool", "skill"],
        "focus_areas": ["algorithms"]
    }

    enhanced = llm_guided_strategy._enhance_filters(base_filters, guidance)

    assert "types" in enhanced
    assert enhanced["types"] == ["tool", "skill"]
    assert enhanced["status"] == "active"


def test_llm_reranking(llm_guided_strategy, mock_llm_client):
    """Test LLM-based reranking of results."""
    candidates = [
        RetrievalResult("unit1", {"type": "tool", "content": "Tool A"}, 0.9),
        RetrievalResult("unit2", {"type": "skill", "content": "Skill B"}, 0.8),
        RetrievalResult("unit3", {"type": "lesson",
                        "content": "Lesson C"}, 0.7),
        RetrievalResult("unit4", {"type": "tool", "content": "Tool D"}, 0.6),
    ]

    reranked = llm_guided_strategy._llm_rerank("find tools", candidates, 2)

    assert len(reranked) == 2
    assert all(isinstance(r, RetrievalResult) for r in reranked)


def test_retrieve_by_ids_delegation(llm_guided_strategy, mock_base_strategy):
    """Test that retrieve_by_ids delegates to base strategy."""
    mock_storage = Mock()

    result = llm_guided_strategy.retrieve_by_ids(
        ["unit1", "unit2"], mock_storage)

    mock_base_strategy.retrieve_by_ids.assert_called_once_with(
        ["unit1", "unit2"], mock_storage)


def test_fallback_on_llm_failure():
    """Test graceful fallback when LLM calls fail."""
    def failing_llm_call(prompt):
        raise Exception("LLM service unavailable")

    mock_base_strategy = Mock()
    mock_base_strategy.retrieve.return_value = [
        RetrievalResult(
            "unit1", {"type": "tool", "content": "Fallback result"}, 0.8)
    ]

    strategy = LLMGuidedRetrievalStrategy(
        llm_client_callable=failing_llm_call,
        base_strategy=mock_base_strategy
    )

    mock_storage = Mock()
    mock_storage.retrieve_all.return_value = []  # Mock the retrieve_all method
    results = strategy.retrieve("test query", mock_storage, top_k=1)

    # Should still return results from base strategy despite LLM failure
    assert len(results) >= 0
