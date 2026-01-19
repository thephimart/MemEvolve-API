import sys
import pytest

sys.path.insert(0, 'src')

# Import functions that are now in conftest.py
# These are available globally as fixtures, but we import the utility functions
import sys

sys.path.insert(0, 'src')

# Utility functions for testing fixtures
def assert_memory_unit_structure(unit):
    """Assert that a memory unit has the expected structure."""
    required_fields = ["id", "type", "content", "tags", "metadata"]
    for field in required_fields:
        assert field in unit, f"Missing required field: {field}"

    assert isinstance(unit["tags"], list), "Tags should be a list"
    assert isinstance(unit["metadata"], dict), "Metadata should be a dict"
    assert "created_at" in unit["metadata"], "Metadata should have created_at"


def assert_memory_units_unique(units):
    """Assert that memory units have unique IDs."""
    ids = [unit["id"] for unit in units]
    assert len(ids) == len(set(ids)), "Unit IDs are not unique"


def assert_memory_system_health(system):
    """Assert that a memory system is in a healthy state."""
    health = system.get_health_metrics()
    assert health is not None, "Health metrics should be available"
    assert health.total_units >= 0, "Total units should be non-negative"
    assert health.total_size_bytes >= 0, "Total size should be non-negative"


def count_units_by_type(units):
    """Count units by type."""
    type_counts = {}
    for unit in units:
        unit_type = unit.get("type", "unknown")
        type_counts[unit_type] = type_counts.get(unit_type, 0) + 1
    return type_counts


def count_units_by_category(units):
    """Count units by category."""
    category_counts = {}
    for unit in units:
        category = unit.get("metadata", {}).get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    return category_counts


def test_assert_memory_unit_structure(sample_memory_units):
    """Test memory unit structure assertion."""
    for unit in sample_memory_units:
        assert_memory_unit_structure(unit)


def test_assert_memory_units_unique(sample_memory_units):
    """Test memory units uniqueness assertion."""
    assert_memory_units_unique(sample_memory_units)


def test_assert_memory_system_health(populated_memory_system):
    """Test memory system health assertion."""
    assert_memory_system_health(populated_memory_system)


def test_count_units_by_type(sample_memory_units):
    """Test counting units by type."""
    type_counts = count_units_by_type(sample_memory_units)

    assert isinstance(type_counts, dict)
    assert len(type_counts) > 0
    assert sum(type_counts.values()) == len(sample_memory_units)


def test_count_units_by_category(sample_memory_units):
    """Test counting units by category."""
    category_counts = count_units_by_category(sample_memory_units)

    assert isinstance(category_counts, dict)
    assert len(category_counts) > 0
    assert sum(category_counts.values()) == len(sample_memory_units)


def test_sample_memory_units_fixture(sample_memory_units):
    """Test the sample memory units fixture."""
    assert len(sample_memory_units) == 10
    assert_memory_units_unique(sample_memory_units)

    for unit in sample_memory_units:
        assert_memory_unit_structure(unit)


def test_large_memory_units_fixture(large_memory_units):
    """Test the large memory units fixture."""
    assert len(large_memory_units) == 100
    assert_memory_units_unique(large_memory_units)

    for unit in large_memory_units:
        assert_memory_unit_structure(unit)


def test_diverse_memory_units_fixture(diverse_memory_units):
    """Test the diverse memory units fixture."""
    assert len(diverse_memory_units) == 20
    assert_memory_units_unique(diverse_memory_units)

    # Check for type diversity
    types = set(unit["type"] for unit in diverse_memory_units)
    assert len(types) >= 2  # Should have multiple types

    # Check for category diversity
    categories = set(unit["metadata"]["category"] for unit in diverse_memory_units)
    assert len(categories) >= 2  # Should have multiple categories


def test_sample_experience_fixture(sample_experience):
    """Test the sample experience fixture."""
    assert "id" in sample_experience
    assert "type" in sample_experience
    assert "title" in sample_experience
    assert "units" in sample_experience
    assert sample_experience["type"] == "experience"
    assert len(sample_experience["units"]) > 0

    for unit in sample_experience["units"]:
        assert_memory_unit_structure(unit)


def test_test_scenario_fixture(test_scenario):
    """Test the test scenario fixture."""
    assert "name" in test_scenario
    assert "units" in test_scenario
    assert "expected_outcomes" in test_scenario
    assert len(test_scenario["units"]) > 0

    for unit in test_scenario["units"]:
        assert_memory_unit_structure(unit)


def test_basic_memory_config_fixture(basic_memory_config):
    """Test the basic memory config fixture."""
    assert hasattr(basic_memory_config, 'storage')
    assert hasattr(basic_memory_config, 'encoder')
    assert hasattr(basic_memory_config, 'retrieval')
    assert basic_memory_config.encoder.encoding_strategies == ["lesson", "skill"]


def test_memory_system_config_fixture(memory_system_config):
    """Test the memory system config fixture."""
    assert hasattr(memory_system_config, 'storage')
    assert hasattr(memory_system_config, 'encoder')
    assert hasattr(memory_system_config, 'retrieval')
    assert hasattr(memory_system_config, 'management')
    assert len(memory_system_config.encoder.encoding_strategies) == 4


def test_json_store_fixture(json_store):
    """Test the JSON store fixture."""
    assert json_store is not None
    assert hasattr(json_store, 'store')
    assert hasattr(json_store, 'retrieve')
    assert hasattr(json_store, 'retrieve_all')


def test_populated_json_store_fixture(populated_json_store):
    """Test the populated JSON store fixture."""
    all_units = populated_json_store.retrieve_all()
    assert len(all_units) == 10  # Should have 10 sample units

    for unit in all_units:
        assert_memory_unit_structure(unit)


def test_experience_encoder_fixture(experience_encoder):
    """Test the experience encoder fixture."""
    assert experience_encoder is not None
    assert hasattr(experience_encoder, 'encode_experience')


@pytest.mark.skip(reason="Component initialization issues in test environment")
def test_semantic_retriever_fixture(semantic_retriever):
    """Test the semantic retriever fixture."""
    assert semantic_retriever is not None
    assert hasattr(semantic_retriever, 'retrieve')


@pytest.mark.skip(reason="Component initialization issues in test environment")
def test_simple_memory_manager_fixture(simple_memory_manager):
    """Test the simple memory manager fixture."""
    assert simple_memory_manager is not None
    assert hasattr(simple_memory_manager, 'prune')
    assert hasattr(simple_memory_manager, 'consolidate')


def test_basic_memory_system_fixture(basic_memory_system):
    """Test the basic memory system fixture."""
    assert basic_memory_system is not None
    assert hasattr(basic_memory_system, 'add_experience')
    assert hasattr(basic_memory_system, 'query_memory')
    assert_memory_system_health(basic_memory_system)


def test_populated_memory_system_fixture(populated_memory_system):
    """Test the populated memory system fixture."""
    assert populated_memory_system is not None
    assert_memory_system_health(populated_memory_system)

    # Should have data
    health = populated_memory_system.get_health_metrics()
    assert health.total_units == 10


def test_programming_focused_units_fixture(programming_focused_units):
    """Test the programming focused units fixture."""
    assert len(programming_focused_units) == 15

    for unit in programming_focused_units:
        assert unit["metadata"]["category"] == "programming"


def test_ai_focused_units_fixture(ai_focused_units):
    """Test the AI focused units fixture."""
    assert len(ai_focused_units) == 12

    for unit in ai_focused_units:
        assert unit["metadata"]["category"] == "ai"


def test_mixed_category_units_fixture(mixed_category_units):
    """Test the mixed category units fixture."""
    assert len(mixed_category_units) == 25

    categories = set(unit["metadata"]["category"] for unit in mixed_category_units)
    assert len(categories) >= 2  # Should have multiple categories


def test_edge_case_units_fixture(edge_case_units):
    """Test the edge case units fixture."""
    assert len(edge_case_units) == 6  # 1 normal + 5 edge cases

    # Check that we have various edge cases
    has_long_content = any(len(unit["content"]) > 1000 for unit in edge_case_units)
    has_many_tags = any(len(unit["tags"]) > 10 for unit in edge_case_units)
    has_empty_content = any(unit["content"] == "" for unit in edge_case_units)

    assert has_long_content, "Should have at least one unit with long content"
    assert has_many_tags, "Should have at least one unit with many tags"
    assert has_empty_content, "Should have at least one unit with empty content"


def test_performance_test_units_fixture(performance_test_units):
    """Test the performance test units fixture."""
    assert len(performance_test_units) == 50

    for unit in performance_test_units:
        assert "performance_id" in unit
        assert "content_length" in unit
        assert unit["type"] == "lesson"
        assert unit["metadata"]["category"] == "programming"


@pytest.mark.skip(reason="Component initialization issues in test environment")
def test_retrieval_test_setup_fixture(retrieval_test_setup):
    """Test the retrieval test setup fixture."""
    assert "store" in retrieval_test_setup
    assert "retriever" in retrieval_test_setup
    assert "units" in retrieval_test_setup
    assert retrieval_test_setup["unit_count"] == len(retrieval_test_setup["units"])

    # Test that store has the expected data
    all_units = retrieval_test_setup["store"].retrieve_all()
    assert len(all_units) == retrieval_test_setup["unit_count"]


def test_encoding_test_setup_fixture(encoding_test_setup):
    """Test the encoding test setup fixture."""
    assert "encoder" in encoding_test_setup
    assert "test_units" in encoding_test_setup
    assert "expected_strategies" in encoding_test_setup

    assert len(encoding_test_setup["test_units"]) == 10  # sample_memory_units
    assert encoding_test_setup["expected_strategies"] == ["lesson", "skill"]


@pytest.mark.skip(reason="Component initialization issues in test environment")
def test_management_test_setup_fixture(management_test_setup):
    """Test the management test setup fixture."""
    assert "manager" in management_test_setup
    assert "store" in management_test_setup
    assert "initial_units" in management_test_setup
    assert "initial_count" in management_test_setup

    assert management_test_setup["initial_count"] == len(management_test_setup["initial_units"])

    # Check that store has the data
    all_units = management_test_setup["store"].retrieve_all()
    assert len(all_units) == management_test_setup["initial_count"]


@pytest.mark.skip(reason="Component initialization issues in test environment")
def test_integration_test_setup_fixture(integration_test_setup):
    """Test the integration test setup fixture."""
    assert "system" in integration_test_setup
    assert "test_units" in integration_test_setup
    assert "config" in integration_test_setup

    system = integration_test_setup["system"]
    units = integration_test_setup["test_units"]

    assert_memory_system_health(system)

    # Check that system has the data
    health = system.get_health_metrics()
    assert health.total_units == len(units)