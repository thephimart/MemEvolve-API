from memevolve.components.store import GraphStorageBackend
import sys
import pytest
import tempfile
import json

# sys.path.insert(0, 'src')  # No longer needed with package structure


@pytest.fixture
def graph_backend():
    """Create graph storage backend (will use NetworkX fallback if Neo4j unavailable)."""
    return GraphStorageBackend(
        uri="bolt://localhost:7687",  # Will fallback to NetworkX
        create_relationships=True
    )


def test_graph_backend_initialization(graph_backend):
    """Test graph storage backend initialization."""
    assert graph_backend is not None
    assert hasattr(graph_backend, 'store')
    assert hasattr(graph_backend, 'retrieve')
    assert hasattr(graph_backend, 'query_related')


def test_store_single_unit(graph_backend):
    """Test storing a single memory unit."""
    unit = {
        "type": "lesson",
        "content": "Always validate input data",
        "tags": ["validation", "best-practices"],
        "metadata": {"importance": "high"}
    }

    unit_id = graph_backend.store(unit)
    assert unit_id is not None
    assert isinstance(unit_id, str)
    assert len(unit_id) > 0


def test_retrieve_stored_unit(graph_backend):
    """Test retrieving a stored memory unit."""
    unit = {
        "type": "skill",
        "content": "Binary search algorithm",
        "tags": ["algorithm", "search"],
        "metadata": {"complexity": "O(log n)"}
    }

    unit_id = graph_backend.store(unit)
    retrieved = graph_backend.retrieve(unit_id)

    assert retrieved is not None
    assert retrieved["type"] == unit["type"]
    assert retrieved["content"] == unit["content"]
    assert retrieved["tags"] == unit["tags"]
    assert retrieved["metadata"] == unit["metadata"]


def test_retrieve_nonexistent_unit(graph_backend):
    """Test retrieving a non-existent unit."""
    retrieved = graph_backend.retrieve("nonexistent_id")
    assert retrieved is None


def test_store_batch_units(graph_backend):
    """Test storing multiple units in batch."""
    units = [
        {
            "type": "lesson",
            "content": "Use meaningful variable names",
            "tags": ["coding", "readability"]
        },
        {
            "type": "skill",
            "content": "Quick sort implementation",
            "tags": ["algorithm", "sorting"]
        },
        {
            "type": "tool",
            "content": "Regular expressions for text processing",
            "tags": ["regex", "text-processing"]
        }
    ]

    unit_ids = graph_backend.store_batch(units)
    assert len(unit_ids) == 3
    assert all(isinstance(uid, str) for uid in unit_ids)

    # Verify all units can be retrieved
    for unit_id in unit_ids:
        retrieved = graph_backend.retrieve(unit_id)
        assert retrieved is not None


def test_update_unit(graph_backend):
    """Test updating a memory unit."""
    unit = {
        "type": "lesson",
        "content": "Original content",
        "tags": ["original"]
    }

    unit_id = graph_backend.store(unit)

    # Update the unit
    updated_unit = {
        "type": "lesson",
        "content": "Updated content",
        "tags": ["updated"],
        "metadata": {"status": "reviewed"}
    }

    success = graph_backend.update(unit_id, updated_unit)
    assert success is True

    # Verify update
    retrieved = graph_backend.retrieve(unit_id)
    assert retrieved["content"] == "Updated content"
    assert retrieved["tags"] == ["updated"]
    assert retrieved["metadata"]["status"] == "reviewed"


def test_update_nonexistent_unit(graph_backend):
    """Test updating a non-existent unit."""
    unit = {"type": "lesson", "content": "Test"}
    success = graph_backend.update("nonexistent", unit)
    assert success is False


def test_delete_unit(graph_backend):
    """Test deleting a memory unit."""
    unit = {"type": "skill", "content": "Test skill"}
    unit_id = graph_backend.store(unit)

    # Verify it exists
    assert graph_backend.exists(unit_id)

    # Delete it
    success = graph_backend.delete(unit_id)
    assert success is True

    # Verify it's gone
    assert not graph_backend.exists(unit_id)
    assert graph_backend.retrieve(unit_id) is None


def test_delete_nonexistent_unit(graph_backend):
    """Test deleting a non-existent unit."""
    success = graph_backend.delete("nonexistent")
    assert success is False


def test_exists_check(graph_backend):
    """Test checking if units exist."""
    unit = {"type": "tool", "content": "Test tool"}
    unit_id = graph_backend.store(unit)

    assert graph_backend.exists(unit_id)
    assert not graph_backend.exists("nonexistent")


def test_count_units(graph_backend):
    """Test counting stored units."""
    initial_count = graph_backend.count()

    # Store some units
    units = [
        {"type": "lesson", "content": "Lesson 1"},
        {"type": "skill", "content": "Skill 1"},
        {"type": "tool", "content": "Tool 1"}
    ]

    graph_backend.store_batch(units)
    new_count = graph_backend.count()

    assert new_count == initial_count + 3


def test_retrieve_all_units(graph_backend):
    """Test retrieving all stored units."""
    # Clear first
    graph_backend.clear()

    units = [
        {"type": "lesson", "content": "Lesson A"},
        {"type": "skill", "content": "Skill B"}
    ]

    graph_backend.store_batch(units)
    all_units = graph_backend.retrieve_all()

    assert len(all_units) == 2
    types = [u["type"] for u in all_units]
    assert "lesson" in types
    assert "skill" in types


def test_clear_storage(graph_backend):
    """Test clearing all stored units."""
    # Store some units
    units = [{"type": "lesson", "content": f"Lesson {i}"} for i in range(3)]
    graph_backend.store_batch(units)
    assert graph_backend.count() >= 3

    # Clear
    graph_backend.clear()
    assert graph_backend.count() == 0


def test_query_related_units(graph_backend):
    """Test querying related units through graph relationships."""
    # Store units with similar tags to create relationships
    units = [
        {
            "type": "lesson",
            "content": "Use descriptive variable names",
            "tags": ["coding", "readability", "best-practices"]
        },
        {
            "type": "skill",
            "content": "Code refactoring techniques",
            "tags": ["coding", "refactoring", "best-practices"]
        },
        {
            "type": "tool",
            "content": "Code formatting tools",
            "tags": ["coding", "tools", "formatting"]
        },
        {
            "type": "lesson",
            "content": "Unrelated math lesson",
            "tags": ["math", "algebra"]
        }
    ]

    unit_ids = graph_backend.store_batch(units)

    # Query related units for the first unit
    related = graph_backend.query_related(unit_ids[0], max_depth=2, limit=5)

    # Should find related units (with common tags)
    assert isinstance(related, list)

    # The exact behavior depends on the backend (Neo4j vs NetworkX)
    # but we should get some results for units with shared tags
    if len(related) > 0:
        for item in related:
            assert "unit" in item
            assert "relationship" in item
            assert "depth" in item


def test_get_graph_stats(graph_backend):
    """Test getting graph statistics."""
    # Store some test data
    units = [
        {"type": "lesson", "content": "Test lesson", "tags": ["test"]},
        {"type": "skill", "content": "Test skill", "tags": ["test"]}
    ]
    graph_backend.store_batch(units)

    stats = graph_backend.get_graph_stats()

    assert isinstance(stats, dict)
    assert "nodes" in stats
    assert "relationships" in stats
    assert "storage_type" in stats
    assert stats["nodes"] >= 2
    assert stats["storage_type"] in ["neo4j", "networkx", "dict_fallback"]


def test_deduplication(graph_backend):
    """Test that identical units get the same ID (deduplication)."""
    unit1 = {
        "type": "lesson",
        "content": "Always handle edge cases",
        "tags": ["testing", "robustness"]
    }

    unit2 = {
        "type": "lesson",
        "content": "Always handle edge cases",  # Same content
        "tags": ["testing", "robustness"]       # Same tags
    }

    id1 = graph_backend.store(unit1)
    id2 = graph_backend.store(unit2)

    # Should get the same ID for identical content
    assert id1 == id2

    # Count should still be 1
    assert graph_backend.count() == 1


def test_different_content_same_id(graph_backend):
    """Test that different content gets different IDs."""
    unit1 = {"type": "lesson", "content": "Content A"}
    unit2 = {"type": "lesson", "content": "Content B"}

    id1 = graph_backend.store(unit1)
    id2 = graph_backend.store(unit2)

    assert id1 != id2
    assert graph_backend.count() == 2
