from memevolve.components.manage import SimpleManagementStrategy, MemoryManager
from memevolve.components.store import JSONFileStore
import pytest
import tempfile
import sys

# sys.path.insert(0, 'src')  # No longer needed with package structure


@pytest.fixture
def temp_store():
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix='.json',
        mode='w'
    ) as f:
        filepath = f.name
        f.write("{}")
    store = JSONFileStore(filepath)

    store.store({
        "type": "lesson",
        "content": "Lesson 1",
        "tags": ["lesson"]
    })
    store.store({
        "type": "skill",
        "content": "Skill 1",
        "tags": ["skill"]
    })
    store.store({
        "type": "lesson",
        "content": "Lesson 2",
        "tags": ["lesson"]
    })
    store.store({
        "type": "abstraction",
        "content": "Abstraction 1",
        "tags": ["abstraction"]
    })

    yield store

    import os
    if os.path.exists(filepath):
        os.remove(filepath)


def test_prune_by_type(temp_store):
    strategy = SimpleManagementStrategy()
    manager = MemoryManager(temp_store, strategy)

    count, unit_ids = manager.prune({"type": "lesson"})

    assert count == 2
    assert len(unit_ids) == 2
    assert temp_store.count() == 2


def test_prune_by_max_count(temp_store):
    strategy = SimpleManagementStrategy()
    manager = MemoryManager(temp_store, strategy)

    count, unit_ids = manager.prune({"max_count": 2})

    assert count == 2
    assert len(unit_ids) == 2
    assert temp_store.count() == 2


def test_prune_no_criteria(temp_store):
    strategy = SimpleManagementStrategy()
    manager = MemoryManager(temp_store, strategy)

    count, unit_ids = manager.prune()

    assert count == 0
    assert len(unit_ids) == 0
    assert temp_store.count() == 4


def test_consolidate(temp_store):
    strategy = SimpleManagementStrategy()
    manager = MemoryManager(temp_store, strategy)

    consolidated = manager.consolidate()

    assert isinstance(consolidated, list)
    assert len(consolidated) > 0


def test_consolidate_by_ids(temp_store):
    strategy = SimpleManagementStrategy()
    manager = MemoryManager(temp_store, strategy)

    all_units = temp_store.retrieve_all()
    unit_ids = [u["id"] for u in all_units[:2]]
    consolidated = manager.consolidate(unit_ids)

    assert isinstance(consolidated, list)
    assert len(consolidated) >= 1


def test_deduplicate(temp_store):
    store = JSONFileStore(temp_store.filepath)

    store.store({
        "type": "lesson",
        "content": "Duplicate content",
        "tags": ["test"]
    })
    store.store({
        "type": "skill",
        "content": "Duplicate content",
        "tags": ["test"]
    })

    initial_count = store.count()
    strategy = SimpleManagementStrategy()
    manager = MemoryManager(store, strategy)

    count, unit_ids = manager.deduplicate()

    assert count == 1
    assert len(unit_ids) == 1
    assert store.count() == initial_count - 1


def test_deduplicate_no_duplicates(temp_store):
    strategy = SimpleManagementStrategy()
    manager = MemoryManager(temp_store, strategy)

    count, unit_ids = manager.deduplicate()

    assert count == 0
    assert len(unit_ids) == 0
    assert temp_store.count() == 4


def test_apply_forgetting_lru(temp_store):
    strategy = SimpleManagementStrategy()
    manager = MemoryManager(temp_store, strategy)

    count, unit_ids = manager.apply_forgetting("lru", count=1)

    assert count == 1
    assert len(unit_ids) == 1
    assert temp_store.count() == 3


def test_apply_forgetting_random(temp_store):
    strategy = SimpleManagementStrategy()
    manager = MemoryManager(temp_store, strategy)

    count, unit_ids = manager.apply_forgetting("random", count=1)

    assert count == 1
    assert len(unit_ids) == 1
    assert temp_store.count() == 3


def test_apply_forgetting_auto_count(temp_store):
    strategy = SimpleManagementStrategy()
    manager = MemoryManager(temp_store, strategy)

    initial_count = temp_store.count()
    count, unit_ids = manager.apply_forgetting("lru")

    assert count == max(1, initial_count // 10)
    assert temp_store.count() < initial_count


def test_get_health_metrics(temp_store):
    strategy = SimpleManagementStrategy()
    manager = MemoryManager(temp_store, strategy)

    metrics = manager.get_health_metrics()

    assert metrics.total_units == 4
    assert metrics.total_size_bytes > 0
    assert metrics.average_unit_size > 0
    assert metrics.last_operation == "metrics"


def test_health_metrics_type_distribution(temp_store):
    strategy = SimpleManagementStrategy()
    manager = MemoryManager(temp_store, strategy)

    metrics = manager.get_health_metrics()

    assert "lesson" in metrics.unit_types_distribution
    assert "skill" in metrics.unit_types_distribution
    assert metrics.unit_types_distribution["lesson"] == 2


def test_health_metrics_duplicates(temp_store):
    store = JSONFileStore(temp_store.filepath)
    store.store({
        "type": "test",
        "content": "Duplicate",
        "tags": ["test"]
    })
    store.store({
        "type": "test",
        "content": "Duplicate",
        "tags": ["test"]
    })

    strategy = SimpleManagementStrategy()
    manager = MemoryManager(store, strategy)

    metrics = manager.get_health_metrics()

    assert metrics.duplicate_count == 1


def test_health_metrics_empty_storage():
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix='.json',
        mode='w'
    ) as f:
        filepath = f.name
        f.write("{}")
    store = JSONFileStore(filepath)

    strategy = SimpleManagementStrategy()
    manager = MemoryManager(store, strategy)

    metrics = manager.get_health_metrics()

    assert metrics.total_units == 0
    assert metrics.average_unit_size == 0.0

    import os
    os.remove(filepath)


def test_prune_by_max_age(temp_store):
    strategy = SimpleManagementStrategy()
    manager = MemoryManager(temp_store, strategy)

    count, unit_ids = manager.prune({"max_age": 0})

    assert count == 4
    assert temp_store.count() == 0
