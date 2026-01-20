import pytest
from components.manage import (
    ManagementStrategy,
    MemoryManager,
    HealthMetrics
)
import sys

sys.path.insert(0, 'src')


class MockManagementStrategy(ManagementStrategy):
    """Mock management strategy for testing."""

    def prune(
        self,
        storage_backend,
        criteria=None
    ) -> tuple:
        all_units = storage_backend.retrieve_all()
        pruned_ids = []

        if criteria and criteria.get("type") == "test":
            for unit in all_units:
                if unit.get("type") == "test":
                    pruned_ids.append(unit.get("id", ""))

        for unit_id in pruned_ids:
            storage_backend.delete(unit_id)

        return len(pruned_ids), pruned_ids

    def consolidate(
        self,
        storage_backend,
        units=None
    ) -> list:
        return storage_backend.retrieve_all()

    def deduplicate(
        self,
        storage_backend,
        similarity_threshold=0.9
    ) -> tuple:
        return 0, []

    def apply_forgetting(
        self,
        storage_backend,
        strategy="lru",
        count=None
    ) -> tuple:
        return 0, []

    def get_health_metrics(self, storage_backend) -> HealthMetrics:
        all_units = storage_backend.retrieve_all()
        return HealthMetrics(
            total_units=len(all_units),
            total_size_bytes=0,
            average_unit_size=0.0,
            oldest_unit_timestamp=None,
            newest_unit_timestamp=None,
            unit_types_distribution={},
            duplicate_count=0,
            last_operation="test",
            last_operation_time="2024-01-01T00:00:00Z"
        )


class MockStorage:
    """Mock storage backend for testing."""

    def __init__(self):
        self.data = {}

    def store(self, unit: dict) -> str:
        unit_id = f"unit_{len(self.data)}"
        unit["id"] = unit_id
        unit["metadata"] = unit.get("metadata", {})
        unit["metadata"]["created_at"] = "2024-01-01T00:00:00Z"
        self.data[unit_id] = unit
        return unit_id

    def retrieve(self, unit_id: str):
        return self.data.get(unit_id)

    def retrieve_all(self) -> list:
        return list(self.data.values())

    def delete(self, unit_id: str) -> bool:
        if unit_id in self.data:
            del self.data[unit_id]
            return True
        return False

    def exists(self, unit_id: str) -> bool:
        return unit_id in self.data

    def count(self) -> int:
        return len(self.data)


@pytest.fixture
def mock_storage():
    storage = MockStorage()
    storage.store({
        "type": "test",
        "content": "Test unit 1",
        "tags": ["test"]
    })
    storage.store({
        "type": "lesson",
        "content": "Lesson unit",
        "tags": ["lesson"]
    })
    storage.store({
        "type": "test",
        "content": "Test unit 2",
        "tags": ["test"]
    })
    return storage


@pytest.fixture
def mock_strategy():
    return MockManagementStrategy()


@pytest.fixture
def memory_manager(mock_storage, mock_strategy):
    return MemoryManager(mock_storage, mock_strategy)


def test_health_metrics_creation():
    metrics = HealthMetrics(
        total_units=10,
        total_size_bytes=1000,
        average_unit_size=100.0,
        oldest_unit_timestamp="2024-01-01T00:00:00Z",
        newest_unit_timestamp="2024-01-02T00:00:00Z",
        unit_types_distribution={"lesson": 5, "skill": 5},
        duplicate_count=0,
        last_operation="test",
        last_operation_time="2024-01-01T00:00:00Z"
    )
    assert metrics.total_units == 10
    assert metrics.average_unit_size == 100.0


def test_memory_manager_initialization(memory_manager):
    assert memory_manager.storage_backend is not None
    assert memory_manager.management_strategy is not None
    assert len(memory_manager.operation_history) == 0


def test_set_strategy(memory_manager):
    new_strategy = MockManagementStrategy()
    memory_manager.set_strategy(new_strategy)
    assert memory_manager.management_strategy == new_strategy


def test_prune(memory_manager):
    count, unit_ids = memory_manager.prune({"type": "test"})
    assert count == 2
    assert len(unit_ids) == 2
    assert len(memory_manager.operation_history) == 1


def test_consolidate(memory_manager):
    consolidated = memory_manager.consolidate()
    assert isinstance(consolidated, list)
    assert len(memory_manager.operation_history) == 1


def test_deduplicate(memory_manager):
    count, unit_ids = memory_manager.deduplicate()
    assert count == 0
    assert len(unit_ids) == 0
    assert len(memory_manager.operation_history) == 1


def test_apply_forgetting(memory_manager):
    count, unit_ids = memory_manager.apply_forgetting()
    assert count == 0
    assert len(unit_ids) == 0
    assert len(memory_manager.operation_history) == 1


def test_get_health_metrics(memory_manager):
    metrics = memory_manager.get_health_metrics()
    assert isinstance(metrics, HealthMetrics)
    assert metrics.total_units == 3


def test_operation_history(memory_manager):
    memory_manager.prune({"type": "test"})
    memory_manager.consolidate()

    history = memory_manager.get_operation_history()
    assert len(history) == 2
    assert history[0]["operation"] == "prune"
    assert history[1]["operation"] == "consolidate"


def test_clear_history(memory_manager):
    memory_manager.prune({"type": "test"})
    assert len(memory_manager.operation_history) == 1

    memory_manager.clear_history()
    assert len(memory_manager.operation_history) == 0


def test_prune_affects_storage(memory_manager):
    initial_count = memory_manager.storage_backend.count()
    memory_manager.prune({"type": "test"})
    final_count = memory_manager.storage_backend.count()
    assert final_count < initial_count
