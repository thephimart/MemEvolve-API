from memevolve.components.store import StorageBackend
import pytest
import sys

# sys.path.insert(0, 'src')  # No longer needed with package structure


class MockStorageBackend(StorageBackend):
    """Mock implementation for testing."""

    def __init__(self):
        self.data = {}
        self.counter = 0

    def store(self, unit: dict) -> str:
        unit_id = f"unit_{self.counter}"
        self.data[unit_id] = unit
        self.counter += 1
        return unit_id

    def store_batch(self, units: list) -> list:
        return [self.store(unit) for unit in units]

    def retrieve(self, unit_id: str):
        return self.data.get(unit_id)

    def retrieve_all(self) -> list:
        return list(self.data.values())

    def update(self, unit_id: str, unit: dict) -> bool:
        if unit_id in self.data:
            self.data[unit_id] = unit
            return True
        return False

    def delete(self, unit_id: str) -> bool:
        if unit_id in self.data:
            del self.data[unit_id]
            return True
        return False

    def exists(self, unit_id: str) -> bool:
        return unit_id in self.data

    def count(self) -> int:
        return len(self.data)

    def clear(self) -> None:
        self.data.clear()
        self.counter = 0

    def get_metadata(self, unit_id: str):
        unit = self.retrieve(unit_id)
        if unit and "metadata" in unit:
            return unit["metadata"]
        return None


@pytest.fixture
def storage():
    return MockStorageBackend()


def test_store_and_retrieve(storage):
    unit = {"type": "lesson", "content": "Test lesson"}
    unit_id = storage.store(unit)
    retrieved = storage.retrieve(unit_id)
    assert retrieved["content"] == "Test lesson"


def test_store_batch(storage):
    units = [
        {"type": "lesson", "content": "Lesson 1"},
        {"type": "skill", "content": "Skill 1"}
    ]
    ids = storage.store_batch(units)
    assert len(ids) == 2
    assert storage.count() == 2


def test_update(storage):
    unit_id = storage.store({"type": "lesson", "content": "Original"})
    updated = {"type": "lesson", "content": "Updated"}
    assert storage.update(unit_id, updated) is True
    assert storage.retrieve(unit_id)["content"] == "Updated"


def test_delete(storage):
    unit_id = storage.store({"type": "lesson", "content": "Test"})
    assert storage.delete(unit_id) is True
    assert storage.exists(unit_id) is False


def test_exists(storage):
    unit_id = storage.store({"type": "lesson", "content": "Test"})
    assert storage.exists(unit_id) is True
    assert storage.exists("nonexistent") is False


def test_count(storage):
    storage.store({"type": "lesson", "content": "Test 1"})
    storage.store({"type": "lesson", "content": "Test 2"})
    assert storage.count() == 2


def test_clear(storage):
    storage.store({"type": "lesson", "content": "Test"})
    storage.clear()
    assert storage.count() == 0


def test_retrieve_all(storage):
    storage.store({"type": "lesson", "content": "Test 1"})
    storage.store({"type": "skill", "content": "Test 2"})
    all_units = storage.retrieve_all()
    assert len(all_units) == 2


def test_get_metadata(storage):
    unit = {
        "type": "lesson",
        "content": "Test",
        "metadata": {"created_at": "2024-01-01T00:00:00Z"}
    }
    unit_id = storage.store(unit)
    metadata = storage.get_metadata(unit_id)
    assert metadata is not None
    assert "created_at" in metadata
