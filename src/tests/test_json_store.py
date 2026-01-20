import os
import pytest
from components.store import JSONFileStore
import sys

sys.path.insert(0, 'src')


@pytest.fixture
def json_store(tmp_path):
    filepath = tmp_path / "test_store.json"
    return JSONFileStore(str(filepath))


def test_json_store_initialization(json_store, tmp_path):
    json_store.store({"type": "test", "content": "test"})
    assert os.path.exists(tmp_path / "test_store.json")


def test_store_and_retrieve(json_store):
    unit = {
        "type": "lesson",
        "content": "Test lesson",
        "tags": ["test"]
    }
    unit_id = json_store.store(unit)
    assert json_store.exists(unit_id)
    retrieved = json_store.retrieve(unit_id)
    assert retrieved["content"] == "Test lesson"
    assert retrieved["type"] == "lesson"
    assert "metadata" in retrieved
    assert "created_at" in retrieved["metadata"]


def test_store_with_explicit_id(json_store):
    unit = {
        "id": "custom_id",
        "type": "lesson",
        "content": "Test"
    }
    unit_id = json_store.store(unit)
    assert unit_id == "custom_id"


def test_store_batch(json_store):
    units = [
        {"type": "lesson", "content": "Lesson 1"},
        {"type": "skill", "content": "Skill 1"},
        {"type": "abstraction", "content": "Abstraction 1"}
    ]
    ids = json_store.store_batch(units)
    assert len(ids) == 3
    assert json_store.count() == 3
    for unit_id in ids:
        assert json_store.exists(unit_id)


def test_update(json_store):
    unit_id = json_store.store({
        "type": "lesson",
        "content": "Original",
        "tags": ["original"]
    })
    updated_unit = {
        "type": "lesson",
        "content": "Updated",
        "tags": ["updated"]
    }
    assert json_store.update(unit_id, updated_unit) is True
    retrieved = json_store.retrieve(unit_id)
    assert retrieved["content"] == "Updated"
    assert "updated_at" in retrieved["metadata"]


def test_update_nonexistent(json_store):
    assert json_store.update("nonexistent", {"content": "test"}) is False


def test_delete(json_store):
    unit_id = json_store.store({
        "type": "lesson",
        "content": "Test"
    })
    assert json_store.delete(unit_id) is True
    assert json_store.exists(unit_id) is False
    assert json_store.count() == 0


def test_delete_nonexistent(json_store):
    assert json_store.delete("nonexistent") is False


def test_exists(json_store):
    unit_id = json_store.store({"type": "lesson", "content": "Test"})
    assert json_store.exists(unit_id) is True
    assert json_store.exists("nonexistent") is False


def test_count(json_store):
    assert json_store.count() == 0
    json_store.store({"type": "lesson", "content": "Test 1"})
    json_store.store({"type": "lesson", "content": "Test 2"})
    json_store.store({"type": "lesson", "content": "Test 3"})
    assert json_store.count() == 3


def test_clear(json_store):
    json_store.store({"type": "lesson", "content": "Test 1"})
    json_store.store({"type": "lesson", "content": "Test 2"})
    assert json_store.count() == 2
    json_store.clear()
    assert json_store.count() == 0


def test_retrieve_all(json_store):
    json_store.store({"type": "lesson", "content": "Test 1"})
    json_store.store({"type": "skill", "content": "Test 2"})
    json_store.store({"type": "abstraction", "content": "Test 3"})
    all_units = json_store.retrieve_all()
    assert len(all_units) == 3
    contents = [unit["content"] for unit in all_units]
    assert "Test 1" in contents
    assert "Test 2" in contents
    assert "Test 3" in contents


def test_get_metadata(json_store):
    unit_id = json_store.store({
        "type": "lesson",
        "content": "Test",
        "tags": ["test"]
    })
    metadata = json_store.get_metadata(unit_id)
    assert metadata is not None
    assert "created_at" in metadata


def test_persistence(tmp_path):
    filepath = tmp_path / "persistent_store.json"

    store1 = JSONFileStore(str(filepath))
    unit_id = store1.store({"type": "lesson", "content": "Test"})
    store1_count = store1.count()

    store2 = JSONFileStore(str(filepath))
    assert store2.count() == store1_count
    retrieved = store2.retrieve(unit_id)
    assert retrieved is not None
    assert retrieved["content"] == "Test"


def test_load_existing_file(tmp_path):
    filepath = tmp_path / "existing_store.json"
    existing_data = {
        "unit_1": {
            "id": "unit_1",
            "type": "lesson",
            "content": "Existing content",
            "metadata": {"created_at": "2024-01-01T00:00:00Z"}
        }
    }
    import json
    with open(filepath, 'w') as f:
        json.dump(existing_data, f)

    store = JSONFileStore(str(filepath))
    assert store.count() == 1
    retrieved = store.retrieve("unit_1")
    assert retrieved is not None
    assert retrieved["content"] == "Existing content"
