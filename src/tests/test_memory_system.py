import sys
import tempfile
import pytest

sys.path.insert(0, 'src')

from memory_system import MemorySystem, MemorySystemConfig
from components.store import JSONFileStore
from components.retrieve import KeywordRetrievalStrategy
from components.manage import SimpleManagementStrategy


@pytest.fixture
def temp_json_store():
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix='.json',
        mode='w'
    ) as f:
        filepath = f.name
        f.write("{}")
    return JSONFileStore(filepath)


@pytest.fixture
def memory_system(temp_json_store):
    config = MemorySystemConfig(
        storage_backend=temp_json_store,
        retrieval_strategy=KeywordRetrievalStrategy(),
        management_strategy=SimpleManagementStrategy(),
        log_level="WARNING"
    )
    return MemorySystem(config)


def test_memory_system_initialization(temp_json_store):
    config = MemorySystemConfig(
        storage_backend=temp_json_store,
        retrieval_strategy=KeywordRetrievalStrategy(),
        management_strategy=SimpleManagementStrategy(),
        log_level="WARNING"
    )
    system = MemorySystem(config)
    
    assert system.encoder is not None
    assert system.storage == temp_json_store
    assert system.retrieval_context is not None
    assert system.memory_manager is not None


def test_memory_system_default_config():
    system = MemorySystem()
    assert system.config.llm_base_url == "http://192.168.1.61:11434/v1"
    assert system.config.default_retrieval_top_k == 5
    assert system.config.enable_auto_management is True


def test_add_experience(memory_system):
    experience = {
        "id": "exp_001",
        "action": "search",
        "result": "found documents",
        "feedback": "positive"
    }
    
    unit_id = memory_system.add_experience(experience)
    
    assert unit_id is not None
    assert isinstance(unit_id, str)
    
    retrieved = memory_system.storage.retrieve(unit_id)
    assert retrieved is not None
    assert "type" in retrieved
    assert "content" in retrieved


def test_add_trajectory(memory_system):
    trajectory = [
        {
            "id": "exp_001",
            "action": "search",
            "result": "found documents"
        },
        {
            "id": "exp_002",
            "action": "analyze",
            "result": "completed analysis"
        }
    ]
    
    unit_ids = memory_system.add_trajectory(trajectory)
    
    assert len(unit_ids) == 2
    assert all(isinstance(uid, str) for uid in unit_ids)
    
    for unit_id in unit_ids:
        assert memory_system.storage.exists(unit_id)


def test_query_memory(memory_system):
    memory_system.add_experience({
        "id": "exp_001",
        "content": "Python programming basics",
        "type": "lesson",
        "tags": ["python", "programming"]
    })
    
    memory_system.add_experience({
        "id": "exp_002",
        "content": "Java development techniques",
        "type": "skill",
        "tags": ["java", "development"]
    })
    
    results = memory_system.query_memory("python", top_k=5)
    
    assert len(results) > 0
    assert any(
        "python" in r.get("content", "").lower() or
        "python" in str(r.get("tags", [])).lower()
        for r in results
    )


def test_query_memory_with_filters(memory_system):
    memory_system.add_experience({
        "id": "exp_001",
        "type": "lesson",
        "content": "Test lesson",
        "tags": ["test"]
    })
    
    memory_system.add_experience({
        "id": "exp_002",
        "type": "skill",
        "content": "Test skill",
        "tags": ["test"]
    })
    
    results = memory_system.query_memory(
        "test",
        filters={"type": "lesson"}
    )
    
    assert all(r.get("type") == "lesson" for r in results)


def test_retrieve_by_ids(memory_system):
    memory_system.add_experience({
        "id": "exp_001",
        "content": "Test content 1"
    })
    memory_system.add_experience({
        "id": "exp_002",
        "content": "Test content 2"
    })
    
    all_units = memory_system.storage.retrieve_all()
    unit_ids = [u["id"] for u in all_units]
    
    retrieved = memory_system.retrieve_by_ids(unit_ids)
    
    assert len(retrieved) == 2
    assert all(r.get("content") is not None for r in retrieved)


def test_generate_abstraction(memory_system):
    memory_system.add_experience({
        "id": "exp_001",
        "content": "Python is a programming language",
        "type": "lesson"
    })
    memory_system.add_experience({
        "id": "exp_002",
        "content": "Java is another programming language",
        "type": "lesson"
    })
    
    all_units = memory_system.storage.retrieve_all()
    unit_ids = [u["id"] for u in all_units[:1]]
    
    try:
        abstraction = memory_system.generate_abstraction(unit_ids)
        assert "abstraction" in abstraction or "content" in abstraction
    except RuntimeError:
        pass


def test_manage_prune(memory_system):
    for i in range(10):
        memory_system.add_experience({
            "id": f"exp_{i}",
            "type": "lesson" if i % 2 == 0 else "skill",
            "content": f"Test content {i}"
        })
    
    initial_count = memory_system.storage.count()
    memory_system.manage_memory("prune", criteria={"type": "skill"})
    
    final_count = memory_system.storage.count()
    assert final_count < initial_count


def test_manage_consolidate(memory_system):
    memory_system.add_experience({
        "id": "exp_001",
        "type": "lesson",
        "content": "Test lesson 1"
    })
    memory_system.add_experience({
        "id": "exp_002",
        "type": "lesson",
        "content": "Test lesson 2"
    })
    
    consolidated = memory_system.manage_memory("consolidate")
    
    assert isinstance(consolidated, list)


def test_manage_deduplicate(memory_system):
    memory_system.add_experience({
        "id": "exp_001",
        "content": "Duplicate content",
        "type": "test"
    })
    memory_system.add_experience({
        "id": "exp_002",
        "content": "Duplicate content",
        "type": "test"
    })
    
    initial_count = memory_system.storage.count()
    memory_system.manage_memory("deduplicate")
    
    final_count = memory_system.storage.count()
    assert final_count < initial_count


def test_manage_forget(memory_system):
    for i in range(10):
        memory_system.add_experience({
            "id": f"exp_{i}",
            "content": f"Test content {i}"
        })
    
    initial_count = memory_system.storage.count()
    memory_system.manage_memory("forget", strategy="lru", count=2)
    
    final_count = memory_system.storage.count()
    assert final_count == initial_count - 2


def test_get_health_metrics(memory_system):
    memory_system.add_experience({
        "id": "exp_001",
        "content": "Test content",
        "type": "lesson"
    })
    
    metrics = memory_system.get_health_metrics()
    
    assert metrics is not None
    assert metrics.total_units >= 1
    assert metrics.total_size_bytes > 0


def test_get_health_metrics_without_manager():
    config = MemorySystemConfig(log_level="WARNING")
    system = MemorySystem(config)
    
    metrics = system.get_health_metrics()
    assert metrics is None


def test_operation_log(memory_system):
    memory_system.add_experience({
        "id": "exp_001",
        "content": "Test"
    })
    
    log = memory_system.get_operation_log()
    
    assert len(log) >= 1
    assert "operation" in log[0]
    assert "timestamp" in log[0]


def test_clear_operation_log(memory_system):
    memory_system.add_experience({
        "id": "exp_001",
        "content": "Test"
    })
    
    memory_system.clear_operation_log()
    
    log = memory_system.get_operation_log()
    assert len(log) == 0


def test_auto_management_enabled(memory_system):
    config = MemorySystemConfig(
        storage_backend=memory_system.storage,
        retrieval_strategy=KeywordRetrievalStrategy(),
        management_strategy=SimpleManagementStrategy(),
        enable_auto_management=True,
        auto_prune_threshold=5,
        log_level="WARNING"
    )
    system = MemorySystem(config)
    
    for i in range(10):
        system.add_experience({
            "id": f"exp_{i}",
            "content": f"Test {i}"
        })
    
    assert system.storage is not None
    count = system.storage.count()
    assert count > 0


def test_auto_management_disabled(memory_system):
    config = MemorySystemConfig(
        storage_backend=memory_system.storage,
        retrieval_strategy=KeywordRetrievalStrategy(),
        management_strategy=SimpleManagementStrategy(),
        enable_auto_management=False,
        log_level="WARNING"
    )
    system = MemorySystem(config)
    
    for i in range(20):
        system.add_experience({
            "id": f"exp_{i}",
            "content": f"Test {i}"
        })
    
    assert system.storage is not None
    count = system.storage.count()
    assert count == 20


def test_callbacks(memory_system):
    callbacks = {
        "encode_called": False,
        "retrieve_called": False,
        "manage_called": False
    }
    
    def on_encode(unit_id, unit):
        callbacks["encode_called"] = True
    
    def on_retrieve(query, results):
        callbacks["retrieve_called"] = True
    
    def on_manage(operation, result):
        callbacks["manage_called"] = True
    
    memory_system.config.on_encode_complete = on_encode
    memory_system.config.on_retrieve_complete = on_retrieve
    memory_system.config.on_manage_complete = on_manage
    
    memory_system.add_experience({"id": "exp_001", "content": "Test"})
    memory_system.query_memory("test")
    memory_system.manage_memory("prune", criteria={})
    
    assert callbacks["encode_called"]
    assert callbacks["retrieve_called"]
    assert callbacks["manage_called"]


def test_error_handling_add_experience(memory_system):
    invalid_experience = None
    
    with pytest.raises(RuntimeError):
        memory_system.add_experience(invalid_experience)


def test_error_handling_query_memory(memory_system):
    config = MemorySystemConfig(
        storage_backend=memory_system.storage,
        log_level="WARNING"
    )
    config.retrieval_strategy = None
    system = MemorySystem(config)
    
    with pytest.raises(RuntimeError):
        system.query_memory("test")
