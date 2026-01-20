from utils.debug_utils import (
    MemoryInspector,
    MemoryDebugger,
    inspect_memory_system,
    quick_debug_report
)
import sys
import tempfile
import json
from pathlib import Path
import pytest

sys.path.insert(0, 'src')


class MockStorage:
    """Mock storage backend for testing."""

    def __init__(self):
        self.data = {
            "unit1": {
                "id": "unit1",
                "type": "lesson",
                "content": "Python basics content",
                "tags": ["python", "basics"],
                "metadata": {"created_at": "2024-01-01T00:00:00Z"}
            },
            "unit2": {
                "id": "unit2",
                "type": "skill",
                "content": "Debugging techniques content",
                "tags": ["debugging", "programming"],
                "metadata": {"created_at": "2024-01-02T00:00:00Z"}
            },
            "unit3": {
                "id": "unit3",
                "type": "lesson",
                "content": "Advanced Python content",
                "tags": ["python", "advanced"],
                "metadata": {"created_at": "2024-01-03T00:00:00Z"}
            }
        }

    def retrieve_all(self):
        return list(self.data.values())

    def retrieve(self, unit_id):
        return self.data.get(unit_id)

    def count(self):
        return len(self.data)


class MockMemorySystem:
    """Mock memory system for testing."""

    def __init__(self):
        self.storage = MockStorage()
        self.operation_log = [
            {"operation": "add_experience", "timestamp": "2024-01-01T00:00:00Z"},
            {"operation": "query_memory", "timestamp": "2024-01-02T00:00:00Z"}
        ]

    def get_health_metrics(self):
        from components.manage.base import HealthMetrics
        return HealthMetrics(
            total_units=3,
            total_size_bytes=1500,
            average_unit_size=500.0,
            oldest_unit_timestamp="2024-01-01T00:00:00Z",
            newest_unit_timestamp="2024-01-03T00:00:00Z",
            unit_types_distribution={"lesson": 2, "skill": 1},
            duplicate_count=0,
            last_operation="add_experience",
            last_operation_time="2024-01-03T00:00:00Z"
        )


def test_memory_inspector_initialization():
    """Test memory inspector initialization."""
    memory_system = MockMemorySystem()
    inspector = MemoryInspector(memory_system)
    assert inspector is not None
    assert inspector.memory_system == memory_system


def test_memory_inspector_get_system_overview():
    """Test getting system overview."""
    memory_system = MockMemorySystem()
    inspector = MemoryInspector(memory_system)

    overview = inspector.get_system_overview()

    assert "timestamp" in overview
    assert "system_info" in overview
    assert "health_metrics" in overview
    assert "operation_log" in overview
    assert overview["health_metrics"]["total_units"] == 3
    assert overview["operation_log"]["total_operations"] == 2


def test_memory_inspector_inspect_memory_contents():
    """Test memory contents inspection."""
    memory_system = MockMemorySystem()
    inspector = MemoryInspector(memory_system)

    contents = inspector.inspect_memory_contents()

    assert contents["total_units"] == 3
    assert contents["filtered_units"] == 3
    assert contents["returned_units"] == 3
    assert len(contents["units"]) == 3
    assert contents["unit_types"]["lesson"] == 2
    assert contents["unit_types"]["skill"] == 1

    # Test filtering by type
    lesson_contents = inspector.inspect_memory_contents(unit_type="lesson")
    assert lesson_contents["filtered_units"] == 2
    assert lesson_contents["returned_units"] == 2

    # Test limiting results
    limited_contents = inspector.inspect_memory_contents(limit=1)
    assert limited_contents["returned_units"] == 1
    assert len(limited_contents["units"]) == 1


def test_memory_inspector_inspect_memory_contents_with_content():
    """Test memory contents inspection with full content."""
    memory_system = MockMemorySystem()
    inspector = MemoryInspector(memory_system)

    contents = inspector.inspect_memory_contents(include_content=True, limit=1)

    assert len(contents["units"]) == 1
    unit = contents["units"][0]
    assert "content" in unit
    assert "full_data" in unit
    # Should be the most recent unit (unit3 with newest timestamp)
    assert unit["content"] == "Advanced Python content"


def test_memory_inspector_analyze_component_states():
    """Test component state analysis."""
    memory_system = MockMemorySystem()
    # Add mock components using setattr to avoid type checker issues
    setattr(memory_system, 'encoder', type('MockEncoder', (), {
        'get_metrics': lambda: type('MockMetrics', (), {
            'total_encodings': 10,
            'successful_encodings': 9,
            'failed_encodings': 1,
            'success_rate': 90.0,
            'average_encoding_time': 0.5
        })()
    })())
    setattr(memory_system, 'retriever', type('MockRetriever', (), {
        'get_metrics': lambda: type('MockMetrics', (), {
            'total_retrievals': 20,
            'successful_retrievals': 18,
            'failed_retrievals': 2,
            'calculate_success_rate': lambda: 90.0,
            'average_retrieval_time': 0.25
        })()
    })())

    inspector = MemoryInspector(memory_system)
    analysis = inspector.analyze_component_states()

    assert "components" in analysis
    assert analysis["components"]["encoder"]["initialized"] is True
    assert analysis["components"]["retriever"]["initialized"] is True
    assert analysis["components"]["storage"]["initialized"] is True
    # Note: metrics may not be present if the mock objects don't match expected interface


def test_memory_inspector_find_similar_units():
    """Test finding similar units."""
    memory_system = MockMemorySystem()
    inspector = MemoryInspector(memory_system)

    # Test finding similar units to unit1 (lesson about python basics)
    similar = inspector.find_similar_units("unit1")

    assert "target_unit" in similar
    assert "similar_units" in similar
    assert similar["target_unit"]["id"] == "unit1"
    assert similar["target_unit"]["type"] == "lesson"
    # unit3 should be most similar (python content)
    assert len(similar["similar_units"]) == 2

    # Check that unit3 (python advanced) is more similar than unit2 (debugging)
    unit3_score = next(u["similarity_score"]
                       for u in similar["similar_units"] if u["unit_id"] == "unit3")
    unit2_score = next(u["similarity_score"]
                       for u in similar["similar_units"] if u["unit_id"] == "unit2")
    assert unit3_score > unit2_score


def test_memory_inspector_export_debug_report():
    """Test exporting debug report."""
    memory_system = MockMemorySystem()
    inspector = MemoryInspector(memory_system)

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "debug_report.json"

        success = inspector.export_debug_report(filepath)
        assert success
        assert filepath.exists()

        # Verify report content
        with open(filepath, 'r', encoding='utf-8') as f:
            report = json.load(f)

        assert "debug_report" in report
        assert "system_overview" in report["debug_report"]
        assert "component_analysis" in report["debug_report"]
        assert "memory_contents_summary" in report["debug_report"]


def test_memory_debugger():
    """Test memory debugger functionality."""
    debugger = MemoryDebugger()

    # Add inspectors
    system1 = MockMemorySystem()
    system2 = MockMemorySystem()
    system2.storage.data["unit4"] = {
        "id": "unit4",
        "type": "tool",
        "content": "Tool content",
        "tags": ["tool"],
        "metadata": {"created_at": "2024-01-04T00:00:00Z"}
    }

    debugger.add_inspector("system1", system1)
    debugger.add_inspector("system2", system2)

    # Test listing systems
    systems = debugger.list_systems()
    assert "system1" in systems
    assert "system2" in systems

    # Test getting inspector
    inspector1 = debugger.get_inspector("system1")
    assert inspector1 is not None
    assert isinstance(inspector1, MemoryInspector)

    # Test system comparison
    comparison = debugger.compare_systems(["system1", "system2"])
    assert "systems_compared" in comparison
    assert "comparison" in comparison
    assert len(comparison["systems_compared"]) == 2

    # Test system report
    report = debugger.generate_system_report("system1")
    assert "system_name" in report
    assert report["system_name"] == "system1"
    assert "system_overview" in report

    # Test system report with details
    detailed_report = debugger.generate_system_report(
        "system1", include_details=True)
    assert "memory_contents" in detailed_report


def test_convenience_functions():
    """Test convenience functions."""
    memory_system = MockMemorySystem()

    # Test inspect_memory_system
    inspector = inspect_memory_system(memory_system)
    assert isinstance(inspector, MemoryInspector)

    # Test quick_debug_report
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "quick_debug.json"
        success = quick_debug_report(memory_system, filepath)
        assert success
        assert filepath.exists()


def test_memory_inspector_error_handling():
    """Test error handling in memory inspector."""

    class FailingMemorySystem:
        """Mock system that fails operations."""

        def get_health_metrics(self):
            raise Exception("Health check failed")

    failing_system = FailingMemorySystem()
    inspector = MemoryInspector(failing_system)

    # Should handle errors gracefully - get_system_overview should catch the exception
    # and continue with other information
    overview = inspector.get_system_overview()
    assert "system_info" in overview  # Should still have basic info

    # Test storage error handling
    contents = inspector.inspect_memory_contents()
    assert "error" in contents  # Should return error for failed operations
