import sys

sys.path.insert(0, 'src')

import tempfile
import time
from pathlib import Path

sys.path.insert(0, 'src')

from utils.profiling import (
    MemoryProfiler,
    ProfileResult,
    PerformanceReport,
    profile_memory_operation,
    benchmark_memory_system
)


class MockMemorySystem:
    """Mock memory system for testing."""

    def __init__(self):
        self.operations_called = []

    def add_experience(self, experience):
        """Mock add experience."""
        self.operations_called.append(("add_experience", experience))
        time.sleep(0.01)  # Simulate some work
        return f"id_{len(self.operations_called)}"

    def query_memory(self, query, **kwargs):
        """Mock query memory."""
        self.operations_called.append(("query_memory", query, kwargs))
        time.sleep(0.005)  # Simulate some work
        return [{"id": "result_1", "content": "test result"}]

    def custom_operation(self, param):
        """Mock custom operation."""
        self.operations_called.append(("custom_operation", param))
        time.sleep(0.002)  # Simulate some work
        return f"result_{param}"


def test_memory_profiler_initialization():
    """Test profiler initialization."""
    profiler = MemoryProfiler()
    assert profiler is not None
    assert len(profiler.results) == 0


def test_profile_operation_context_manager():
    """Test profiling with context manager."""
    profiler = MemoryProfiler()

    with profiler.profile_operation("test_operation", param1="value1"):
        time.sleep(0.01)

    assert len(profiler.results) == 1
    result = profiler.results[0]
    assert result.operation_name == "test_operation"
    assert result.duration_seconds >= 0.01
    assert result.metadata["param1"] == "value1"


def test_profile_function():
    """Test profiling a function call."""
    profiler = MemoryProfiler()

    def test_func(x, y=10):
        time.sleep(0.005)
        return x + y

    result = profiler.profile_function(test_func, 5, y=15, operation_name="add_func")

    assert result == 20
    assert len(profiler.results) == 1
    assert profiler.results[0].operation_name == "add_func"
    assert profiler.results[0].duration_seconds >= 0.005


def test_profile_memory_system_operation():
    """Test profiling memory system operations."""
    profiler = MemoryProfiler()
    memory_system = MockMemorySystem()

    result = profiler.profile_memory_system_operation(
        memory_system, "add_experience",
        {"type": "lesson", "content": "test"}
    )

    assert result.startswith("id_")
    assert len(profiler.results) == 1
    assert profiler.results[0].operation_name == "memory_system.add_experience"


def test_generate_report_empty():
    """Test generating report with no data."""
    profiler = MemoryProfiler()

    report = profiler.generate_report("test_report")

    assert report.report_id == "test_report"
    assert report.total_operations == 0
    assert report.total_duration == 0.0
    assert "No profiling data available" in report.recommendations


def test_generate_report_with_data():
    """Test generating report with profiling data."""
    profiler = MemoryProfiler()

    # Add some mock results
    profiler.results = [
        ProfileResult(
            operation_name="fast_op",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:00:01Z",
            duration_seconds=0.1
        ),
        ProfileResult(
            operation_name="slow_op",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:00:05Z",
            duration_seconds=10.0  # Make this very slow to trigger bottleneck
        ),
        ProfileResult(
            operation_name="slow_op",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:00:03Z",
            duration_seconds=8.0
        )
    ]

    report = profiler.generate_report()

    assert report.total_operations == 3
    assert abs(report.total_duration - 18.1) < 0.001
    assert abs(report.average_operation_time - 6.033) < 0.01
    assert "fast_op" in report.operation_breakdown
    assert "slow_op" in report.operation_breakdown
    # Bottleneck detection depends on the threshold logic


def test_clear_results():
    """Test clearing profiling results."""
    profiler = MemoryProfiler()

    with profiler.profile_operation("test"):
        pass

    assert len(profiler.results) == 1

    profiler.clear_results()
    assert len(profiler.results) == 0


def test_export_profile_stats():
    """Test exporting profile statistics."""
    profiler = MemoryProfiler()

    with profiler.profile_operation("test_export"):
        time.sleep(0.001)

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "profile_stats.txt"

        success = profiler.export_profile_stats(filepath)
        assert success
        assert filepath.exists()

        content = filepath.read_text()
        assert len(content) > 0
        assert "function calls" in content.lower()


def test_run_benchmark():
    """Test running benchmark suite."""
    profiler = MemoryProfiler()
    memory_system = MockMemorySystem()

    operations = [
        {
            "name": "add_lesson",
            "type": "add_experience",
            "args": [{"type": "lesson", "content": "test lesson"}]
        },
        {
            "name": "query_test",
            "type": "query_memory",
            "args": ["test query"],
            "kwargs": {"top_k": 5}
        }
    ]

    report = profiler.run_benchmark(memory_system, operations, iterations=1)

    assert report.total_operations == 2
    assert len(memory_system.operations_called) == 2
    assert report.operation_breakdown["memory_system.add_experience"]["count"] == 1
    assert report.operation_breakdown["memory_system.query_memory"]["count"] == 1


def test_convenience_functions():
    """Test convenience profiling functions."""
    memory_system = MockMemorySystem()

    # Test profile_memory_operation
    result = profile_memory_operation(
        memory_system, "add_experience",
        {"type": "lesson", "content": "test"}
    )

    assert result.startswith("id_")

    # Test benchmark_memory_system
    operations = [
        {
            "name": "simple_add",
            "type": "add_experience",
            "args": [{"type": "lesson", "content": "benchmark test"}]
        }
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = Path(temp_dir) / "benchmark_stats.txt"

        report = benchmark_memory_system(
            memory_system, operations, iterations=1,
            export_path=export_path
        )

        assert report.total_operations == 1
        assert export_path.exists()


def test_profile_result_to_dict():
    """Test ProfileResult serialization."""
    result = ProfileResult(
        operation_name="test_op",
        start_time="2024-01-01T00:00:00Z",
        end_time="2024-01-01T00:00:01Z",
        duration_seconds=1.5,
        memory_usage_mb=50.0,
        cpu_usage_percent=25.0,
        metadata={"key": "value"}
    )

    data = result.to_dict()

    assert data["operation_name"] == "test_op"
    assert data["duration_seconds"] == 1.5
    assert data["memory_usage_mb"] == 50.0
    assert data["cpu_usage_percent"] == 25.0
    assert data["metadata"]["key"] == "value"


def test_performance_report_to_dict():
    """Test PerformanceReport serialization."""
    report = PerformanceReport(
        report_id="test_report",
        timestamp="2024-01-01T00:00:00Z",
        total_operations=10,
        total_duration=5.0,
        average_operation_time=0.5,
        operation_breakdown={"op1": {"count": 5, "avg_time": 0.3}},
        bottlenecks=["op1 is slow"],
        recommendations=["Optimize op1"]
    )

    data = report.to_dict()

    assert data["report_id"] == "test_report"
    assert data["total_operations"] == 10
    assert data["total_duration"] == 5.0
    assert len(data["bottlenecks"]) == 1
    assert len(data["recommendations"]) == 1
