from components.manage.base import HealthMetrics
from components.retrieve.metrics import RetrievalMetrics
from components.encode.metrics import EncodingMetrics
from utils.metrics import (
    SystemMetrics,
    MetricsCollector
)
from pathlib import Path
import csv
import json
import tempfile
import sys

sys.path.insert(0, 'src')


class MockMemorySystem:
    """Mock memory system for testing metrics collection."""

    def __init__(self):
        self.encoder = MockEncoder()
        self.retriever = MockRetriever()

    def get_health_metrics(self):
        return HealthMetrics(
            total_units=100,
            total_size_bytes=1024000,
            average_unit_size=10240.0,
            oldest_unit_timestamp="2024-01-01T00:00:00Z",
            newest_unit_timestamp="2024-01-19T12:00:00Z",
            unit_types_distribution={"lesson": 60, "skill": 30, "tool": 10},
            duplicate_count=5,
            last_operation="consolidate",
            last_operation_time="2024-01-19T12:00:00Z"
        )


class MockEncoder:
    """Mock encoder for testing."""

    def get_metrics(self):
        return EncodingMetrics(
            total_encodings=50,
            successful_encodings=45,
            failed_encodings=5,
            total_encoding_time=25.0,
            average_encoding_time=0.5,
            success_rate=90.0,
            type_distribution={"lesson": 30, "skill": 15, "tool": 5},
            error_types={"parse_error": 3, "validation_error": 2},
            last_encoding_time="2024-01-19T11:00:00Z",
            last_encoding_status="success"
        )


class MockRetriever:
    """Mock retriever for testing."""

    def get_metrics(self):
        return RetrievalMetrics(
            total_retrievals=200,
            successful_retrievals=180,
            failed_retrievals=20,
            total_retrieval_time=50.0,
            average_retrieval_time=0.25,
            total_results_retrieved=540,
            average_results_per_retrieval=2.7,
            strategy_distribution={"semantic": 120,
                                   "keyword": 60, "hybrid": 20},
            query_length_distribution={"medium": 100, "long": 80, "short": 20},
            top_k_distribution={"5": 80, "10": 100, "20": 20},
            last_retrieval_time="2024-01-19T12:00:00Z",
            last_retrieval_status="success",
            last_query="test query"
        )


def test_system_metrics_initialization():
    """Test system metrics initialization."""
    metrics = SystemMetrics()

    assert metrics.encoding is None
    assert metrics.retrieval is None
    assert metrics.health is None
    assert metrics.total_operations == 0
    assert metrics.successful_operations == 0
    assert metrics.failed_operations == 0
    assert metrics.average_operation_time == 0.0
    assert isinstance(metrics.timestamp, str)


def test_system_metrics_calculate_overall_success_rate():
    """Test overall success rate calculation."""
    metrics = SystemMetrics(
        total_operations=100,
        successful_operations=85,
        failed_operations=15
    )

    assert metrics.calculate_overall_success_rate() == 85.0

    # Test with zero operations
    empty_metrics = SystemMetrics()
    assert empty_metrics.calculate_overall_success_rate() == 0.0


def test_system_metrics_get_performance_summary():
    """Test performance summary generation."""
    metrics = SystemMetrics(
        total_operations=100,
        successful_operations=85,
        failed_operations=15,
        average_operation_time=1.5
    )

    summary = metrics.get_performance_summary()

    assert "overall_success_rate" in summary
    assert "total_operations" in summary
    assert summary["overall_success_rate"] == "85.00%"
    assert summary["average_operation_time"] == "1.5000s"


def test_metrics_collector_initialization():
    """Test metrics collector initialization."""
    collector = MetricsCollector()

    assert len(collector.metrics_history) == 0
    assert isinstance(collector.current_metrics, SystemMetrics)


def test_metrics_collector_collect_from_memory_system():
    """Test collecting metrics from memory system."""
    collector = MetricsCollector()
    memory_system = MockMemorySystem()

    metrics = collector.collect_from_memory_system(memory_system)

    assert metrics.encoding is not None
    assert metrics.retrieval is not None
    assert metrics.health is not None
    assert metrics.total_operations == 250  # 50 encoding + 200 retrieval
    assert metrics.successful_operations == 225  # 45 + 180
    assert metrics.failed_operations == 25  # 5 + 20


def test_metrics_collector_snapshot():
    """Test taking metrics snapshots."""
    collector = MetricsCollector()
    memory_system = MockMemorySystem()

    collector.collect_from_memory_system(memory_system)
    snapshot = collector.snapshot()

    assert len(collector.metrics_history) == 1
    assert snapshot.total_operations == 250

    # Verify snapshot is independent copy
    collector.current_metrics.total_operations = 999
    assert snapshot.total_operations == 250


def test_metrics_collector_analyze_trends():
    """Test trend analysis."""
    collector = MetricsCollector()

    # Create some mock history with increasing success rates
    for i in range(5):
        metrics = SystemMetrics(
            total_operations=100,
            successful_operations=80 + i * 2,  # 80, 82, 84, 86, 88
            failed_operations=20 - i * 2,      # 20, 18, 16, 14, 12
            average_operation_time=1.0 + i * 0.1
        )
        collector.metrics_history.append(metrics)

    trends = collector.analyze_trends(window_size=3)

    assert "success_rate_trend" in trends
    assert "operation_time_trend" in trends
    assert trends["success_rate_trend"]["direction"] == "increasing"
    assert trends["operation_time_trend"]["direction"] == "increasing"


def test_metrics_collector_analyze_trends_insufficient_data():
    """Test trend analysis with insufficient data."""
    collector = MetricsCollector()

    # Only one data point
    metrics = SystemMetrics(total_operations=100, successful_operations=85)
    collector.metrics_history.append(metrics)

    trends = collector.analyze_trends()

    assert "error" in trends
    assert trends["error"] == "Insufficient data for trend analysis"


def test_metrics_collector_export_to_json():
    """Test JSON export functionality."""
    collector = MetricsCollector()
    memory_system = MockMemorySystem()

    collector.collect_from_memory_system(memory_system)
    collector.snapshot()

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "test_metrics.json"

        success = collector.export_to_json(filepath)
        assert success
        assert filepath.exists()

        # Verify JSON content
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert len(data) == 1
        assert "total_operations" in data[0]
        assert data[0]["total_operations"] == 250


def test_metrics_collector_export_to_csv():
    """Test CSV export functionality."""
    collector = MetricsCollector()
    memory_system = MockMemorySystem()

    collector.collect_from_memory_system(memory_system)
    collector.snapshot()

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "test_metrics.csv"

        success = collector.export_to_csv(filepath)
        assert success
        assert filepath.exists()

        # Verify CSV content
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 2  # Header + 1 data row
        assert "total_operations" in rows[0]
        assert rows[1][1] == "250"  # total_operations value


def test_metrics_collector_reset():
    """Test metrics collector reset."""
    collector = MetricsCollector()
    memory_system = MockMemorySystem()

    collector.collect_from_memory_system(memory_system)
    collector.snapshot()

    assert len(collector.metrics_history) == 1
    assert collector.current_metrics.total_operations > 0

    collector.reset()

    assert len(collector.metrics_history) == 0
    assert collector.current_metrics.total_operations == 0


def test_metrics_collector_get_summary_report():
    """Test summary report generation."""
    collector = MetricsCollector()
    memory_system = MockMemorySystem()

    collector.collect_from_memory_system(memory_system)
    collector.snapshot()

    report = collector.get_summary_report()

    assert "current_metrics" in report
    assert "history_length" in report
    assert "analysis" in report
    assert "historical_stats" in report

    assert report["history_length"] == 1
    assert "avg_success_rate" in report["historical_stats"]


def test_metrics_collector_empty_history_summary():
    """Test summary report with empty history."""
    collector = MetricsCollector()

    report = collector.get_summary_report()

    assert "current_metrics" in report
    assert report["history_length"] == 0
    assert report["analysis"] is None
    assert "historical_stats" not in report
