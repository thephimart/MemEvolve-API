import sys

sys.path.insert(0, 'src')

from components.encode import (
    EncodingMetrics,
    EncodingMetricsCollector
)
import pytest


def test_encoding_metrics_initialization():
    """Test encoding metrics initialization."""
    metrics = EncodingMetrics()

    assert metrics.total_encodings == 0
    assert metrics.successful_encodings == 0
    assert metrics.failed_encodings == 0
    assert metrics.total_encoding_time == 0.0
    assert metrics.average_encoding_time == 0.0
    assert metrics.success_rate == 0.0
    assert metrics.type_distribution == {}
    assert metrics.error_types == {}


def test_encoding_metrics_calculate_success_rate():
    """Test success rate calculation."""
    metrics = EncodingMetrics(
        total_encodings=10,
        successful_encodings=8,
        failed_encodings=2
    )

    success_rate = metrics.calculate_success_rate()
    assert success_rate == 80.0


def test_encoding_metrics_calculate_average_time():
    """Test average time calculation."""
    metrics = EncodingMetrics(
        successful_encodings=5,
        total_encoding_time=2.5
    )

    avg_time = metrics.calculate_average_time()
    assert avg_time == 0.5


def test_encoding_metrics_calculate_with_zero_encodings():
    """Test calculations with zero encodings."""
    metrics = EncodingMetrics()

    assert metrics.calculate_success_rate() == 0.0
    assert metrics.calculate_average_time() == 0.0


def test_metrics_collector_initialization():
    """Test metrics collector initialization."""
    collector = EncodingMetricsCollector()

    assert collector.metrics.total_encodings == 0
    assert len(collector._encoding_history) == 0


def test_metrics_collector_start_encoding():
    """Test starting an encoding operation."""
    collector = EncodingMetricsCollector()

    operation_id = collector.start_encoding("exp_001")

    assert operation_id.startswith("encoding_exp_001_")


def test_metrics_collector_end_successful_encoding():
    """Test ending a successful encoding operation."""
    collector = EncodingMetricsCollector()
    operation_id = collector.start_encoding("exp_001")

    collector.end_encoding(
        operation_id=operation_id,
        experience_id="exp_001",
        success=True,
        encoded_unit={"type": "lesson", "content": "test"},
        duration=0.5
    )

    assert collector.metrics.total_encodings == 1
    assert collector.metrics.successful_encodings == 1
    assert collector.metrics.failed_encodings == 0
    assert collector.metrics.total_encoding_time == 0.5
    assert collector.metrics.success_rate == 100.0
    assert collector.metrics.average_encoding_time == 0.5
    assert collector.metrics.type_distribution.get("lesson") == 1
    assert len(collector._encoding_history) == 1


def test_metrics_collector_end_failed_encoding():
    """Test ending a failed encoding operation."""
    collector = EncodingMetricsCollector()
    operation_id = collector.start_encoding("exp_002")

    collector.end_encoding(
        operation_id=operation_id,
        experience_id="exp_002",
        success=False,
        error="LLM error occurred",
        duration=0.3
    )

    assert collector.metrics.total_encodings == 1
    assert collector.metrics.successful_encodings == 0
    assert collector.metrics.failed_encodings == 1
    assert collector.metrics.total_encoding_time == 0.0
    assert collector.metrics.success_rate == 0.0
    assert "llm_error" in collector.metrics.error_types


def test_metrics_collector_multiple_encodings():
    """Test tracking multiple encoding operations."""
    collector = EncodingMetricsCollector()

    for i in range(10):
        operation_id = collector.start_encoding(f"exp_{i}")
        success = i < 8
        if success:
            collector.end_encoding(
                operation_id=operation_id,
                experience_id=f"exp_{i}",
                success=True,
                encoded_unit={"type": "lesson", "content": f"content_{i}"},
                duration=0.1
            )
        else:
            collector.end_encoding(
                operation_id=operation_id,
                experience_id=f"exp_{i}",
                success=False,
                error="error",
                duration=0.1
            )

    assert collector.metrics.total_encodings == 10
    assert collector.metrics.successful_encodings == 8
    assert collector.metrics.failed_encodings == 2
    assert collector.metrics.success_rate == 80.0


def test_metrics_collector_type_distribution():
    """Test type distribution tracking."""
    collector = EncodingMetricsCollector()

    types = ["lesson", "skill", "abstraction", "lesson"]
    for i, unit_type in enumerate(types):
        operation_id = collector.start_encoding(f"exp_{i}")
        collector.end_encoding(
            operation_id=operation_id,
            experience_id=f"exp_{i}",
            success=True,
            encoded_unit={"type": unit_type, "content": f"content_{i}"},
            duration=0.1
        )

    assert collector.metrics.type_distribution["lesson"] == 2
    assert collector.metrics.type_distribution["skill"] == 1
    assert collector.metrics.type_distribution["abstraction"] == 1


def test_metrics_collector_error_categorization():
    """Test error categorization."""
    collector = EncodingMetricsCollector()

    errors = [
        "LLM timeout error",
        "JSON parse error",
        "Network connection failed",
        "Empty response from LLM",
        "Unknown error occurred"
    ]

    for i, error in enumerate(errors):
        operation_id = collector.start_encoding(f"exp_{i}")
        collector.end_encoding(
            operation_id=operation_id,
            experience_id=f"exp_{i}",
            success=False,
            error=error,
            duration=0.1
        )

    assert collector.metrics.error_types.get("timeout", 0) == 1
    assert collector.metrics.error_types.get("parse_error", 0) == 1
    assert collector.metrics.error_types.get("network_error", 0) == 1
    assert collector.metrics.error_types.get("empty_response", 0) == 1
    assert collector.metrics.error_types.get("unknown", 0) == 1


def test_metrics_collector_get_metrics():
    """Test getting metrics."""
    collector = EncodingMetricsCollector()
    operation_id = collector.start_encoding("exp_001")

    collector.end_encoding(
        operation_id=operation_id,
        experience_id="exp_001",
        success=True,
        encoded_unit={"type": "lesson", "content": "test"},
        duration=0.5
    )

    metrics = collector.get_metrics()

    assert metrics.total_encodings == 1
    assert metrics.successful_encodings == 1


def test_metrics_collector_get_encoding_history():
    """Test getting encoding history."""
    collector = EncodingMetricsCollector()

    for i in range(3):
        operation_id = collector.start_encoding(f"exp_{i}")
        collector.end_encoding(
            operation_id=operation_id,
            experience_id=f"exp_{i}",
            success=True,
            encoded_unit={"type": "lesson", "content": f"content_{i}"},
            duration=0.1
        )

    history = collector.get_encoding_history()

    assert len(history) == 3
    assert all("operation_id" in entry for entry in history)
    assert all("experience_id" in entry for entry in history)
    assert all("success" in entry for entry in history)


def test_metrics_collector_clear_history():
    """Test clearing encoding history."""
    collector = EncodingMetricsCollector()
    operation_id = collector.start_encoding("exp_001")

    collector.end_encoding(
        operation_id=operation_id,
        experience_id="exp_001",
        success=True,
        encoded_unit={"type": "lesson", "content": "test"},
        duration=0.5
    )

    assert len(collector._encoding_history) == 1

    collector.clear_history()

    assert len(collector._encoding_history) == 0


def test_metrics_collector_reset_metrics():
    """Test resetting all metrics."""
    collector = EncodingMetricsCollector()
    operation_id = collector.start_encoding("exp_001")

    collector.end_encoding(
        operation_id=operation_id,
        experience_id="exp_001",
        success=True,
        encoded_unit={"type": "lesson", "content": "test"},
        duration=0.5
    )

    assert collector.metrics.total_encodings == 1

    collector.reset_metrics()

    assert collector.metrics.total_encodings == 0
    assert len(collector._encoding_history) == 0


def test_metrics_collector_get_summary():
    """Test getting metrics summary."""
    collector = EncodingMetricsCollector()

    for i in range(10):
        operation_id = collector.start_encoding(f"exp_{i}")
        success = i < 8
        if success:
            collector.end_encoding(
                operation_id=operation_id,
                experience_id=f"exp_{i}",
                success=True,
                encoded_unit={"type": "lesson", "content": f"content_{i}"},
                duration=0.1
            )
        else:
            collector.end_encoding(
                operation_id=operation_id,
                experience_id=f"exp_{i}",
                success=False,
                error="error",
                duration=0.1
            )

    summary = collector.get_summary()

    assert summary["total_encodings"] == 10
    assert summary["successful_encodings"] == 8
    assert summary["failed_encodings"] == 2
    assert summary["success_rate"] == "80.00%"
    assert summary["average_encoding_time"] == "0.1000s"
    assert summary["last_encoding_status"] == "failed"
