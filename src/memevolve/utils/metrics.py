"""
Unified metrics collection and analysis system for MemEvolve.

This module provides tools for collecting, analyzing, and exporting metrics
across all memory system components (Encode, Store, Retrieve, Manage).
"""

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..components.encode.metrics import EncodingMetrics
    from ..components.manage.base import HealthMetrics
    from ..components.retrieve.metrics import RetrievalMetrics

from .logging_manager import LoggingManager

logger = LoggingManager.get_logger(__name__)

try:
    from ..components.encode.metrics import EncodingMetrics
    from ..components.manage.base import HealthMetrics
    from ..components.retrieve.metrics import RetrievalMetrics
except ImportError:
    # Handle case where components might not be available during testing
    EncodingMetrics = None
    RetrievalMetrics = None
    HealthMetrics = None


@dataclass
class SystemMetrics:
    """Aggregated metrics across all memory system components."""

    timestamp: str = field(default_factory=lambda: datetime.now(
        timezone.utc).isoformat() + "Z")

    # Component metrics
    encoding: Optional[Any] = None
    retrieval: Optional[Any] = None
    health: Optional[Any] = None

    # System-level aggregations
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_operation_time: float = 0.0

    def calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across components."""
        if self.total_operations == 0:
            return 0.0
        return (self.successful_operations / self.total_operations) * 100

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all components."""
        summary = {
            "timestamp": self.timestamp,
            "overall_success_rate":
                f"{self.calculate_overall_success_rate():.2f}%",
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "average_operation_time":
                f"{self.average_operation_time:.4f}s"
        }

        # Add component-specific summaries
        if self.encoding:
            summary["encoding"] = {
                "total_encodings": self.encoding.total_encodings,
                "success_rate": f"{self.encoding.success_rate:.2f}%",
                "average_time": f"{self.encoding.average_encoding_time:.4f}s"
            }

        if self.retrieval:
            summary["retrieval"] = {
                "total_retrievals": self.retrieval.total_retrievals,
                "success_rate":
                    f"{self.retrieval.calculate_success_rate():.2f}%",
                "average_time":
                    f"{self.retrieval.average_retrieval_time:.4f}s"
            }

        if self.health:
            summary["health"] = {
                "total_units": self.health.total_units,
                "total_size_bytes": self.health.total_size_bytes,
                "duplicate_count": self.health.duplicate_count
            }

        return summary


class MetricsCollector:
    """Collector for unified memory system metrics."""

    def __init__(self):
        self.logger = LoggingManager.get_logger("memevolve.utils.metrics.metrics_collector")
        self.metrics_history: List[SystemMetrics] = []
        self.current_metrics = SystemMetrics()

    def collect_from_memory_system(self, memory_system) -> SystemMetrics:
        """Collect metrics from a memory system instance.

        Args:
            memory_system: MemorySystem instance with metrics methods

        Returns:
            Updated SystemMetrics
        """
        try:
            # Collect encoding metrics
            if hasattr(memory_system, 'encoder') and hasattr(memory_system.encoder, 'get_metrics'):
                self.current_metrics.encoding = memory_system.encoder.get_metrics()

            # Collect retrieval metrics
            if (hasattr(memory_system, 'retriever') and
                    hasattr(memory_system.retriever, 'get_metrics')):
                self.current_metrics.retrieval = memory_system.retriever.get_metrics()

            # Collect health metrics
            if hasattr(memory_system, 'get_health_metrics'):
                self.current_metrics.health = memory_system.get_health_metrics()

            # Calculate system-level aggregations
            self._calculate_system_aggregations()

            return self.current_metrics

        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {str(e)}")
            return self.current_metrics

    def _calculate_system_aggregations(self):
        """Calculate system-level aggregated metrics."""
        total_ops = 0
        successful_ops = 0
        failed_ops = 0
        total_time = 0.0

        # Aggregate from encoding
        if self.current_metrics.encoding:
            total_ops += self.current_metrics.encoding.total_encodings
            successful_ops += self.current_metrics.encoding.successful_encodings
            failed_ops += self.current_metrics.encoding.failed_encodings
            total_time += self.current_metrics.encoding.total_encoding_time

        # Aggregate from retrieval
        if self.current_metrics.retrieval:
            total_ops += self.current_metrics.retrieval.total_retrievals
            successful_ops += self.current_metrics.retrieval.successful_retrievals
            failed_ops += self.current_metrics.retrieval.failed_retrievals
            total_time += self.current_metrics.retrieval.total_retrieval_time

        self.current_metrics.total_operations = total_ops
        self.current_metrics.successful_operations = successful_ops
        self.current_metrics.failed_operations = failed_ops

        if total_ops > 0:
            self.current_metrics.average_operation_time = total_time / total_ops
        else:
            self.current_metrics.average_operation_time = 0.0

    def snapshot(self) -> SystemMetrics:
        """Take a snapshot of current metrics and store in history.

        Returns:
            Current metrics snapshot
        """
        snapshot = SystemMetrics(
            timestamp=self.current_metrics.timestamp,
            encoding=self.current_metrics.encoding,
            retrieval=self.current_metrics.retrieval,
            health=self.current_metrics.health,
            total_operations=self.current_metrics.total_operations,
            successful_operations=self.current_metrics.successful_operations,
            failed_operations=self.current_metrics.failed_operations,
            average_operation_time=self.current_metrics.average_operation_time
        )

        self.metrics_history.append(snapshot)
        return snapshot

    def get_current_metrics(self) -> SystemMetrics:
        """Get current metrics."""
        return self.current_metrics

    def get_metrics_history(self) -> List[SystemMetrics]:
        """Get metrics history."""
        return self.metrics_history.copy()

    def analyze_trends(self, window_size: int = 5) -> Dict[str, Any]:
        """Analyze trends in metrics over recent history.

        Args:
            window_size: Number of recent snapshots to analyze

        Returns:
            Dictionary with trend analysis
        """
        if len(self.metrics_history) < 2:
            return {"error": "Insufficient data for trend analysis"}

        recent = (self.metrics_history[-window_size:]
                  if len(self.metrics_history) >= window_size
                  else self.metrics_history)

        trends = {
            "analysis_window": len(recent),
            "time_range": f"{recent[0].timestamp} to {recent[-1].timestamp}"
        }

        # Analyze success rate trends
        success_rates = [m.calculate_overall_success_rate() for m in recent]
        trends["success_rate_trend"] = self._calculate_trend(success_rates)

        # Analyze operation time trends
        operation_times = [m.average_operation_time for m in recent]
        trends["operation_time_trend"] = self._calculate_trend(operation_times)

        return trends

    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend for a series of values."""
        if len(values) < 2:
            return {"direction": "insufficient_data", "change": 0.0}

        first = values[0]
        last = values[-1]
        change = last - first
        change_percent = (change / first * 100) if first != 0 else 0

        if change > 0:
            direction = "increasing"
        elif change < 0:
            direction = "decreasing"
        else:
            direction = "stable"

        return {
            "direction": direction,
            "absolute_change": change,
            "percentage_change": change_percent,
            "start_value": first,
            "end_value": last
        }

    def export_to_json(self, filepath: Union[str, Path]) -> bool:
        """Export metrics history to JSON file.

        Args:
            filepath: Path to export file

        Returns:
            True if export successful, False otherwise
        """
        try:
            filepath = Path(filepath)

            # Convert dataclasses to dicts for JSON serialization
            history_data = []
            for metrics in self.metrics_history:
                metrics_dict = {
                    "timestamp": metrics.timestamp,
                    "total_operations": metrics.total_operations,
                    "successful_operations": metrics.successful_operations,
                    "failed_operations": metrics.failed_operations,
                    "average_operation_time": metrics.average_operation_time
                }

                if metrics.encoding:
                    metrics_dict["encoding"] = {
                        "total_encodings": metrics.encoding.total_encodings,
                        "successful_encodings": metrics.encoding.successful_encodings,
                        "failed_encodings": metrics.encoding.failed_encodings,
                        "success_rate": metrics.encoding.success_rate,
                        "average_encoding_time": metrics.encoding.average_encoding_time
                    }

                if metrics.retrieval:
                    metrics_dict["retrieval"] = {
                        "total_retrievals": metrics.retrieval.total_retrievals,
                        "successful_retrievals": metrics.retrieval.successful_retrievals,
                        "failed_retrievals": metrics.retrieval.failed_retrievals,
                        "average_retrieval_time": metrics.retrieval.average_retrieval_time
                    }

                if metrics.health:
                    metrics_dict["health"] = {
                        "total_units": metrics.health.total_units,
                        "total_size_bytes": metrics.health.total_size_bytes,
                        "duplicate_count": metrics.health.duplicate_count
                    }

                history_data.append(metrics_dict)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Metrics exported to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export metrics: {str(e)}")
            return False

    def export_to_csv(self, filepath: Union[str, Path]) -> bool:
        """Export metrics history to CSV file.

        Args:
            filepath: Path to export file

        Returns:
            True if export successful, False otherwise
        """
        try:
            filepath = Path(filepath)

            if not self.metrics_history:
                self.logger.warning("No metrics history to export")
                return False

            # Prepare CSV data
            csv_data = []
            headers = [
                "timestamp", "total_operations", "successful_operations",
                "failed_operations", "average_operation_time"
            ]

            # Add component-specific headers
            if any(m.encoding for m in self.metrics_history):
                headers.extend([
                    "encoding_total", "encoding_successful", "encoding_failed",
                    "encoding_success_rate", "encoding_avg_time"
                ])

            if any(m.retrieval for m in self.metrics_history):
                headers.extend([
                    "retrieval_total", "retrieval_successful", "retrieval_failed",
                    "retrieval_avg_time"
                ])

            if any(m.health for m in self.metrics_history):
                headers.extend([
                    "health_total_units", "health_total_size", "health_duplicates"
                ])

            csv_data.append(headers)

            # Add data rows
            for metrics in self.metrics_history:
                row = [
                    metrics.timestamp,
                    metrics.total_operations,
                    metrics.successful_operations,
                    metrics.failed_operations,
                    metrics.average_operation_time
                ]

                # Add encoding data
                if metrics.encoding:
                    row.extend([
                        metrics.encoding.total_encodings,
                        metrics.encoding.successful_encodings,
                        metrics.encoding.failed_encodings,
                        metrics.encoding.success_rate,
                        metrics.encoding.average_encoding_time
                    ])
                elif any(m.encoding for m in self.metrics_history):
                    row.extend([0, 0, 0, 0.0, 0.0])

                # Add retrieval data
                if metrics.retrieval:
                    row.extend([
                        metrics.retrieval.total_retrievals,
                        metrics.retrieval.successful_retrievals,
                        metrics.retrieval.failed_retrievals,
                        metrics.retrieval.average_retrieval_time
                    ])
                elif any(m.retrieval for m in self.metrics_history):
                    row.extend([0, 0, 0, 0.0])

                # Add health data
                if metrics.health:
                    row.extend([
                        metrics.health.total_units,
                        metrics.health.total_size_bytes,
                        metrics.health.duplicate_count
                    ])
                elif any(m.health for m in self.metrics_history):
                    row.extend([0, 0, 0])

                csv_data.append(row)

            # Write CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)

            self.logger.info(f"Metrics exported to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export metrics: {str(e)}")
            return False

    def reset(self):
        """Reset all metrics and history."""
        self.metrics_history.clear()
        self.current_metrics = SystemMetrics()
        self.logger.info("Metrics collector reset")

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report.

        Returns:
            Dictionary with summary statistics
        """
        report = {
            "current_metrics": self.current_metrics.get_performance_summary(),
            "history_length": len(self.metrics_history),
            "analysis": self.analyze_trends() if self.metrics_history else None
        }

        if self.metrics_history:
            # Calculate historical statistics
            success_rates = [m.calculate_overall_success_rate()
                             for m in self.metrics_history]
            operation_times = [
                m.average_operation_time for m in self.metrics_history]

            report["historical_stats"] = {
                "avg_success_rate": sum(success_rates) / len(success_rates),
                "avg_operation_time": sum(operation_times) / len(operation_times),
                "total_operations_all_time":
                    sum(m.total_operations for m in self.metrics_history),
                "total_successful_all_time":
                    sum(m.successful_operations for m in self.metrics_history)
            }

        return report
