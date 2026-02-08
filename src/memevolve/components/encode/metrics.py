import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class EncodingMetrics:
    """Metrics for encoding operations."""

    total_encodings: int = 0
    successful_encodings: int = 0
    failed_encodings: int = 0
    total_encoding_time: float = 0.0
    average_encoding_time: float = 0.0
    success_rate: float = 0.0
    type_distribution: Dict[str, int] = field(default_factory=dict)
    error_types: Dict[str, int] = field(default_factory=dict)
    last_encoding_time: Optional[str] = None
    last_encoding_status: str = "none"

    def calculate_success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_encodings == 0:
            return 0.0
        return (self.successful_encodings / self.total_encodings) * 100

    def calculate_average_time(self) -> float:
        """Calculate average encoding time in seconds."""
        if self.successful_encodings == 0:
            return 0.0
        return self.total_encoding_time / self.successful_encodings


class EncodingMetricsCollector:
    """Collector for encoding metrics."""

    def __init__(self):
        self.metrics = EncodingMetrics()
        self._encoding_history: List[Dict[str, Any]] = []

    def start_encoding(self, experience_id: str) -> str:
        """Start tracking an encoding operation.

        Returns:
            Operation ID for tracking
        """
        operation_id = f"encoding_{experience_id}_{int(time.time() * 1000)}"
        return operation_id

    def end_encoding(
        self,
        operation_id: str,
        experience_id: str,
        success: bool,
        encoded_unit: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        duration: float = 0.0
    ):
        """End tracking an encoding operation."""
        self.metrics.total_encodings += 1
        self.metrics.last_encoding_time = self._get_timestamp()

        if success:
            self.metrics.successful_encodings += 1
            self.metrics.total_encoding_time += duration
            self.metrics.last_encoding_status = "success"

            if encoded_unit:
                unit_type = encoded_unit.get("type", "unknown")
                self.metrics.type_distribution[unit_type] = (
                    self.metrics.type_distribution.get(unit_type, 0) + 1
                )
        else:
            self.metrics.failed_encodings += 1
            self.metrics.last_encoding_status = "failed"

            if error:
                error_key = self._categorize_error(error)
                self.metrics.error_types[error_key] = (
                    self.metrics.error_types.get(error_key, 0) + 1
                )

        self.metrics.success_rate = self.metrics.calculate_success_rate()
        self.metrics.average_encoding_time = (
            self.metrics.calculate_average_time()
        )

        self._encoding_history.append({
            "operation_id": operation_id,
            "experience_id": experience_id,
            "success": success,
            "duration": duration,
            "unit_type": encoded_unit.get("type") if encoded_unit else None,
            "error": error,
            "timestamp": self._get_timestamp()
        })

    def get_metrics(self) -> EncodingMetrics:
        """Get current metrics."""
        return self.metrics

    def get_encoding_history(self) -> List[Dict[str, Any]]:
        """Get encoding history."""
        return self._encoding_history.copy()

    def clear_history(self):
        """Clear encoding history."""
        self._encoding_history.clear()

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = EncodingMetrics()
        self._encoding_history.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of encoding metrics."""
        return {
            "total_encodings": self.metrics.total_encodings,
            "successful_encodings":
                self.metrics.successful_encodings,
            "failed_encodings": self.metrics.failed_encodings,
            "success_rate": f"{self.metrics.success_rate:.2f}%",
            "average_encoding_time":
                f"{self.metrics.average_encoding_time:.4f}s",
            "type_distribution": self.metrics.type_distribution,
            "error_types": self.metrics.error_types,
            "last_encoding_status": self.metrics.last_encoding_status,
            "last_encoding_time": self.metrics.last_encoding_time
        }

    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.now(timezone.utc).isoformat() + "Z"

    def _categorize_error(self, error: str) -> str:
        """Categorize error type."""
        error_lower = error.lower()

        if "empty" in error_lower:
            return "empty_response"
        elif "timeout" in error_lower:
            return "timeout"
        elif "json" in error_lower:
            return "parse_error"
        elif "network" in error_lower or "connection" in error_lower:
            return "network_error"
        elif "llm" in error_lower or "client" in error_lower:
            return "llm_error"
        else:
            return "unknown"
