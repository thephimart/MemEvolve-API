from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval operations."""

    total_retrievals: int = 0
    successful_retrievals: int = 0
    failed_retrievals: int = 0
    total_retrieval_time: float = 0.0
    average_retrieval_time: float = 0.0
    total_results_retrieved: int = 0
    average_results_per_retrieval: float = 0.0
    strategy_distribution: Dict[str, int] = field(default_factory=dict)
    query_length_distribution: Dict[str, int] = field(default_factory=dict)
    top_k_distribution: Dict[str, int] = field(default_factory=dict)
    last_retrieval_time: Optional[str] = None
    last_retrieval_status: str = "none"
    last_query: Optional[str] = None

    def calculate_success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_retrievals == 0:
            return 0.0
        return (self.successful_retrievals / self.total_retrievals) * 100

    def calculate_average_time(self) -> float:
        """Calculate average retrieval time in seconds."""
        if self.successful_retrievals == 0:
            return 0.0
        return self.total_retrieval_time / self.successful_retrievals

    def calculate_average_results(self) -> float:
        """Calculate average results per retrieval."""
        if self.successful_retrievals == 0:
            return 0.0
        return self.total_results_retrieved / self.successful_retrievals


class RetrievalMetricsCollector:
    """Collector for retrieval metrics."""

    def __init__(self):
        self.metrics = RetrievalMetrics()
        self._retrieval_history: List[Dict[str, Any]] = []

    def start_retrieval(self, query: str) -> str:
        """Start tracking a retrieval operation.

        Returns:
            Operation ID for tracking
        """
        operation_id = f"retrieval_{int(time.time() * 1000)}"
        return operation_id

    def end_retrieval(
        self,
        operation_id: str,
        query: str,
        strategy_name: str,
        success: bool,
        results_count: int,
        top_k: int,
        error: Optional[str] = None,
        duration: float = 0.0
    ):
        """End tracking a retrieval operation."""
        self.metrics.total_retrievals += 1
        self.metrics.last_retrieval_time = self._get_timestamp()
        self.metrics.last_query = query

        if success:
            self.metrics.successful_retrievals += 1
            self.metrics.total_retrieval_time += duration
            self.metrics.total_results_retrieved += results_count
            self.metrics.last_retrieval_status = "success"

            strategy_name_lower = strategy_name.lower()
            self.metrics.strategy_distribution[strategy_name_lower] = (
                self.metrics.strategy_distribution.get(
                    strategy_name_lower, 0) + 1
            )

            query_length_key = self._categorize_query_length(query)
            self.metrics.query_length_distribution[query_length_key] = (
                self.metrics.query_length_distribution.get(
                    query_length_key, 0
                ) + 1
            )

            top_k_key = str(top_k)
            self.metrics.top_k_distribution[top_k_key] = (
                self.metrics.top_k_distribution.get(top_k_key, 0) + 1
            )
        else:
            self.metrics.failed_retrievals += 1
            self.metrics.last_retrieval_status = "failed"

        self.metrics.average_retrieval_time = (
            self.metrics.calculate_average_time()
        )
        self.metrics.average_results_per_retrieval = (
            self.metrics.calculate_average_results()
        )

        self._retrieval_history.append({
            "operation_id": operation_id,
            "query": query,
            "strategy": strategy_name,
            "success": success,
            "results_count": results_count,
            "top_k": top_k,
            "error": error,
            "duration": duration,
            "timestamp": self._get_timestamp()
        })

    def get_metrics(self) -> RetrievalMetrics:
        """Get current metrics."""
        return self.metrics

    def get_retrieval_history(self) -> List[Dict[str, Any]]:
        """Get retrieval history."""
        return self._retrieval_history.copy()

    def clear_history(self):
        """Clear retrieval history."""
        self._retrieval_history.clear()

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = RetrievalMetrics()
        self._retrieval_history.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of retrieval metrics."""
        return {
            "total_retrievals": self.metrics.total_retrievals,
            "successful_retrievals": self.metrics.successful_retrievals,
            "failed_retrievals": self.metrics.failed_retrievals,
            "success_rate": f"{self.metrics.calculate_success_rate():.2f}%",
            "average_retrieval_time":
                f"{self.metrics.average_retrieval_time:.4f}s",
            "average_results_per_retrieval":
                f"{self.metrics.average_results_per_retrieval:.2f}",
            "strategy_distribution": self.metrics.strategy_distribution,
            "query_length_distribution":
                self.metrics.query_length_distribution,
            "top_k_distribution": self.metrics.top_k_distribution,
            "last_retrieval_status": self.metrics.last_retrieval_status,
            "last_query": self.metrics.last_query,
            "last_retrieval_time": self.metrics.last_retrieval_time
        }

    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.now(timezone.utc).isoformat() + "Z"

    def _categorize_query_length(self, query: str) -> str:
        """Categorize query by length."""
        length = len(query)
        if length <= 10:
            return "short"
        elif length <= 30:
            return "medium"
        elif length <= 60:
            return "long"
        else:
            return "very_long"
