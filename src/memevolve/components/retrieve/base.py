import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...utils.logging_manager import LoggingManager

logger = LoggingManager.get_logger(__name__)


@dataclass
class RetrievalResult:
    """Data class for retrieval results."""
    unit_id: str
    unit: Dict[str, Any]
    score: float
    metadata: Optional[Dict[str, Any]] = None


class RetrievalStrategy(ABC):
    """Abstract base class for all retrieval strategies."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        storage_backend,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve memory units based on a query."""
        pass

    @abstractmethod
    def retrieve_by_ids(
        self,
        unit_ids: List[str],
        storage_backend
    ) -> List[RetrievalResult]:
        """Retrieve specific memory units by their IDs."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        storage_backend,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Search for memory units matching the query."""
        pass

    @abstractmethod
    def count_relevant(
        self,
        query: str,
        storage_backend,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count memory units relevant to the query."""
        pass

    def save_state(self) -> Dict[str, Any]:
        """Save current component state for hot-swapping.

        Returns:
            Dictionary containing component state
        """
        return {}

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore component state after hot-swapping.

        Args:
            state: State dictionary from save_state()
        """
        pass


class RetrievalContext:
    """Context for retrieval operations, maintaining state and config."""

    def __init__(
        self,
        strategy: RetrievalStrategy,
        default_top_k: int = 5,
        default_filters: Optional[Dict[str, Any]] = None,
        enable_metrics: bool = True
    ):
        self.strategy = strategy
        self.default_top_k = default_top_k
        self.default_filters = default_filters or {}
        self.retrieval_history: List[Dict[str, Any]] = []
        self.enable_metrics = enable_metrics
        self._metrics_collector: Optional[Any] = None

        if enable_metrics:
            from .metrics import RetrievalMetricsCollector
            self._metrics_collector = RetrievalMetricsCollector()

    def set_strategy(self, strategy: RetrievalStrategy):
        """Change the retrieval strategy."""
        self.strategy = strategy

    def retrieve(
        self,
        query: str,
        storage_backend,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve using the current strategy."""
        if top_k is None:
            top_k = self.default_top_k
        if filters is None:
            filters = self.default_filters

        operation_id = None
        start_time = None

        if self.enable_metrics:
            operation_id = self._metrics_collector.start_retrieval(query)
            start_time = time.time()

        strategy_name = self.strategy.__class__.__name__

        try:
            results = self.strategy.retrieve(
                query, storage_backend, top_k, filters)

            if self.enable_metrics and operation_id and start_time:
                duration = time.time() - start_time
                self._metrics_collector.end_retrieval(
                    operation_id=operation_id,
                    query=query,
                    strategy_name=strategy_name,
                    success=True,
                    results_count=len(results),
                    top_k=top_k,
                    duration=duration
                )

            self.retrieval_history.append({
                "query": query,
                "top_k": top_k,
                "filters": filters,
                "results_count": len(results),
                "timestamp": self._get_timestamp()
            })

            return results
        except Exception as e:
            if self.enable_metrics and operation_id and start_time:
                duration = time.time() - start_time
                self._metrics_collector.end_retrieval(
                    operation_id=operation_id,
                    query=query,
                    strategy_name=strategy_name,
                    success=False,
                    results_count=0,
                    top_k=top_k,
                    error=str(e),
                    duration=duration
                )

            raise

    def retrieve_by_ids(
        self,
        unit_ids: List[str],
        storage_backend
    ) -> List[RetrievalResult]:
        """Retrieve specific units by ID."""
        return self.strategy.retrieve_by_ids(unit_ids, storage_backend)

    def search(
        self,
        query: str,
        storage_backend,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Search using the current strategy."""
        if top_k is None:
            top_k = self.default_top_k
        if filters is None:
            filters = self.default_filters

        return self.strategy.search(query, storage_backend, top_k, filters)

    def count_relevant(
        self,
        query: str,
        storage_backend,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count relevant units."""
        if filters is None:
            filters = self.default_filters

        return self.strategy.count_relevant(query, storage_backend, filters)

    def get_retrieval_history(self) -> List[Dict[str, Any]]:
        """Get retrieval history."""
        return self.retrieval_history

    def clear_history(self):
        """Clear retrieval history."""
        self.retrieval_history.clear()

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat() + "Z"
