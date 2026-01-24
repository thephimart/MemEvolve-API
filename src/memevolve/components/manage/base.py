from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class HealthMetrics:
    """Data class for memory health metrics."""
    total_units: int
    total_size_bytes: int
    average_unit_size: float
    oldest_unit_timestamp: Optional[str]
    newest_unit_timestamp: Optional[str]
    unit_types_distribution: Dict[str, int]
    duplicate_count: int
    last_operation: str
    last_operation_time: str


class ManagementStrategy(ABC):
    """Abstract base class for memory management strategies."""

    @abstractmethod
    def prune(
        self,
        storage_backend,
        criteria: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, List[str]]:
        """Prune memory units based on criteria.

        Returns:
            Tuple of (count of pruned units, list of pruned unit IDs)
        """
        pass

    @abstractmethod
    def consolidate(
        self,
        storage_backend,
        units: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Consolidate memory units.

        Returns:
            List of consolidated units
        """
        pass

    @abstractmethod
    def deduplicate(
        self,
        storage_backend,
        similarity_threshold: float = 0.9
    ) -> Tuple[int, List[str]]:
        """Deduplicate memory units.

        Returns:
            Tuple of (count of removed duplicates, list of removed unit IDs)
        """
        pass

    @abstractmethod
    def apply_forgetting(
        self,
        storage_backend,
        strategy: str = "lru",
        count: Optional[int] = None
    ) -> Tuple[int, List[str]]:
        """Apply forgetting mechanism to memory.

        Args:
            strategy: Forgetting strategy ("lru", "lfu", "random")
            count: Number of units to forget (None for auto-determine)

        Returns:
            Tuple of (count of forgotten units, list of forgotten unit IDs)
        """
        pass

    @abstractmethod
    def get_health_metrics(
        self,
        storage_backend
    ) -> HealthMetrics:
        """Get health metrics for memory system."""
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


class MemoryManager:
    """Main memory manager coordinating management strategies."""

    def __init__(
        self,
        storage_backend,
        management_strategy: ManagementStrategy
    ):
        self.storage_backend = storage_backend
        self.management_strategy = management_strategy
        self.operation_history: List[Dict[str, Any]] = []

    def set_strategy(self, strategy: ManagementStrategy):
        """Change management strategy."""
        self.management_strategy = strategy

    def prune(
        self,
        criteria: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, List[str]]:
        """Prune memory units using current strategy."""
        count, unit_ids = self.management_strategy.prune(
            self.storage_backend,
            criteria
        )

        self._log_operation("prune", {"count": count, "unit_ids": unit_ids})
        return count, unit_ids

    def consolidate(
        self,
        units: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Consolidate memory units using current strategy."""
        consolidated = self.management_strategy.consolidate(
            self.storage_backend,
            units
        )

        self._log_operation("consolidate", {"count": len(consolidated)})
        return consolidated

    def deduplicate(
        self,
        similarity_threshold: float = 0.9
    ) -> Tuple[int, List[str]]:
        """Deduplicate memory units using current strategy."""
        count, unit_ids = self.management_strategy.deduplicate(
            self.storage_backend,
            similarity_threshold
        )

        self._log_operation(
            "deduplicate",
            {"count": count, "threshold": similarity_threshold}
        )
        return count, unit_ids

    def apply_forgetting(
        self,
        strategy: str = "lru",
        count: Optional[int] = None
    ) -> Tuple[int, List[str]]:
        """Apply forgetting using current strategy."""
        count, unit_ids = self.management_strategy.apply_forgetting(
            self.storage_backend,
            strategy,
            count
        )

        self._log_operation(
            "forget",
            {"strategy": strategy, "count": count}
        )
        return count, unit_ids

    def get_health_metrics(self) -> HealthMetrics:
        """Get health metrics using current strategy."""
        return self.management_strategy.get_health_metrics(
            self.storage_backend
        )

    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get operation history."""
        return self.operation_history

    def clear_history(self):
        """Clear operation history."""
        self.operation_history.clear()

    def _log_operation(self, operation: str, details: Dict[str, Any]):
        """Log management operation."""
        self.operation_history.append({
            "operation": operation,
            "details": details,
            "timestamp": self._get_timestamp()
        })

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.now(timezone.utc).isoformat() + "Z"
