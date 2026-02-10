import time
from typing import Any, Dict, List, Optional, Tuple

from .base import HealthMetrics, ManagementStrategy
from ...utils.logging_manager import LoggingManager

logger = LoggingManager.get_logger(__name__)
logger.info("Simple management strategy initialized")


class SimpleManagementStrategy(ManagementStrategy):
    """Simple memory management strategy with basic operations."""

    def prune(
        self,
        storage_backend,
        criteria: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, List[str]]:
        """Prune memory units based on criteria.

        Supported criteria:
            - max_age: Remove units older than max_age (in days)
            - max_count: Keep only the most recent max_count units
            - type: Remove units of specific type
        """
        all_units = storage_backend.retrieve_all()
        pruned_ids = []

        if not criteria:
            return 0, pruned_ids

        if "max_age" in criteria:
            max_age_days = criteria["max_age"]
            cutoff_time = self._get_cutoff_time(max_age_days)
            for unit in all_units:
                if self._is_older_than(unit, cutoff_time):
                    pruned_ids.append(unit.get("id", ""))

        elif "max_count" in criteria:
            max_count = criteria["max_count"]
            if len(all_units) > max_count:
                sorted_units = sorted(
                    all_units,
                    key=lambda u: u.get("metadata", {}).get("created_at", ""),
                    reverse=True
                )
                units_to_remove = sorted_units[max_count:]
                pruned_ids = [u.get("id", "") for u in units_to_remove]

        elif "type" in criteria:
            target_type = criteria["type"]
            for unit in all_units:
                if unit.get("type") == target_type:
                    pruned_ids.append(unit.get("id", ""))

        for unit_id in pruned_ids:
            storage_backend.delete(unit_id)

        return len(pruned_ids), pruned_ids

    def consolidate(
        self,
        storage_backend,
        units: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Consolidate memory units by type."""
        if units:
            target_units = [
                storage_backend.retrieve(uid) for uid in units
                if storage_backend.exists(uid)
            ]
        else:
            target_units = storage_backend.retrieve_all()

        consolidated = []

        by_type = {}
        for unit in target_units:
            unit_type = unit.get("type", "unknown")
            if unit_type not in by_type:
                by_type[unit_type] = []
            by_type[unit_type].append(unit)

        for unit_type, type_units in by_type.items():
            if len(type_units) > 1:
                consolidated_unit = self._merge_units(type_units)
                consolidated.append(consolidated_unit)
            else:
                consolidated.append(type_units[0])

        return consolidated

    def deduplicate(
        self,
        storage_backend,
        similarity_threshold: float = 0.9
    ) -> Tuple[int, List[str]]:
        """Deduplicate memory units based on content similarity."""
        all_units = storage_backend.retrieve_all()
        removed_ids = []
        seen_contents = set()

        for unit in all_units:
            content = unit.get("content", "")
            unit_id = unit.get("id", "")

            if content in seen_contents:
                removed_ids.append(unit_id)
                storage_backend.delete(unit_id)
            else:
                seen_contents.add(content)

        return len(removed_ids), removed_ids

    def apply_forgetting(
        self,
        storage_backend,
        strategy: str = "lru",
        count: Optional[int] = None
    ) -> Tuple[int, List[str]]:
        """Apply forgetting mechanism to memory."""
        all_units = storage_backend.retrieve_all()
        forgotten_ids = []

        if strategy == "lru":
            sorted_units = sorted(
                all_units,
                key=lambda u: u.get("metadata", {}).get("created_at", "")
            )
        elif strategy == "random":
            import random
            sorted_units = all_units.copy()
            random.shuffle(sorted_units)
        else:
            sorted_units = all_units

        if count is None:
            count = max(1, len(all_units) // 10)

        forgotten_units = sorted_units[:count]
        forgotten_ids = [u.get("id", "") for u in forgotten_units]

        for unit_id in forgotten_ids:
            storage_backend.delete(unit_id)

        return len(forgotten_ids), forgotten_ids

    def get_health_metrics(
        self,
        storage_backend
    ) -> HealthMetrics:
        """Get health metrics for memory system."""
        # FAST PATH: Use cache for large databases to prevent blocking
        if hasattr(storage_backend, '_health_cache') and storage_backend._health_cache:
            cache_age = time.time() - storage_backend._health_cache_time
            if cache_age < 30:  # 30 second cache
                return storage_backend._health_cache

        # FALLBACK: Still use retrieve_all but only for small databases
        all_units = storage_backend.retrieve_all()
        total_count = len(all_units)

        if total_count == 0:
            return HealthMetrics(
                total_units=0,
                total_size_bytes=0,
                average_unit_size=0.0,
                oldest_unit_timestamp=None,
                newest_unit_timestamp=None,
                unit_types_distribution={},
                duplicate_count=0,
                last_operation="none",
                last_operation_time=self._get_timestamp()
            )

        import json
        total_size = 0
        timestamps = []
        type_dist = {}
        contents_seen = set()

        for unit in all_units:
            unit_json = json.dumps(unit)
            total_size += len(unit_json.encode('utf-8'))

            metadata = unit.get("metadata", {})
            created_at = metadata.get("created_at", "")
            if created_at:
                timestamps.append(created_at)

            unit_type = unit.get("type", "unknown")
            type_dist[unit_type] = type_dist.get(unit_type, 0) + 1

            content = unit.get("content", "")
            if content in contents_seen:
                pass
            contents_seen.add(content)

        duplicate_count = len(all_units) - len(contents_seen)

        health_metrics = HealthMetrics(
            total_units=total_count,
            total_size_bytes=total_size,
            average_unit_size=total_size / total_count,
            oldest_unit_timestamp=min(timestamps) if timestamps else None,
            newest_unit_timestamp=max(timestamps) if timestamps else None,
            unit_types_distribution=type_dist,
            duplicate_count=duplicate_count,
            last_operation="metrics",
            last_operation_time=self._get_timestamp()
        )

        # Cache the result to prevent future blocking operations
        if hasattr(storage_backend, '_health_cache'):
            storage_backend._health_cache = health_metrics
            storage_backend._health_cache_time = time.time()

        return health_metrics

    def _get_cutoff_time(self, max_age_days: int) -> str:
        """Get cutoff timestamp for pruning."""
        from datetime import datetime, timedelta, timezone
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        return cutoff.isoformat() + "Z"

    def _is_older_than(
        self,
        unit: Dict[str, Any],
        cutoff_time: str
    ) -> bool:
        """Check if unit is older than cutoff time."""
        created_at = unit.get("metadata", {}).get("created_at", "")
        return created_at < cutoff_time if created_at else False

    def _merge_units(
        self,
        units: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge multiple units into a consolidated unit."""
        if not units:
            return {}

        merged = {
            "type": units[0].get("type", "consolidated"),
            "content": "\n\n".join(
                [u.get("content", "") for u in units]
            ),
            "tags": [],
            "metadata": {
                "consolidated_from": len(units),
                "created_at": self._get_timestamp()
            }
        }

        all_tags = set()
        for unit in units:
            tags = unit.get("tags", [])
            if isinstance(tags, list):
                all_tags.update(tags)

        merged["tags"] = list(all_tags)
        return merged

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat() + "Z"
