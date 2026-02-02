from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime


class StorageBackend(ABC):
    """Abstract base class for all storage backends."""

    @abstractmethod
    def store(self, unit: Dict[str, Any]) -> str:
        """Store a memory unit and return its ID."""
        pass

    @abstractmethod
    def store_batch(self, units: List[Dict[str, Any]]) -> List[str]:
        """Store multiple memory units and return their IDs."""
        pass

    @abstractmethod
    def retrieve(self, unit_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory unit by ID."""
        pass

    @abstractmethod
    def retrieve_all(self) -> List[Dict[str, Any]]:
        """Retrieve all stored memory units."""
        pass

    @abstractmethod
    def update(self, unit_id: str, unit: Dict[str, Any]) -> bool:
        """Update a memory unit by ID."""
        pass

    @abstractmethod
    def delete(self, unit_id: str) -> bool:
        """Delete a memory unit by ID."""
        pass

    @abstractmethod
    def exists(self, unit_id: str) -> bool:
        """Check if a memory unit exists."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Get the count of stored memory units."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored memory units."""
        pass

    @abstractmethod
    def get_metadata(self, unit_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific unit."""
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


class MetadataMixin:
    """Mixin class for common metadata operations."""

    def _generate_timestamp(self) -> str:
        """Generate current ISO timestamp."""
        return datetime.now().isoformat()

    def _add_metadata(self, unit: Dict[str, Any]) -> Dict[str, Any]:
        """Add creation timestamp and other metadata to unit."""
        if "metadata" not in unit:
            unit["metadata"] = {}
        unit["metadata"]["created_at"] = self._generate_timestamp()
        return unit
