import json
import os
from typing import Any, Dict, List, Optional

from .base import MetadataMixin, StorageBackend
from ...utils.logging_manager import LoggingManager

logger = LoggingManager.get_logger(__name__)
logger.debug("JSON file store backend initialized")


class JSONFileStore(StorageBackend, MetadataMixin):
    """JSON file-based storage backend for memory units."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        """Load data from JSON file."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.data = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load data: {str(e)}")
        else:
            self.data = {}

    def _save(self):
        """Save data to JSON file."""
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save data: {str(e)}")

    def store(self, unit: Dict[str, Any]) -> str:
        """Store a memory unit and return its ID."""
        unit = self._add_metadata(unit.copy())
        unit_id = unit.get("id", f"unit_{len(self.data)}")
        if "id" not in unit:
            unit["id"] = unit_id
        self.data[unit_id] = unit
        self._save()
        return unit_id

    def store_batch(self, units: List[Dict[str, Any]]) -> List[str]:
        """Store multiple memory units and return their IDs."""
        ids = []
        for unit in units:
            ids.append(self.store(unit))
        return ids

    def retrieve(self, unit_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory unit by ID."""
        return self.data.get(unit_id)

    def retrieve_all(self) -> List[Dict[str, Any]]:
        """Retrieve all stored memory units."""
        return list(self.data.values())

    def update(self, unit_id: str, unit: Dict[str, Any]) -> bool:
        """Update a memory unit by ID."""
        if unit_id in self.data:
            if "metadata" not in unit:
                unit["metadata"] = {}
            unit["metadata"]["updated_at"] = self._generate_timestamp()
            unit["id"] = unit_id
            self.data[unit_id] = unit
            self._save()
            return True
        return False

    def delete(self, unit_id: str) -> bool:
        """Delete a memory unit by ID."""
        if unit_id in self.data:
            del self.data[unit_id]
            self._save()
            return True
        return False

    def exists(self, unit_id: str) -> bool:
        """Check if a memory unit exists."""
        return unit_id in self.data

    def count(self) -> int:
        """Get the count of stored memory units."""
        return len(self.data)

    def clear(self) -> None:
        """Clear all stored memory units."""
        self.data.clear()
        self._save()

    def get_metadata(self, unit_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific unit."""
        unit = self.retrieve(unit_id)
        if unit and "metadata" in unit:
            return unit["metadata"]
        return None
