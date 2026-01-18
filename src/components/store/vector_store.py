from typing import Dict, List, Any, Optional, Callable
import numpy as np
import pickle
import os
from .base import StorageBackend, MetadataMixin


class VectorStore(StorageBackend, MetadataMixin):
    """FAISS-based vector store backend for memory units."""

    def __init__(
        self,
        index_file: str,
        embedding_function: Callable[[str], np.ndarray],
        embedding_dim: int = 384
    ):
        self.index_file = index_file
        self.embedding_function = embedding_function
        self.embedding_dim = embedding_dim

        self.data: Dict[str, Dict[str, Any]] = {}
        self.index = None
        self._load_index()
        self._load_data()

    def _load_index(self):
        """Load FAISS index from file."""
        if os.path.exists(self.index_file + ".index"):
            try:
                import faiss
                self.index = faiss.read_index(self.index_file + ".index")
            except Exception as e:
                raise RuntimeError(f"Failed to load index: {str(e)}")
        else:
            self._create_index()

    def _create_index(self):
        """Create new FAISS index."""
        try:
            import faiss
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        except Exception as e:
            raise RuntimeError(f"Failed to create index: {str(e)}")

    def _load_data(self):
        """Load metadata from file."""
        if os.path.exists(self.index_file + ".data"):
            try:
                with open(self.index_file + ".data", 'rb') as f:
                    self.data = pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load data: {str(e)}")

    def _save_index(self):
        """Save FAISS index to file."""
        try:
            import faiss
            faiss.write_index(self.index, self.index_file + ".index")
        except Exception as e:
            raise RuntimeError(f"Failed to save index: {str(e)}")

    def _save_data(self):
        """Save metadata to file."""
        try:
            with open(self.index_file + ".data", 'wb') as f:
                pickle.dump(self.data, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save data: {str(e)}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        embedding = self.embedding_function(text)
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"expected {self.embedding_dim}, "
                f"got {embedding.shape[0]}"
            )
        return embedding.reshape(1, -1).astype('float32')

    def store(self, unit: Dict[str, Any]) -> str:
        """Store a memory unit and return its ID."""
        unit = self._add_metadata(unit.copy())
        unit_id = unit.get("id", f"unit_{len(self.data)}")
        if "id" not in unit:
            unit["id"] = unit_id

        text = self._unit_to_text(unit)
        embedding = self._get_embedding(text)

        self.data[unit_id] = unit
        self.index.add(embedding)

        self._save_index()
        self._save_data()

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

            self._save_index()
            self._save_data()

            return True
        return False

    def delete(self, unit_id: str) -> bool:
        """Delete a memory unit by ID."""
        if unit_id in self.data:
            del self.data[unit_id]
            self._rebuild_index()
            self._save_index()
            self._save_data()
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
        self._create_index()
        self._save_index()
        self._save_data()

    def get_metadata(self, unit_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific unit."""
        unit = self.retrieve(unit_id)
        if unit and "metadata" in unit:
            return unit["metadata"]
        return None

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[tuple]:
        """Search for similar memory units.

        Returns:
            List of (distance, unit_id) tuples
        """
        if self.index.ntotal == 0:
            return []

        embedding = self._get_embedding(query)
        distances, indices = self.index.search(embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                unit_id = list(self.data.keys())[idx]
                results.append((float(dist), unit_id))

        return results

    def _rebuild_index(self):
        """Rebuild index from current data."""
        self._create_index()

        if len(self.data) > 0:
            embeddings = []
            for unit in self.data.values():
                text = self._unit_to_text(unit)
                embedding = self._get_embedding(text)
                embeddings.append(embedding)

            all_embeddings = np.vstack(embeddings).astype('float32')
            self.index.add(all_embeddings)

    def _unit_to_text(self, unit: Dict[str, Any]) -> str:
        """Convert unit to text for embedding."""
        text_parts = []

        if "content" in unit:
            text_parts.append(str(unit["content"]))

        if "tags" in unit and isinstance(unit["tags"], list):
            text_parts.append(" ".join(unit["tags"]))

        if "type" in unit:
            text_parts.append(str(unit["type"]))

        return " ".join(text_parts)
