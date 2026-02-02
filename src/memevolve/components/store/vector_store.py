"""
Vector store implementation using FAISS for efficient similarity search.

FAISS deprecation warnings: The warnings about SwigPyPacked, SwigPyObject, and swigvarlink
are from FAISS's internal SWIG-generated Python bindings. These are cosmetic warnings that
don't affect functionality. FAISS is still actively maintained and provides the best
performance for vector similarity search. These warnings are safely suppressed.
"""

from .base import StorageBackend, MetadataMixin
import os
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import warnings
import logging

logger = logging.getLogger(__name__)
# Suppress FAISS SWIG deprecation warnings (cosmetic, don't affect functionality)
warnings.filterwarnings(
    "ignore", message=".*SwigPyPacked.*", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore", message=".*SwigPyObject.*", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore", message=".*swigvarlink.*", category=DeprecationWarning)


class VectorStore(StorageBackend, MetadataMixin):
    """FAISS-based vector store backend for memory units."""

    def __init__(
        self,
        index_file: str,
        embedding_function: Callable[[str], np.ndarray],
        embedding_dim: int = 384,
        index_type: str = 'flat'
    ):
        self.index_file = index_file
        self.embedding_function = embedding_function
        self.embedding_dim = embedding_dim
        self.index_type = index_type

        self.data: Dict[str, Dict[str, Any]] = {}
        self.index = None
        
        # Health metrics cache to prevent blocking operations
        self._health_cache = None
        self._health_cache_time = 0

        # Try to load existing index and data, fall back to creating new one
        if not self._load_index():
            self._create_index()
        self._load_data()

    def _load_index(self) -> bool:
        """Load FAISS index from file. Returns True if successful."""
        try:
            import faiss
            self.index = faiss.read_index(self.index_file + ".index")
            return True
        except Exception:
            # File doesn't exist or is corrupted - will create new index
            return False

    def _create_index(self):
        """Create new FAISS index based on index_type."""
        try:
            import faiss
            if self.index_type == 'flat':
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            elif self.index_type == 'ivf':
                # IVF index with flat quantization - requires training before adding vectors
                nlist = min(100, max(4, int(4 * (self.embedding_dim ** 0.5))))
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                # IVF indexes are not trained initially - training happens automatically
                # on first add
            elif self.index_type == 'hnsw':
                # HNSW index
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            else:
                # Default to flat
                self.index = faiss.IndexFlatL2(self.embedding_dim)
        except Exception as e:
            raise RuntimeError(f"Failed to create {self.index_type} index: {str(e)}")

    def _train_ivf_if_needed(self, embedding: np.ndarray):
        """Train IVF index if not already trained."""
        if self.index_type == 'ivf':
            try:
                import faiss
                # Check if it's an IVF index and needs training
                if hasattr(self.index, 'is_trained') and not self.index.is_trained:  # type: ignore
                    # Need at least nlist training vectors, but for small datasets use what's
                    # available
                    nlist = getattr(self.index, 'nlist', 4)  # type: ignore
                    if embedding.shape[0] < nlist:
                        # Duplicate the embedding to meet minimum training requirement
                        train_data = np.tile(embedding, (nlist, 1)).astype('float32')
                    else:
                        train_data = embedding.reshape(1, -1).astype('float32')

                    self.index.train(train_data)  # type: ignore
                    print(f"Trained IVF index with {train_data.shape[0]} vectors")
            except Exception as e:
                print(f"Failed to train IVF index: {e}, falling back to flat index")
                # Fallback to flat index if training fails
                import faiss
                self.index = faiss.IndexFlatL2(self.embedding_dim)

    def _save_index(self):
        """Save FAISS index to file."""
        try:
            import faiss
            faiss.write_index(self.index, self.index_file + ".index")
        except Exception as e:
            raise RuntimeError(f"Failed to save index: {str(e)}")

    def _load_data(self):
        """Load metadata from file."""
        try:
            data_file = self.index_file + ".data"
            if os.path.exists(data_file):
                with open(data_file, 'rb') as f:
                    self.data = pickle.load(f)
        except Exception as e:
            # If data file is corrupted, start with empty data
            self.data = {}

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
        """Store a memory unit and return its ID with improved error handling."""
        unit = self._add_metadata(unit.copy())
        unit_id = unit.get("id", f"unit_{len(self.data)}")
        if "id" not in unit:
            unit["id"] = unit_id

        # Validate embedding generation with better error logging
        try:
            text = self._unit_to_text(unit)
            embedding = self._get_embedding(text)
        except Exception as e:
            logger.error(f"Failed to generate embedding for unit {unit_id}: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")

        # Train IVF index if needed before first add
        try:
            self._train_ivf_if_needed(embedding)
        except Exception as e:
            logger.error(f"Failed to train IVF index for unit {unit_id}: {e}")
            raise RuntimeError(f"Index training failed: {e}")

        # Validate index is ready
        if self.index is None:
            raise RuntimeError("FAISS index not initialized")

        # Add with error handling and logging
        try:
            # Add to data structure first
            self.data[unit_id] = unit
            
            # Add to FAISS index with proper parameter name
            self.index.add(x=embedding)

            # Persist to disk - critical for preventing data loss
            try:
                self._save_index()
                self._save_data()
                logger.debug(f"Successfully stored unit {unit_id}")
            except Exception as save_error:
                logger.error(f"Failed to persist unit {unit_id} to disk: {save_error}")
                
                # Remove from in-memory data since persistence failed
                if unit_id in self.data:
                    del self.data[unit_id]
                
                raise RuntimeError(f"Storage persistence failed: {save_error}")

            return unit_id
            
        except Exception as e:
            logger.error(f"Failed to store unit {unit_id}: {e}")
            raise RuntimeError(f"Storage failed: {e}")

    def store_batch(self, units: List[Dict[str, Any]]) -> List[str]:
        """Store multiple memory units using individual store() calls for reliability."""
        if not units:
            return []

        logger.info(f"Starting batch storage of {len(units)} units")
        ids = []
        failed_units = []

        # Use the proven single store() method for each unit
        # This ensures proper error handling and persistence for each unit
        for i, unit in enumerate(units):
            try:
                unit_id = self.store(unit)
                ids.append(unit_id)
                logger.debug(f"Batch unit {i+1}/{len(units)} stored successfully: {unit_id}")
            except Exception as e:
                logger.error(f"Batch unit {i+1}/{len(units)} failed: {e}")
                failed_units.append((i, str(e)))
                
                # Continue trying to store remaining units (fail-fast approach)
                continue
        
        if failed_units:
            failed_summary = [f"Unit {idx}: {err}" for idx, err in failed_units]
            logger.error(f"Batch storage partially failed: {len(failed_units)}/{len(units)} units failed")
            logger.error(f"Failed units: {failed_summary}")
            
            # Return partial success - don't raise exception to allow partial completion
            # This is better than all-or-nothing for large batches
            if len(failed_units) == len(units):
                # All failed - raise exception
                raise RuntimeError(f"All {len(units)} units in batch failed to store")
            else:
                # Partial success - log but continue
                logger.warning(f"Batch storage: {len(ids)} succeeded, {len(failed_units)} failed")
        
        logger.info(f"Batch storage completed: {len(ids)}/{len(units)} units stored successfully")
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
