from typing import Dict, List, Any, Optional, Callable
from .base import RetrievalStrategy, RetrievalResult
import numpy as np


class SemanticRetrievalStrategy(RetrievalStrategy):
    """Semantic retrieval strategy using vector embeddings."""

    def __init__(
        self,
        embedding_function: Callable[[str], np.ndarray],
        similarity_threshold: float = 0.0
    ):
        self.embedding_function = embedding_function
        self.similarity_threshold = similarity_threshold
        self._cache: Dict[str, np.ndarray] = {}

    def retrieve(
        self,
        query: str,
        storage_backend,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve memory units based on semantic similarity."""
        all_units = storage_backend.retrieve_all()
        filtered_units = self._apply_filters(all_units, filters)
        scored_units = self._score_units(query, filtered_units)

        filtered_scores = [
            u for u in scored_units
            if u.score >= self.similarity_threshold
        ]

        sorted_units = sorted(
            filtered_scores,
            key=lambda x: x.score,
            reverse=True
        )

        return sorted_units[:top_k]

    def retrieve_by_ids(
        self,
        unit_ids: List[str],
        storage_backend
    ) -> List[RetrievalResult]:
        """Retrieve specific memory units by their IDs."""
        results = []
        for unit_id in unit_ids:
            unit = storage_backend.retrieve(unit_id)
            if unit:
                results.append(RetrievalResult(
                    unit_id=unit_id,
                    unit=unit,
                    score=1.0
                ))
        return results

    def search(
        self,
        query: str,
        storage_backend,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Search for memory units matching query."""
        return self.retrieve(query, storage_backend, top_k, filters)

    def count_relevant(
        self,
        query: str,
        storage_backend,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count memory units relevant to query."""
        all_units = storage_backend.retrieve_all()
        filtered_units = self._apply_filters(all_units, filters)
        scored_units = self._score_units(query, filtered_units)
        relevant = [
            u for u in scored_units
            if u.score >= self.similarity_threshold
        ]
        return len(relevant)

    def _apply_filters(
        self,
        units: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply filters to units."""
        if not filters:
            return units

        filtered = []
        for unit in units:
            match = True
            for key, value in filters.items():
                if key not in unit or unit[key] != value:
                    match = False
                    break
            if match:
                filtered.append(unit)
        return filtered

    def _score_units(
        self,
        query: str,
        units: List[Dict[str, Any]]
    ) -> List[RetrievalResult]:
        """Score units based on semantic similarity."""
        results = []
        query_embedding = self._get_embedding(query)

        for unit in units:
            unit_text = self._unit_to_text(unit)
            unit_embedding = self._get_embedding(unit_text)
            score = self._cosine_similarity(query_embedding, unit_embedding)

            unit_id = unit.get("id", "")
            results.append(RetrievalResult(
                unit_id=unit_id,
                unit=unit,
                score=score,
                metadata={"embedding_dim": len(query_embedding)}
            ))

        return results

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching."""
        if text in self._cache:
            return self._cache[text]

        embedding = self.embedding_function(text)
        self._cache[text] = embedding
        return embedding

    def _cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def _unit_to_text(self, unit: Dict[str, Any]) -> str:
        """Convert unit to searchable text."""
        text_parts = []

        if "content" in unit:
            text_parts.append(str(unit["content"]))

        if "tags" in unit and isinstance(unit["tags"], list):
            text_parts.append(" ".join(unit["tags"]))

        if "type" in unit:
            text_parts.append(str(unit["type"]))

        return " ".join(text_parts)

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
