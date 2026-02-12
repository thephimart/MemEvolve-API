import re
from typing import Any, Dict, List, Optional

from .base import RetrievalResult, RetrievalStrategy
from ...utils.logging_manager import LoggingManager

logger = LoggingManager.get_logger(__name__)
logger.debug("Keyword retrieval strategy initialized")


class KeywordRetrievalStrategy(RetrievalStrategy):
    """Keyword-based retrieval strategy using simple text matching."""

    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    def retrieve(
        self,
        query: str,
        storage_backend,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve memory units based on keyword matching."""
        all_units = storage_backend.retrieve_all()
        filtered_units = self._apply_filters(all_units, filters)
        scored_units = self._score_units(query, filtered_units)

        sorted_units = sorted(
            scored_units,
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
        relevant = [u for u in scored_units if u.score > 0]
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
        """Score units based on keyword matches."""
        results = []
        query_terms = self._extract_terms(query)

        for unit in units:
            score = self._calculate_score(query_terms, unit)
            if score > 0:
                unit_id = unit.get("id", "")
                results.append(RetrievalResult(
                    unit_id=unit_id,
                    unit=unit,
                    score=score,
                    metadata={"matching_terms": query_terms}
                ))

        return results

    def _extract_terms(self, query: str) -> List[str]:
        """Extract searchable terms from query."""
        if not self.case_sensitive:
            query = query.lower()

        terms = re.findall(r'\w+', query)
        return [t for t in terms if len(t) > 2]

    def _calculate_score(
        self,
        query_terms: List[str],
        unit: Dict[str, Any]
    ) -> float:
        """Calculate relevance score for unit with improved weighting."""
        if not query_terms:
            return 0.0

        unit_text = self._unit_to_text(unit)
        if not self.case_sensitive:
            unit_text = unit_text.lower()

        total_weight = 0.0
        matched_weight = 0.0

        for term in query_terms:
            term_weight = len(term) ** 2
            total_weight += term_weight

            if term in unit_text:
                matched_weight += term_weight

        base_score = matched_weight / total_weight if total_weight > 0 else 0.0

        phrase_bonus = 0.0
        query_text = " ".join(query_terms)
        if len(query_text) > 20 and query_text in unit_text:
            phrase_bonus = 0.3

        overlap_bonus = 0.0
        content_words = set(unit_text.split())
        query_word_set = set(query_terms)
        overlap = len(content_words & query_word_set)
        if overlap > 5:
            overlap_bonus = 0.1

        final_score = min(base_score + phrase_bonus + overlap_bonus, 1.0)
        return final_score

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
