from typing import Dict, List, Any, Optional
from .base import RetrievalStrategy, RetrievalResult
from .keyword_strategy import KeywordRetrievalStrategy
from .semantic_strategy import SemanticRetrievalStrategy


class HybridRetrievalStrategy(RetrievalStrategy):
    """Hybrid retrieval strategy combining semantic and keyword matching."""

    def __init__(
        self,
        embedding_function,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        similarity_threshold: float = 0.0
    ):
        self.semantic_strategy = SemanticRetrievalStrategy(
            embedding_function=embedding_function,
            similarity_threshold=similarity_threshold
        )
        self.keyword_strategy = KeywordRetrievalStrategy()
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

    def retrieve(
        self,
        query: str,
        storage_backend,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve using hybrid semantic + keyword approach."""
        semantic_results = self.semantic_strategy.retrieve(
            query,
            storage_backend,
            top_k=top_k * 2,
            filters=filters
        )

        keyword_results = self.keyword_strategy.retrieve(
            query,
            storage_backend,
            top_k=top_k * 2,
            filters=filters
        )

        hybrid_results = self._combine_results(
            semantic_results,
            keyword_results,
            query
        )

        sorted_results = sorted(
            hybrid_results,
            key=lambda x: x.score,
            reverse=True
        )

        return sorted_results[:top_k]

    def retrieve_by_ids(
        self,
        unit_ids: List[str],
        storage_backend
    ) -> List[RetrievalResult]:
        """Retrieve specific memory units by their IDs."""
        return self.semantic_strategy.retrieve_by_ids(
            unit_ids,
            storage_backend
        )

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
        hybrid_results = self.retrieve(
            query,
            storage_backend,
            top_k=1000,
            filters=filters
        )

        relevant = [r for r in hybrid_results if r.score > 0]
        return len(relevant)

    def _combine_results(
        self,
        semantic_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        query: str
    ) -> List[RetrievalResult]:
        """Combine semantic and keyword results using weighted fusion."""
        combined = {}

        for idx, result in enumerate(semantic_results):
            if result.unit_id not in combined:
                combined[result.unit_id] = {
                    "unit": result.unit,
                    "semantic_score": result.score,
                    "semantic_rank": idx
                }
            else:
                combined[result.unit_id]["semantic_score"] = result.score
                combined[result.unit_id]["semantic_rank"] = idx

        for idx, result in enumerate(keyword_results):
            if result.unit_id not in combined:
                combined[result.unit_id] = {
                    "unit": result.unit,
                    "keyword_score": result.score,
                    "keyword_rank": idx
                }
            else:
                combined[result.unit_id]["keyword_score"] = result.score
                combined[result.unit_id]["keyword_rank"] = idx

        hybrid_results = []
        for unit_id, unit_data in combined.items():
            semantic_score = unit_data.get("semantic_score", 0)
            keyword_score = unit_data.get("keyword_score", 0)

            semantic_found = "semantic_score" in unit_data
            keyword_found = "keyword_score" in unit_data

            if semantic_found and keyword_found:
                normalized_semantic = semantic_score
                normalized_keyword = keyword_score

                hybrid_score = (
                    self.semantic_weight * normalized_semantic +
                    self.keyword_weight * normalized_keyword
                )
            elif semantic_found:
                hybrid_score = semantic_score
            elif keyword_found:
                hybrid_score = keyword_score
            else:
                # Fallback: if both are 0, use small positive score to preserve some semantic signal
                hybrid_score = max(semantic_score, keyword_score, 0.1)

            result = RetrievalResult(
                unit_id=unit_id,
                unit=unit_data["unit"],
                score=hybrid_score,
                metadata={
                    "semantic_score": semantic_score,
                    "keyword_score": keyword_score,
                    "semantic_rank": unit_data.get("semantic_rank"),
                    "keyword_rank": unit_data.get("keyword_rank")
                }
            )

            hybrid_results.append(result)

        return hybrid_results

    def set_weights(
        self,
        semantic_weight: float,
        keyword_weight: float
    ):
        """Update semantic and keyword weights."""
        total = semantic_weight + keyword_weight

        if total > 0:
            self.semantic_weight = semantic_weight / total
            self.keyword_weight = keyword_weight / total
        else:
            self.semantic_weight = 0.5
            self.keyword_weight = 0.5
