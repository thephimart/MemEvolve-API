import logging
from typing import Any, Dict, List

import numpy as np

from ..utils.config import load_config

logger = logging.getLogger(__name__)


class MemoryScorer:
    """Calculate memory relevance scores for retrieved memories."""

    def __init__(self, config=None):
        self.config = config or load_config()

    def calculate_memory_relevance(self, query: str, memories: List[Dict]) -> List[float]:
        """Calculate semantic relevance scores for retrieved memories."""
        relevance_scores = []

        for memory in memories:
            if memory.get('embedding') and self._get_query_embedding(query):
                similarity = self._cosine_similarity(
                    self._get_query_embedding(query),
                    memory['embedding']
                )
                relevance_scores.append(float(similarity))
            else:
                # Fallback to text overlap using config-based tokenization
                relevance_scores.append(self._text_overlap_score(query, memory.get('content', '')))

        return relevance_scores

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding using centralized embedding configuration."""
        try:
            from ..utils.embeddings import create_embedding_function
            embedding_func = create_embedding_function(self.config.embedding)
            return embedding_func(query)
        except Exception as e:
            logger.warning(f"Failed to get query embedding: {e}")
            return None

    def _text_overlap_score(self, query: str, content: str) -> float:
        """Text overlap fallback using config-based processing."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.0

        overlap = len(query_words.intersection(content_words))
        return overlap / len(query_words)

    def calculate_memory_relevance_batch(self, requests: List[Dict]) -> Dict[str, Any]:
        """Calculate memory relevance for batch of requests."""
        all_relevance_scores = []
        memory_count_stats = []

        for request in requests:
            memories = request.get('memories_injected', [])
            query = request.get('original_query', '')

            if memories and query:
                relevance_scores = self.calculate_memory_relevance(query, memories)
                all_relevance_scores.extend(relevance_scores)
                memory_count_stats.append(len(memories))

        if not all_relevance_scores:
            return {
                'avg_relevance_score': 0.0,
                'max_relevance_score': 0.0,
                'min_relevance_score': 0.0,
                'total_memories_scored': 0,
                'avg_memories_per_request': 0.0
            }

        return {
            'avg_relevance_score': sum(all_relevance_scores) / len(all_relevance_scores),
            'max_relevance_score': max(all_relevance_scores),
            'min_relevance_score': min(all_relevance_scores),
            'total_memories_scored': len(all_relevance_scores),
            'avg_memories_per_request': sum(memory_count_stats) / len(memory_count_stats) if memory_count_stats else 0.0}
