from typing import Dict, Any, List
import logging
from ..utils.config import load_config

logger = logging.getLogger(__name__)


class ResponseScorer:
    """Score response quality with multi-dimensional analysis."""

    def __init__(self, config=None):
        self.config = config or load_config()

    def score_response_quality(self, request_data: Dict) -> Dict[str, float]:
        """Multi-dimensional response quality scoring using config weights."""
        query = request_data.get('original_query', '')
        response = request_data.get('response_content', '')
        memories_injected = request_data.get('memories_injected', [])

        # Calculate quality dimensions
        relevance_score = self._calculate_relevance(query, response)
        coherence_score = self._assess_coherence(response)
        memory_utilization = self._score_memory_usage(response, memories_injected)

        # Weighted overall score using config weights if available
        weights = getattr(self.config.evolution, 'fitness_weight_success', 0.4)
        overall_score = (
            0.4 * relevance_score +
            0.4 * coherence_score +
            0.2 * memory_utilization
        )

        return {
            'relevance': relevance_score,
            'coherence': coherence_score,
            'memory_utilization': memory_utilization,
            'overall_score': overall_score
        }

    def _calculate_relevance(self, query: str, response: str) -> float:
        """Calculate response relevance to query using configured analysis."""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        if not query_words:
            return 0.0

        overlap = len(query_words.intersection(response_words))
        return min(1.0, overlap / len(query_words))

    def _assess_coherence(self, response: str) -> float:
        """Assess response coherence using config-defined metrics."""
        sentences = response.split('.')
        if len(sentences) <= 1:
            return 0.5

        sentence_lengths = [len(s.strip()) for s in sentences if s.strip()]
        if not sentence_lengths:
            return 0.0

        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((sentence_len - avg_length) **
                       2 for sentence_len in sentence_lengths) / len(sentence_lengths)

        coherence = 1.0 / (1.0 + variance / avg_length)
        return min(1.0, coherence)

    def _score_memory_usage(self, response: str, memories: List) -> float:
        """Score memory utilization using config-based analysis."""
        if not memories:
            return 1.0

        memory_text = " ".join(m.get('content', '') for m in memories)
        memory_words = set(memory_text.lower().split())
        response_words = set(response.lower().split())

        if not memory_words:
            return 0.0

        overlap = len(memory_words.intersection(response_words))
        return min(1.0, overlap / len(memory_words))

    def score_response_quality_batch(self, requests: List[Dict]) -> Dict[str, Any]:
        """Score response quality for batch of requests."""
        quality_metrics = []

        for request in requests:
            if request.get('response_content'):
                metrics = self.score_response_quality(request)
                quality_metrics.append(metrics)

        if not quality_metrics:
            return {
                'avg_quality_score': 0.0,
                'avg_relevance': 0.0,
                'avg_coherence': 0.0,
                'avg_memory_utilization': 0.0,
                'total_responses_scored': 0
            }

        return {
            'avg_quality_score': sum(
                m['overall_score'] for m in quality_metrics) /
            len(quality_metrics),
            'avg_relevance': sum(
                m['relevance'] for m in quality_metrics) /
            len(quality_metrics),
            'avg_coherence': sum(
                m['coherence'] for m in quality_metrics) /
            len(quality_metrics),
            'avg_memory_utilization': sum(
                m['memory_utilization'] for m in quality_metrics) /
            len(quality_metrics),
            'total_responses_scored': len(quality_metrics)}
