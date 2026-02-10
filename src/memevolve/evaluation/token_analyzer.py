from typing import Any, Dict, List

from ..utils.config import load_config
from ..utils.logging_manager import LoggingManager

logger = LoggingManager.get_logger(__name__)


class TokenAnalyzer:
    """Analyze token efficiency with realistic baselines and cost-benefit evaluation."""

    def __init__(self, config=None):
        self.config = config or load_config()

    def calculate_efficiency_metrics(self, request_data: Dict) -> Dict[str, float]:
        """Calculate token efficiency using config-defined baselines."""
        actual_tokens = request_data.get('total_tokens_used', 0)
        memory_tokens = request_data.get('memory_tokens', 0)

        # Realistic baseline using config-based estimation
        realistic_baseline = self._estimate_realistic_baseline(
            request_data.get('original_query', ''))

        # Calculate efficiency scores
        efficiency_score = self._calculate_efficiency_score(
            actual_tokens, realistic_baseline, memory_tokens)
        memory_value = self._calculate_memory_value(request_data.get('memories_injected', []))

        return {
            'actual_tokens': actual_tokens,
            'realistic_baseline': realistic_baseline,
            'memory_tokens': memory_tokens,
            'efficiency_score': efficiency_score,
            'memory_value_score': memory_value,
            'net_savings': realistic_baseline - actual_tokens,
            'cost_per_token': self._calculate_cost_per_token(request_data)
        }

    def _estimate_realistic_baseline(self, query: str) -> int:
        """Estimate realistic baseline using config-defined factors."""
        # Use config-based estimation factors
        base_factor = getattr(self.config.evolution_boundaries, 'baseline_token_factor', 3.0)
        min_baseline = getattr(self.config.evolution_boundaries, 'min_baseline_tokens', 50)
        max_baseline = getattr(self.config.evolution_boundaries, 'max_baseline_tokens', 200)

        query_words = len(query.split())
        base_tokens = max(min_baseline, query_words * base_factor)
        return min(max_baseline, base_tokens)

    def _calculate_efficiency_score(self, actual: int, baseline: int, memory_tokens: int) -> float:
        """Calculate efficiency score (0-1, higher is better)."""
        if actual <= baseline:
            return 1.0

        overhead = actual - baseline
        if memory_tokens == 0:
            return max(0.0, 1.0 - (overhead / baseline))

        memory_ratio = memory_tokens / actual
        efficiency = 1.0 - (overhead / baseline) * (1 - memory_ratio)
        return max(0.0, min(1.0, efficiency))

    def _calculate_memory_value(self, memories: List) -> float:
        """Calculate memory value using config-defined thresholds."""
        if not memories:
            return 0.0

        max_valuable_memories = getattr(
            self.config.evolution_boundaries, 'max_valuable_memories', 5)
        value_score = min(1.0, len(memories) / max_valuable_memories)
        return value_score

    def _calculate_cost_per_token(self, request_data: Dict) -> float:
        """Calculate cost per token using config-defined metrics."""
        total_tokens = request_data.get('total_tokens_used', 1)
        total_time = request_data.get('total_request_time_ms', 1)

        return total_time / total_tokens

    def calculate_efficiency_metrics_batch(self, requests: List[Dict]) -> Dict[str, Any]:
        """Calculate efficiency metrics for batch of requests."""
        efficiency_metrics = []

        for request in requests:
            if request.get('total_tokens_used'):
                metrics = self.calculate_efficiency_metrics(request)
                efficiency_metrics.append(metrics)

        if not efficiency_metrics:
            return {
                'avg_efficiency_score': 0.0,
                'avg_net_savings': 0.0,
                'total_memory_tokens': 0,
                'total_actual_tokens': 0,
                'total_requests_analyzed': 0
            }

        return {
            'avg_efficiency_score': sum(
                m['efficiency_score'] for m in efficiency_metrics) /
            len(efficiency_metrics),
            'avg_net_savings': sum(
                m['net_savings'] for m in efficiency_metrics) /
            len(efficiency_metrics),
            'total_memory_tokens': sum(
                m['memory_tokens'] for m in efficiency_metrics),
            'total_actual_tokens': sum(
                m['actual_tokens'] for m in efficiency_metrics),
            'total_requests_analyzed': len(efficiency_metrics)}
