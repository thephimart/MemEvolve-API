"""
Endpoint Metrics Collector for MemEvolve-API
==========================================

Comprehensive tracking of tokens, timing, and performance for each endpoint:
- Upstream API endpoint tracking
- Memory API endpoint tracking
- Embedding API endpoint tracking
- Full request/response pipeline tracking
"""

import json
import logging
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict
import statistics
import os

# Import enhanced scoring systems
try:
    from ..evaluation.memory_scorer import MemoryScorer
    from ..evaluation.response_scorer import ResponseScorer
    from ..evaluation.token_analyzer import TokenAnalyzer
    SCORING_AVAILABLE = True
except ImportError:
    SCORING_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(
        "Enhanced scoring systems not available - using legacy scoring")

logger = logging.getLogger(__name__)


@dataclass
class EndpointMetrics:
    """Single endpoint call metrics."""
    endpoint_type: str  # 'upstream', 'memory', 'embedding'
    call_id: str
    timestamp: float

    # Token metrics
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Timing metrics
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0

    # Status metrics
    success: bool = True
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Additional metrics
    model: Optional[str] = None
    temperature: Optional[float] = None
    request_type: Optional[str] = None


@dataclass
class RequestMetrics:
    """Complete request pipeline metrics."""
    request_id: str
    timestamp: float

    # Request composition
    original_query: str
    query_tokens: int

    # Endpoint calls tracking
    upstream_call: Optional[EndpointMetrics] = None
    memory_calls: List[EndpointMetrics] = field(default_factory=list)
    embedding_calls: List[EndpointMetrics] = field(default_factory=list)

    # Aggregate metrics
    total_tokens_used: int = 0
    baseline_tokens_estimate: int = 0
    net_token_savings: int = 0

    # Timing
    total_request_time_ms: float = 0.0
    memory_overhead_ms: float = 0.0
    upstream_time_ms: float = 0.0

    # Quality & effectiveness
    memories_retrieved: int = 0
    memories_injected: int = 0
    memory_relevance_scores: List[float] = field(default_factory=list)

    # Business impact (legacy - to be replaced by enhanced scoring)
    business_value_score: float = 0.0
    roi_score: float = 0.0

    # Enhanced scoring fields (Phase 2 implementation)
    memory_relevance_score: float = 0.0
    response_quality_score: float = 0.0
    token_efficiency_score: float = 0.0
    response_relevance: float = 0.0
    response_coherence: float = 0.0
    memory_utilization: float = 0.0
    enhanced_net_token_savings: float = 0.0


class EndpointMetricsCollector:
    """Thread-safe metrics collector for all endpoints."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.metrics_dir = self.data_dir / "endpoint_metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Thread-safe storage
        self._lock = threading.Lock()
        self._active_requests: Dict[str, RequestMetrics] = {}
        self._completed_requests: List[RequestMetrics] = []

        # Initialize enhanced scoring systems
        self._scoring_systems_initialized = False
        self.memory_scorer = None
        self.response_scorer = None
        self.token_analyzer = None

        if SCORING_AVAILABLE:
            self._initialize_scoring_systems()

    def _initialize_scoring_systems(self):
        """Initialize enhanced scoring systems."""
        try:
            from ..utils.config import load_config
            config = load_config()

            self.memory_scorer = MemoryScorer(config)
            self.response_scorer = ResponseScorer(config)
            self.token_analyzer = TokenAnalyzer(config)
            self._scoring_systems_initialized = True

            logger.info("Enhanced scoring systems initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced scoring systems: {e}")
            self._scoring_systems_initialized = False

        # Endpoint-specific metrics storage
        self.upstream_metrics_file = self.metrics_dir / "upstream_metrics.json"
        self.memory_metrics_file = self.metrics_dir / "memory_metrics.json"
        self.embedding_metrics_file = self.metrics_dir / "embedding_metrics.json"
        self.request_pipeline_file = self.metrics_dir / "request_pipeline.json"

        # In-memory caches for performance
        self._endpoint_cache: Dict[str, List[EndpointMetrics]] = defaultdict(list)
        self._max_cache_size = 1000

        logger.debug(f"EndpointMetricsCollector initialized with data_dir: {self.data_dir}")

    def start_request_tracking(self, request_id: str, query: str,
                               query_tokens: int) -> RequestMetrics:
        """Start tracking a new request."""
        with self._lock:
            request_metrics = RequestMetrics(
                request_id=request_id,
                timestamp=time.time(),
                original_query=query,
                query_tokens=query_tokens
            )
            self._active_requests[request_id] = request_metrics
            logger.debug(f"Started tracking request {request_id}")
            return request_metrics

    def start_endpoint_call(
        self,
        request_id: str,
        endpoint_type: str,
        input_tokens: int = 0,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        request_type: Optional[str] = None
    ) -> str:
        """Start tracking an endpoint call."""
        call_id = f"{request_id}_{endpoint_type}_{int(time.time() * 1000)}"

        endpoint_metrics = EndpointMetrics(
            endpoint_type=endpoint_type,
            call_id=call_id,
            timestamp=time.time(),
            input_tokens=input_tokens,
            start_time=time.time(),
            model=model,
            temperature=temperature,
            request_type=request_type
        )

        with self._lock:
            if request_id in self._active_requests:
                request_metrics = self._active_requests[request_id]
                if endpoint_type == 'upstream':
                    request_metrics.upstream_call = endpoint_metrics
                elif endpoint_type == 'memory':
                    request_metrics.memory_calls.append(endpoint_metrics)
                elif endpoint_type == 'embedding':
                    request_metrics.embedding_calls.append(endpoint_metrics)

            # Add to endpoint cache
            if len(self._endpoint_cache[endpoint_type]) >= self._max_cache_size:
                self._endpoint_cache[endpoint_type] = self._endpoint_cache[endpoint_type][-self._max_cache_size // 2:]
            self._endpoint_cache[endpoint_type].append(endpoint_metrics)

        logger.debug(f"Started {endpoint_type} call {call_id} for request {request_id}")
        return call_id

    def end_endpoint_call(
        self,
        call_id: str,
        output_tokens: int = 0,
        success: bool = True,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """Complete tracking of an endpoint call."""
        with self._lock:
            # Find the endpoint metrics
            endpoint_metrics = self._find_endpoint_metrics(call_id)
            if not endpoint_metrics:
                logger.warning(f"Endpoint call {call_id} not found for completion")
                return

            # Update completion metrics
            endpoint_metrics.end_time = time.time()
            endpoint_metrics.duration_ms = (
                endpoint_metrics.end_time - endpoint_metrics.start_time) * 1000
            endpoint_metrics.output_tokens = output_tokens
            endpoint_metrics.total_tokens = endpoint_metrics.input_tokens + output_tokens
            endpoint_metrics.success = success
            endpoint_metrics.error_code = error_code
            endpoint_metrics.error_message = error_message

            # Update parent request metrics if available
            request_id = call_id.split('_')[0]
            if request_id in self._active_requests:
                request_metrics = self._active_requests[request_id]
                request_metrics.total_tokens_used += endpoint_metrics.total_tokens

        logger.debug(
            f"Completed {
                endpoint_metrics.endpoint_type} call {call_id} in {
                endpoint_metrics.duration_ms:.1f}ms")

    def end_request_tracking(
        self,
        request_id: str,
        response_tokens: int = 0,
        total_time_ms: float = 0.0,
        business_value_score: float = 0.0
    ):
        """Complete tracking of a request and calculate business metrics."""
        with self._lock:
            if request_id not in self._active_requests:
                logger.warning(f"Request {request_id} not found for completion")
                return

            request_metrics = self._active_requests.pop(request_id)

            # Update completion metrics
            request_metrics.total_request_time_ms = total_time_ms
            request_metrics.business_value_score = business_value_score

            # Calculate aggregate metrics
            if request_metrics.upstream_call:
                request_metrics.upstream_time_ms = request_metrics.upstream_call.duration_ms
                request_metrics.total_tokens_used += request_metrics.upstream_call.total_tokens

            # Calculate memory overhead
            total_memory_time = sum(call.duration_ms for call in request_metrics.memory_calls)
            request_metrics.memory_overhead_ms = total_memory_time

            # Calculate memory effectiveness
            request_metrics.memories_retrieved = len(request_metrics.memory_calls)
            request_metrics.memories_injected = len(
                [call for call in request_metrics.memory_calls if call.success])

            # Calculate baseline estimate and savings
            request_metrics.baseline_tokens_estimate = self._estimate_baseline_tokens(
                request_metrics.original_query, request_metrics.query_tokens
            )
            request_metrics.net_token_savings = request_metrics.baseline_tokens_estimate - \
                request_metrics.total_tokens_used

            # Calculate ROI score
            request_metrics.roi_score = self._calculate_roi_score(request_metrics)

            # Calculate enhanced scores (Phase 2)
            self.calculate_enhanced_scores(request_metrics)

            # Add to completed requests
            self._completed_requests.append(request_metrics)

            # Limit memory usage
            if len(self._completed_requests) > 10000:
                self._completed_requests = self._completed_requests[-5000:]

        # Trigger async save
        self._save_metrics_async()
        logger.info(
            f"Completed tracking request {request_id} with ROI score: {
                request_metrics.roi_score:.3f}")

    def get_endpoint_stats(self, endpoint_type: str, limit: int = 100) -> Dict[str, Any]:
        """Get statistics for a specific endpoint type."""
        with self._lock:
            endpoint_calls = self._endpoint_cache.get(endpoint_type, [])
            if not endpoint_calls:
                return {"error": f"No {endpoint_type} calls recorded"}

            recent_calls = endpoint_calls[-limit:] if len(
                endpoint_calls) > limit else endpoint_calls

            # Calculate statistics
            successful_calls = [call for call in recent_calls if call.success]
            failed_calls = [call for call in recent_calls if not call.success]

            durations = [call.duration_ms for call in recent_calls]
            input_tokens = [call.input_tokens for call in recent_calls]
            output_tokens = [call.output_tokens for call in recent_calls]
            total_tokens = [call.total_tokens for call in recent_calls]

            return {
                "endpoint_type": endpoint_type,
                "total_calls": len(recent_calls),
                "successful_calls": len(successful_calls),
                "failed_calls": len(failed_calls),
                "success_rate": len(successful_calls) /
                len(recent_calls) *
                100 if recent_calls else 0,
                "timing_metrics": {
                    "average_duration_ms": statistics.mean(durations) if durations else 0,
                    "median_duration_ms": statistics.median(durations) if durations else 0,
                    "min_duration_ms": min(durations) if durations else 0,
                    "max_duration_ms": max(durations) if durations else 0,
                    "p95_duration_ms": self._percentile(
                        durations,
                        95) if durations else 0,
                    "p99_duration_ms": self._percentile(
                        durations,
                        99) if durations else 0},
                "token_metrics": {
                    "average_input_tokens": statistics.mean(input_tokens) if input_tokens else 0,
                    "average_output_tokens": statistics.mean(output_tokens) if output_tokens else 0,
                    "average_total_tokens": statistics.mean(total_tokens) if total_tokens else 0,
                    "total_input_tokens": sum(input_tokens),
                    "total_output_tokens": sum(output_tokens),
                    "total_tokens_used": sum(total_tokens)},
                "error_analysis": {
                    "most_common_error": self._most_common_error(failed_calls),
                    "error_rate": len(failed_calls) /
                    len(recent_calls) *
                    100 if recent_calls else 0},
                "performance_trend": self._calculate_performance_trend(recent_calls)}

    def get_request_pipeline_stats(self, limit: int = 1000) -> Dict[str, Any]:
        """Get comprehensive request pipeline statistics."""
        with self._lock:
            recent_requests = self._completed_requests[-limit:] if len(
                self._completed_requests) > limit else self._completed_requests

            if not recent_requests:
                return {"error": "No completed requests found"}

            # Extract metrics
            token_savings = [req.net_token_savings for req in recent_requests]
            roi_scores = [req.roi_score for req in recent_requests]
            request_times = [req.total_request_time_ms for req in recent_requests]
            memory_overheads = [req.memory_overhead_ms for req in recent_requests]

            positive_savings = [s for s in token_savings if s > 0]
            negative_savings = [s for s in token_savings if s < 0]

            return {
                "pipeline_overview": {
                    "total_requests_analyzed": len(recent_requests),
                    "analysis_period_hours": (recent_requests[-1].timestamp - recent_requests[0].timestamp) / 3600 if len(recent_requests) > 1 else 0,
                    "average_requests_per_hour": len(recent_requests) / max(1, (recent_requests[-1].timestamp - recent_requests[0].timestamp) / 3600)
                },

                "token_economics": {
                    "total_tokens_saved": sum(s for s in token_savings if s > 0),
                    "total_tokens_wasted": sum(abs(s) for s in token_savings if s < 0),
                    "net_token_change": sum(token_savings),
                    "average_token_savings": statistics.mean(token_savings) if token_savings else 0,
                    "median_token_savings": statistics.median(token_savings) if token_savings else 0,
                    "savings_volatility": statistics.stdev(token_savings) if len(token_savings) > 1 else 0,
                    "success_rate": len(positive_savings) / len(token_savings) * 100 if token_savings else 0,
                    "token_efficiency_ratio": sum(s for s in token_savings if s > 0) / max(sum(abs(s) for s in token_savings), 1) if token_savings else 0
                },

                "performance_metrics": {
                    "average_request_time_ms": statistics.mean(request_times) if request_times else 0,
                    "median_request_time_ms": statistics.median(request_times) if request_times else 0,
                    "p95_request_time_ms": self._percentile(request_times, 95) if request_times else 0,
                    "average_memory_overhead_ms": statistics.mean(memory_overheads) if memory_overheads else 0,
                    "memory_overhead_percentage": statistics.mean(memory_overheads) / statistics.mean(request_times) * 100 if request_times and memory_overheads else 0
                },

                "business_impact": {
                    "average_roi_score": statistics.mean(roi_scores) if roi_scores else 0,
                    "high_roi_requests": len([r for r in roi_scores if r > 0.7]),
                    "positive_roi_percentage": len([r for r in roi_scores if r > 0.5]) / len(roi_scores) * 100 if roi_scores else 0,
                    "business_value_distribution": {
                        "excellent": len([r for r in roi_scores if r > 0.8]),
                        "good": len([r for r in roi_scores if 0.6 < r <= 0.8]),
                        "moderate": len([r for r in roi_scores if 0.4 < r <= 0.6]),
                        "poor": len([r for r in roi_scores if r <= 0.4])
                    }
                },

                "optimization_insights": self._generate_optimization_insights(recent_requests)
            }

    def _find_endpoint_metrics(self, call_id: str) -> Optional[EndpointMetrics]:
        """Find endpoint metrics by call ID."""
        # Search in caches
        for endpoint_calls in self._endpoint_cache.values():
            for call in endpoint_calls:
                if call.call_id == call_id:
                    return call
        return None

    def _estimate_baseline_tokens(self, query: str, query_tokens: int) -> int:
        """Estimate baseline tokens without memory system."""
        # Simple heuristic: baseline would be query + estimated response
        # Typically response is 2-3x query length without memory context
        baseline_multiplier = 2.5
        return int(query_tokens * baseline_multiplier)

    def _calculate_roi_score(self, request_metrics: RequestMetrics) -> float:
        """Calculate ROI score for a request (0-1 scale)."""
        # Token efficiency component
        if request_metrics.baseline_tokens_estimate > 0:
            token_efficiency = max(0, request_metrics.net_token_savings) / \
                request_metrics.baseline_tokens_estimate
        else:
            token_efficiency = 0

        # Time efficiency component
        if request_metrics.total_request_time_ms > 0:
            time_efficiency = max(0, 1 -
                                  (request_metrics.memory_overhead_ms /
                                   request_metrics.total_request_time_ms))
        else:
            time_efficiency = 1

        # Memory relevance component
        if request_metrics.memory_relevance_scores:
            relevance_score = statistics.mean(request_metrics.memory_relevance_scores)
        else:
            relevance_score = 0.5

        # Weighted combination
        roi_score = (
            token_efficiency * 0.5 +  # 50% weight on token efficiency
            time_efficiency * 0.3 +   # 30% weight on time efficiency
            relevance_score * 0.2       # 20% weight on relevance
        )

        return min(1.0, max(0.0, roi_score))

    def calculate_enhanced_scores(self, request_metrics: RequestMetrics) -> None:
        """Calculate enhanced scoring metrics for a request."""
        if not self._scoring_systems_initialized:
            return

        try:
            # Import scoring systems locally to avoid circular import issues
            from ..evaluation.memory_scorer import MemoryScorer
            from ..evaluation.response_scorer import ResponseScorer
            from ..evaluation.token_analyzer import TokenAnalyzer

            # Prepare request data for scoring
            # Calculate memory tokens from memory calls
            memory_tokens = sum(
                call.total_tokens for call in request_metrics.memory_calls) if request_metrics.memory_calls else 0

            request_data = {
                'original_query': request_metrics.original_query,
                'response_content': getattr(request_metrics, 'response_content', ''),
                'memories_injected': getattr(request_metrics, 'memories_injected_data', []),
                'total_tokens_used': request_metrics.total_tokens_used,
                'memory_tokens': memory_tokens,
                'total_request_time_ms': request_metrics.total_request_time_ms
            }

            # Calculate memory relevance scores if memories were retrieved
            if hasattr(
                    request_metrics,
                    'memory_relevance_scores') and request_metrics.memory_relevance_scores:
                avg_relevance = sum(request_metrics.memory_relevance_scores) / \
                    len(request_metrics.memory_relevance_scores)
                request_metrics.memory_relevance_score = avg_relevance
            else:
                request_metrics.memory_relevance_score = 0.0

            # Calculate response quality scores
            if self.response_scorer:
                response_scores = self.response_scorer.score_response_quality(request_data)
                request_metrics.response_quality_score = response_scores['overall_score']
                request_metrics.response_relevance = response_scores['relevance']
                request_metrics.response_coherence = response_scores['coherence']
                request_metrics.memory_utilization = response_scores['memory_utilization']

            # Calculate token efficiency scores
            if self.token_analyzer:
                efficiency_metrics = self.token_analyzer.calculate_efficiency_metrics(request_data)
                request_metrics.token_efficiency_score = efficiency_metrics['efficiency_score']
                request_metrics.enhanced_net_token_savings = float(
                    efficiency_metrics['net_savings'])

        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to calculate enhanced scores: {e}")

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def _most_common_error(self, failed_calls: List[EndpointMetrics]) -> Optional[str]:
        """Find most common error code."""
        if not failed_calls:
            return None

        error_counts = defaultdict(int)
        for call in failed_calls:
            error_key = call.error_code or "unknown_error"
            error_counts[error_key] += 1

        if error_counts:
            return max(error_counts.items(), key=lambda x: x[1])[0]
        return None

    def _calculate_performance_trend(self, calls: List[EndpointMetrics]) -> str:
        """Calculate performance trend from recent calls."""
        if len(calls) < 10:
            return "insufficient_data"

        # Split into halves
        mid_point = len(calls) // 2
        first_half = calls[:mid_point]
        second_half = calls[mid_point:]

        first_avg = statistics.mean([call.duration_ms for call in first_half])
        second_avg = statistics.mean([call.duration_ms for call in second_half])

        change = (second_avg - first_avg) / max(abs(first_avg), 1)

        if change > 0.1:
            return "degrading"
        elif change < -0.1:
            return "improving"
        else:
            return "stable"

    def _generate_optimization_insights(self, requests: List[RequestMetrics]) -> List[str]:
        """Generate optimization insights from request data."""
        insights = []

        if not requests:
            return insights

        # Token efficiency insights
        token_savings = [req.net_token_savings for req in requests]
        avg_savings = statistics.mean(token_savings) if token_savings else 0

        if avg_savings < 0:
            insights.append("CRITICAL: Memory system increasing token usage")
        elif avg_savings < 50:
            insights.append("OPTIMIZE: Low token savings - improve memory relevance")

        # Time efficiency insights
        memory_overheads = [req.memory_overhead_ms for req in requests]
        avg_overhead = statistics.mean(memory_overheads) if memory_overheads else 0

        if avg_overhead > 500:  # > 500ms
            insights.append("PERFORMANCE: High memory overhead - optimize retrieval speed")

        # ROI insights
        roi_scores = [req.roi_score for req in requests]
        avg_roi = statistics.mean(roi_scores) if roi_scores else 0

        if avg_roi < 0.4:
            insights.append("BUSINESS: Low ROI - review memory system effectiveness")
        elif avg_roi > 0.8:
            insights.append("EXCELLENT: High ROI - consider increasing memory usage")

        return insights

    def _save_metrics_async(self):
        """Save metrics to disk asynchronously."""
        try:
            # Save endpoint-specific metrics
            self._save_endpoint_metrics()

            # Save request pipeline metrics
            self._save_request_pipeline_metrics()

        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def _save_endpoint_metrics(self):
        """Save endpoint-specific metrics to separate files."""
        # Prepare endpoint-specific data
        upstream_data = []
        memory_data = []
        embedding_data = []

        for endpoint_calls in self._endpoint_cache.values():
            for call in endpoint_calls:
                call_dict = {
                    "call_id": call.call_id,
                    "timestamp": call.timestamp,
                    "duration_ms": call.duration_ms,
                    "input_tokens": call.input_tokens,
                    "output_tokens": call.output_tokens,
                    "total_tokens": call.total_tokens,
                    "success": call.success,
                    "error_code": call.error_code,
                    "error_message": call.error_message,
                    "model": call.model,
                    "temperature": call.temperature,
                    "request_type": call.request_type
                }

                if call.endpoint_type == 'upstream':
                    upstream_data.append(call_dict)
                elif call.endpoint_type == 'memory':
                    memory_data.append(call_dict)
                elif call.endpoint_type == 'embedding':
                    embedding_data.append(call_dict)

        # Save to separate files
        self._save_json_file(self.upstream_metrics_file, upstream_data)
        self._save_json_file(self.memory_metrics_file, memory_data)
        self._save_json_file(self.embedding_metrics_file, embedding_data)

    def _save_request_pipeline_metrics(self):
        """Save complete request pipeline metrics."""
        pipeline_data = []

        for request in self._completed_requests:
            request_dict = {
                "request_id": request.request_id,
                "timestamp": request.timestamp,
                "original_query": request.original_query,
                "query_tokens": request.query_tokens,

                "endpoint_calls": {
                    "upstream": {
                        "call_id": request.upstream_call.call_id if request.upstream_call else None,
                        "duration_ms": request.upstream_call.duration_ms if request.upstream_call else 0,
                        "total_tokens": request.upstream_call.total_tokens if request.upstream_call else 0,
                        "success": request.upstream_call.success if request.upstream_call else False
                    } if request.upstream_call else None,

                    "memory": [
                        {
                            "call_id": call.call_id,
                            "duration_ms": call.duration_ms,
                            "total_tokens": call.total_tokens,
                            "success": call.success
                        } for call in request.memory_calls
                    ] if request.memory_calls else [],

                    "embedding": [
                        {
                            "call_id": call.call_id,
                            "duration_ms": call.duration_ms,
                            "total_tokens": call.total_tokens,
                            "success": call.success
                        } for call in request.embedding_calls
                    ] if request.embedding_calls else []
                },

                "aggregate_metrics": {
                    "total_tokens_used": request.total_tokens_used,
                    "baseline_tokens_estimate": request.baseline_tokens_estimate,
                    "net_token_savings": request.net_token_savings,
                    "total_request_time_ms": request.total_request_time_ms,
                    "memory_overhead_ms": request.memory_overhead_ms,
                    "upstream_time_ms": request.upstream_time_ms,
                    "memories_retrieved": request.memories_retrieved,
                    "memories_injected": request.memories_injected,
                    "memory_relevance_scores": request.memory_relevance_scores,
                    "business_value_score": request.business_value_score,
                    "roi_score": request.roi_score
                }
            }
            pipeline_data.append(request_dict)

        self._save_json_file(self.request_pipeline_file, pipeline_data)

    def _save_json_file(self, file_path: Path, data: List[Dict]):
        """Save data to JSON file with rotation."""
        try:
            # Load existing data
            if file_path.exists():
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []

            # Append new data and limit size
            existing_data.extend(data)
            if len(existing_data) > 5000:  # Keep last 5000 records
                existing_data = existing_data[-2500:]

            # Save updated data
            with open(file_path, 'w') as f:
                json.dump(existing_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving metrics to {file_path}: {e}")


# Global instance
_endpoint_metrics_collector: Optional[EndpointMetricsCollector] = None


def get_endpoint_metrics_collector(config: Optional[Any] = None) -> EndpointMetricsCollector:
    """Get the global endpoint metrics collector instance."""
    global _endpoint_metrics_collector
    if _endpoint_metrics_collector is None:
        if config and hasattr(config, 'data_dir'):
            data_dir = config.data_dir
        else:
            data_dir = os.getenv('MEMEVOLVE_DATA_DIR', './data')
        _endpoint_metrics_collector = EndpointMetricsCollector(data_dir)
    return _endpoint_metrics_collector
