from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import time

import requests
import logging
from .genotype import MemoryGenotype


@dataclass
class FitnessMetrics:
    """Fitness metrics for evaluating memory architectures."""

    performance: float = 0.0
    cost: float = 0.0
    retrieval_accuracy: float = 0.0
    storage_efficiency: float = 0.0
    response_time: float = 0.0
    memory_size_mb: float = 0.0

    def calculate_fitness_score(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate overall fitness score from metrics.

        Args:
            weights: Optional weights for each metric. If None, uses defaults.

        Returns:
            Combined fitness score (higher is better)
        """
        if weights is None:
            weights = {
                "performance": 0.4,
                "cost": -0.3,
                "retrieval_accuracy": 0.2,
                "storage_efficiency": 0.1
            }

        score = (
            weights.get("performance", 0) * self.performance +
            weights.get("cost", 0) * self.cost +
            weights.get("retrieval_accuracy", 0) * self.retrieval_accuracy +
            weights.get("storage_efficiency", 0) * self.storage_efficiency
        )

        return score


@dataclass
class EvaluationResult:
    """Result of evaluating a memory architecture."""

    genotype: MemoryGenotype
    fitness_metrics: FitnessMetrics
    fitness_score: float
    is_pareto_optimal: bool = False
    dominated_by: Optional[List[int]] = None
    dominates_list: List[int] = field(default_factory=list)

    def dominates(self, other: 'EvaluationResult') -> bool:
        """Check if this result dominates another.

        A result dominates if it's better or equal in all objectives
        and strictly better in at least one.
        """
        self_is_better = (
            self.fitness_metrics.performance >=
            other.fitness_metrics.performance and
            self.fitness_metrics.retrieval_accuracy >=
            other.fitness_metrics.retrieval_accuracy and
            self.fitness_metrics.storage_efficiency >=
            other.fitness_metrics.storage_efficiency and
            self.fitness_metrics.response_time <=
            other.fitness_metrics.response_time and
            self.fitness_metrics.memory_size_mb <=
            other.fitness_metrics.memory_size_mb
        )

        self_is_strictly_better = (
            self.fitness_metrics.performance >
            other.fitness_metrics.performance or
            self.fitness_metrics.retrieval_accuracy >
            other.fitness_metrics.retrieval_accuracy or
            self.fitness_metrics.storage_efficiency >
            other.fitness_metrics.storage_efficiency or
            self.fitness_metrics.response_time <
            other.fitness_metrics.response_time or
            self.fitness_metrics.memory_size_mb <
            other.fitness_metrics.memory_size_mb
        )

        return self_is_better and self_is_strictly_better


class ParetoSelector:
    """Selects best genotypes using Pareto ranking."""

    def __init__(
        self,
        performance_weight: float = 0.4,
        cost_weight: float = 0.3
    ):
        self.performance_weight = performance_weight
        self.cost_weight = cost_weight

    def evaluate(
        self,
        genotypes: List[MemoryGenotype],
        performance_data: Optional[Dict[str, float]] = None,
        cost_data: Optional[Dict[str, float]] = None
    ) -> List['EvaluationResult']:
        """Evaluate all genotypes and assign fitness scores.

        Args:
            genotypes: List of genotypes to evaluate
            performance_data: Optional dict mapping genome_id to performance scores
            cost_data: Optional dict mapping genome_id to cost scores

        Returns:
            List of evaluation results
        """
        if performance_data is None:
            performance_data = {}
        if cost_data is None:
            cost_data = {}

        results = []

        for genotype in genotypes:
            genome_id = genotype.get_genome_id()

            fitness_metrics = self._calculate_fitness_metrics(
                genotype,
                genome_id,
                performance_data,
                cost_data
            )

            fitness_score = fitness_metrics.calculate_fitness_score({
                "performance": self.performance_weight,
                "cost": -self.cost_weight
            })

            result = EvaluationResult(
                genotype=genotype,
                fitness_metrics=fitness_metrics,
                fitness_score=fitness_score
            )

            results.append(result)

        return results

    def _calculate_fitness_metrics(
        self,
        genotype: MemoryGenotype,
        genome_id: str,
        performance_data: Dict[str, float],
        cost_data: Dict[str, float]
    ) -> FitnessMetrics:
        """Calculate fitness metrics for a genotype.

        Performance score (higher is better):
        - Retrieved by evaluation on benchmark tasks

        Cost score (lower is better):
        - Inverse of storage size
        - Inverse of response time
        - Inverse of retrieval time

        Combining these into a single fitness score allows Pareto optimization.
        """
        # Always run trajectory testing for real fitness evaluation
        trajectory_results = self._run_test_trajectories(genotype)
        performance = trajectory_results["success_rate"]
        retrieval_accuracy = trajectory_results["retrieval_accuracy"]
        avg_response_time = trajectory_results["avg_response_time"]

        # Only use external data if explicitly provided for specific genome_id
        if performance_data and genome_id in performance_data:
            performance = performance_data[genome_id]
        if cost_data and genome_id in cost_data:
            avg_response_time = cost_data[genome_id]

        cost = self._calculate_cost(genotype, cost_data.get(genome_id, avg_response_time))

        storage_efficiency = self._calculate_storage_efficiency(genotype)

        # Use trajectory testing response time, override with external data if provided
        response_time = trajectory_results["avg_response_time"]
        if cost_data and genome_id in cost_data:
            response_time = cost_data[genome_id]

        memory_size_mb = cost_data.get(genome_id, 10.0)

        return FitnessMetrics(
            performance=performance,
            cost=cost,
            retrieval_accuracy=retrieval_accuracy,
            storage_efficiency=storage_efficiency,
            response_time=response_time,
            memory_size_mb=memory_size_mb
        )

    def _calculate_cost(
        self,
        genotype: MemoryGenotype,
        base_cost: float
    ) -> float:
        """Calculate combined cost metric.

        Lower is better. Factors:
        - Storage backend complexity
        - Vector storage overhead
        - Retrieval strategy complexity
        - Management overhead
        """
        backend_cost = self._get_backend_cost(genotype.store.backend_type)
        strategy_cost = self._get_strategy_cost(
            genotype.retrieve.strategy_type)

        total_cost = base_cost * backend_cost * strategy_cost
        return total_cost

    def _get_backend_cost(self, backend_type: str) -> float:
        """Get cost multiplier for storage backend."""
        costs = {
            "json": 1.0,
            "vector": 1.5,
            "graph": 2.0
        }
        return costs.get(backend_type, 1.0)

    def _get_strategy_cost(self, strategy_type: str) -> float:
        """Get cost multiplier for retrieval strategy."""
        costs = {
            "keyword": 1.0,
            "semantic": 1.3,
            "hybrid": 1.5,
            "llm_guided": 2.0
        }
        return costs.get(strategy_type, 1.0)

    def _calculate_storage_efficiency(
        self,
        genotype: MemoryGenotype
    ) -> float:
        """Calculate storage efficiency score.

        Higher is better. Factors:
        - Consolidation enabled (merges similar memories)
        - Forgetting strategy efficiency
        - Auto-management enabled

        Note: Storage backend parameters (backend_type, persistence, etc.)
        are user-configurable and not evolved.
        """
        efficiency = 0.5

        # Consolidation reduces storage by merging similar memories
        if genotype.manage.consolidate_enabled:
            efficiency += 0.2

        # Auto-management enables pruning and optimization
        if genotype.manage.enable_auto_management:
            efficiency += 0.15

        # Forgetting strategy affects storage efficiency
        if genotype.manage.forgetting_strategy in ["lru", "cost_based"]:
            efficiency += 0.15
        elif genotype.manage.forgetting_strategy == "lfu":
            efficiency += 0.1

        return min(max(efficiency, 0.0), 1.0)

    def select_pareto_front(
        self,
        results: List['EvaluationResult']
    ) -> List['EvaluationResult']:
        """Select Pareto front (non-dominated solutions).

        Returns:
            List of evaluation results that are not dominated
        """
        pareto_front = []

        for i, result in enumerate(results):
            is_dominated = False
            dominated_by_indices = []

            for j, other in enumerate(results):
                if i == j:
                    continue

                if other.dominates(result):
                    is_dominated = True
                    dominated_by_indices.append(j)

            if not is_dominated:
                result.dominated_by = dominated_by_indices
                result.dominates_list = []
                pareto_front.append(result)

        return pareto_front

    def select_best(
        self,
        results: List['EvaluationResult']
    ) -> EvaluationResult:
        """Select single best result by fitness score.

        Returns:
            Best evaluation result
        """
        if not results:
            raise ValueError("No results to select from")

        best = max(results, key=lambda r: r.fitness_score)

        for result in results:
            result.dominated_by = []
            result.dominates_list = []
        best.is_pareto_optimal = True

        return best

    def select_top_n(
        self,
        results: List['EvaluationResult'],
        n: int = 10
    ) -> List['EvaluationResult']:
        """Select top N results by fitness score.

        Returns:
            List of top N evaluation results
        """
        sorted_results = sorted(
            results,
            key=lambda r: r.fitness_score,
            reverse=True
        )

        for i, result in enumerate(sorted_results):
            result.dominated_by = []
            result.dominates_list = []

        return sorted_results[:n]

    def _run_test_trajectories(
        self,
        genotype: MemoryGenotype,
        test_queries: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Run real performance tests on configured endpoints.

        Makes actual API calls to measure fitness based on real endpoint performance.
        This replaces hardcoded placeholder values with genuine trajectory testing.

        Args:
            genotype: The memory architecture to test
            test_queries: Optional list of test queries, uses defaults if None

        Returns:
            Dict with performance metrics:
            - success_rate: API call success ratio (0-1)
            - avg_response_time: Average response time in seconds
            - token_efficiency: Tokens processed per second
            - retrieval_accuracy: Memory retrieval relevance score
        """
        logger = logging.getLogger(__name__)

        # Default test queries if none provided
        if test_queries is None:
            test_queries = [
                "What are the key principles of effective memory management?",
                "How do you optimize retrieval performance in vector databases?",
                "What strategies work best for consolidating similar memories?",
                "Describe the evolution process for memory architectures.",
                "How do you balance storage efficiency with recall accuracy?"
            ]

        # Initialize metrics
        results = {
            "success_rate": 0.0,
            "avg_response_time": 0.0,
            "token_efficiency": 0.0,
            "retrieval_accuracy": 0.0
        }

        # Load environment configuration for endpoints
        import os
        upstream_url = os.getenv("MEMEVOLVE_UPSTREAM_BASE_URL")
        memory_url = os.getenv("MEMEVOLVE_MEMORY_BASE_URL")
        embedding_url = os.getenv("MEMEVOLVE_EMBEDDING_BASE_URL")

        if not upstream_url:
            logger.warning("Missing MEMEVOLVE_UPSTREAM_BASE_URL for trajectory testing")
            return results
        # Test trajectory execution
        successful_calls = 0
        total_response_time = 0.0
        total_tokens = 0
        retrieval_scores = []

        for query in test_queries:
            try:
                start_time = time.time()

                # Test 1: Upstream API call (primary performance metric)
                upstream_response = self._test_upstream_endpoint(upstream_url, query, genotype)
                call_time = time.time() - start_time

                if upstream_response.get("success"):
                    successful_calls += 1
                    total_response_time += call_time
                    total_tokens += upstream_response.get("tokens_used", 0)

                    # Test 2: Memory retrieval (if memory URL available and responsive)
                    if memory_url:
                        try:
                            retrieval_score = self._test_memory_retrieval(memory_url, query)
                            retrieval_scores.append(retrieval_score)
                        except Exception:
                            # Memory API might be down/slow, skip to avoid timeouts
                            pass

                    # Test 3: Embedding generation (if embedding URL available)
                    if embedding_url:
                        try:
                            embedding_result = self._test_embedding_generation(embedding_url, query)
                            if embedding_result.get("success"):
                                total_tokens += embedding_result.get("tokens_used", 0)
                        except Exception:
                            # Embedding API might be down/slow, skip to avoid timeouts
                            pass

            except Exception as e:
                logger.debug(f"Trajectory test failed for query '{query}': {e}")
                continue

        # Calculate final metrics
        total_tests = len(test_queries)
        if total_tests > 0:
            results["success_rate"] = successful_calls / total_tests

            if successful_calls > 0:
                results["avg_response_time"] = total_response_time / successful_calls

                if total_tokens > 0:
                    results["token_efficiency"] = total_tokens / \
                        total_response_time if total_response_time > 0 else 0

                if retrieval_scores:
                    results["retrieval_accuracy"] = sum(retrieval_scores) / len(retrieval_scores)

        logger.debug(f"Trajectory results for {genotype.get_genome_id()}: {results}")
        return results

    def _test_upstream_endpoint(self, base_url: str, query: str,
                                genotype: MemoryGenotype) -> Dict[str, Any]:
        """Test upstream LLM endpoint performance."""
        try:
            # Resolve available models first
            models_url = f"{base_url.rstrip('/')}/models"
            models_response = requests.get(models_url, timeout=10)

            if models_response.status_code != 200:
                return {"success": False, "error": "Models endpoint failed"}

            models_data = models_response.json()
            model_name = None

            # Extract first available model
            if "data" in models_data and len(models_data["data"]) > 0:
                model_name = models_data["data"][0].get("id", "default")
            elif isinstance(models_data, list) and len(models_data) > 0:
                model_name = models_data[0].get("id", "default")
            else:
                model_name = "default"

            # Test chat completion with genotype-specific parameters
            chat_url = f"{base_url.rstrip('/')}/chat/completions"
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": query}],
                "max_tokens": genotype.encode.max_tokens,
                "temperature": genotype.encode.temperature
            }

            response = requests.post(chat_url, json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {})
                tokens_used = usage.get("total_tokens", 0)

                return {
                    "success": True,
                    "tokens_used": tokens_used,
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _test_memory_retrieval(self, base_url: str, query: str) -> float:
        """Test memory retrieval endpoint relevance score."""
        try:
            # Test memory retrieval if available
            retrieve_url = f"{base_url.rstrip('/')}/memory/search"
            payload = {
                "query": query,
                "limit": 3
            }

            response = requests.post(retrieve_url, json=payload, timeout=15)

            if response.status_code == 200:
                data = response.json()
                memories = data.get("memories", [])

                # Simple relevance scoring based on keyword matches
                if memories:
                    query_words = set(query.lower().split())
                    relevance_scores = []

                    for memory in memories:
                        content = memory.get("content", "").lower()
                        memory_words = set(content.split())

                        # Calculate Jaccard similarity
                        intersection = len(query_words & memory_words)
                        union = len(query_words | memory_words)

                        if union > 0:
                            relevance = intersection / union
                            relevance_scores.append(relevance)

                    if relevance_scores:
                        return sum(relevance_scores) / len(relevance_scores)

            return 0.0  # Default if no successful retrieval

        except Exception:
            return 0.0

    def _test_embedding_generation(self, base_url: str, query: str) -> Dict[str, Any]:
        """Test embedding generation endpoint performance."""
        try:
            # Resolve available models first
            models_url = f"{base_url.rstrip('/')}/models"
            models_response = requests.get(models_url, timeout=10)

            if models_response.status_code != 200:
                return {"success": False, "error": "Embedding models endpoint failed"}

            models_data = models_response.json()
            model_name = None

            # Extract first available embedding model
            if "data" in models_data and len(models_data["data"]) > 0:
                model_name = models_data["data"][0].get("id", "default")
            elif isinstance(models_data, list) and len(models_data) > 0:
                model_name = models_data[0].get("id", "default")
            else:
                model_name = "default"

            # Test embedding generation
            embed_url = f"{base_url.rstrip('/')}/embeddings"
            payload = {
                "model": model_name,
                "input": query,
                "encoding_format": "float"
            }

            response = requests.post(embed_url, json=payload, timeout=15)

            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {})
                tokens_used = usage.get("prompt_tokens", 0)

                return {
                    "success": True,
                    "tokens_used": tokens_used,
                    "embedding_dim": len(data.get("data", [{}])[0].get("embedding", []))
                }
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}
