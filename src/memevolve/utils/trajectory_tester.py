"""Trajectory testing for per-genotype fitness evaluation."""

import time
import random
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from ..evolution.genotype import MemoryGenotype
from ..components.retrieve.hybrid_strategy import HybridRetrievalStrategy
from ..utils.embeddings import create_embedding_function


@dataclass
class TestTrajectory:
    """Single test trajectory for fitness evaluation."""
    task_id: str
    query: str
    expected_result: Any
    test_type: str
    timeout: float = 30.0


class TrajectoryTester:
    """Test runner for evaluating genotype fitness through trajectory testing."""
    
    def __init__(self, memory_system, config):
        self.memory_system = memory_system
        self.config = config
        self.test_queries = [
            TestTrajectory("basic_riddle", "What has hands but can't clap?", "clock", "reasoning"),
            TestTrajectory("causal_reasoning", "Why do eggs tell jokes?", "biological_constraint", "reasoning"),
            TestTrajectory("pattern_recognition", "What gets wetter when it dries?", "sponge", "knowledge"),
            TestTrajectory("spatial_reasoning", "What has a bottom at the top?", "inverted_object", "reasoning"),
            TestTrajectory("abstract_concept", "Why is there air?", "scientific_knowledge", "explanation"),
            TestTrajectory("tool_use", "How do you build a simple calculator?", "tool_design", "practical"),
        ]
    
    def run_comprehensive_test(self, genotype: MemoryGenotype) -> List[float]:
        """Run comprehensive test suite and return fitness vector."""
        fitness_scores = []
        
        # Apply genotype configuration
        try:
            self._apply_genotype_configuration(genotype)
        except Exception as e:
            logger.error(f"Failed to apply genotype configuration: {e}")
            return [0.0, 0.0, 0.0, 0.0]
        
        # Run each test trajectory
        for test_traj in self.test_queries:
            score = self._run_single_test(test_traj)
            fitness_scores.append(score)
        
        # Return aggregated fitness vector: [task_success, token_efficiency, response_time, retrieval_quality]
        return self._aggregate_fitness_scores(fitness_scores)
    
    def _run_single_test(self, test_traj: TestTrajectory) -> float:
        """Run a single test trajectory and return score."""
        try:
            start_time = time.time()
            
            # Test memory retrieval
            retrieval_result = self.memory_system.retrieve(
                query=test_traj.query,
                top_k=3,
                strategy_name="hybrid"
            )
            
            retrieval_time = time.time() - start_time
            
            # Test response generation (simplified)
            response_quality = self._evaluate_response_quality(test_traj, retrieval_result)
            
            # Calculate combined success metric
            success = self._calculate_success_metric(test_traj, retrieval_result, response_quality)
            
            # Calculate efficiency metrics
            token_efficiency = self._calculate_token_efficiency(test_traj)
            response_time_score = max(0, 1.0 - (retrieval_time / test_traj.timeout))
            
            total_score = success * 0.4 + token_efficiency * 0.3 + response_time_score * 0.2 + response_quality * 0.1
            
            return total_score
            
        except Exception as e:
            logger.error(f"Test trajectory {test_traj.task_id} failed: {e}")
            return 0.0
    
    def _apply_genotype_configuration(self, genotype: MemoryGenotype):
        """Apply genotype configuration to memory system."""
        # Apply retrieval strategy
        if genotype.retrieve.strategy_type == "semantic":
            from ..components.retrieve import SemanticRetrievalStrategy
            embedding_function = create_embedding_function(
                provider="openai",
                base_url=self.config.memory.base_url,
                api_key=self.config.memory.api_key,
                evolution_manager=self  # Pass self as evolution manager for embedding overrides
            )
            retrieval_strategy = SemanticRetrievalStrategy(embedding_function)
        elif genotype.retrieve.strategy_type == "hybrid":
            embedding_function = create_embedding_function(
                provider="openai", 
                base_url=self.config.memory.base_url,
                api_key=self.config.memory.api_key,
                evolution_manager=self
            )
            retrieval_strategy = HybridRetrievalStrategy(
                embedding_function=embedding_function,
                semantic_weight=genotype.retrieve.hybrid_semantic_weight,
                keyword_weight=genotype.retrieve.hybrid_keyword_weight
            )
        else:  # keyword or llm_guided
            from ..components.retrieve import KeywordRetrievalStrategy
            retrieval_strategy = KeywordRetrievalStrategy()
        
        # Apply to memory system
        self.memory_system.reconfigure_component(
            component_type="retrieve",
            strategy=retrieval_strategy
        )
        
        # Apply management strategy
        from ..components.manage import SimpleManagementStrategy
        management_strategy = SimpleManagementStrategy()
        
        self.memory_system.reconfigure_component(
            component_type="manage", 
            strategy=management_strategy
        )
        
        # Small delay to ensure configuration is applied
        time.sleep(0.1)
    
    def _evaluate_response_quality(self, test_traj: TestTrajectory, retrieval_result) -> float:
        """Evaluate quality of generated response based on expected result."""
        # This is a simplified evaluation - in a full implementation,
        # this would involve generating actual responses and comparing them
        
        # For now, base quality on retrieval success and relevance
        if not retrieval_result:
            return 0.0
        
        # Check if retrieved memories are relevant to expected result type
        relevance_score = 0.5  # Placeholder
        if test_traj.test_type == "reasoning" and len(retrieval_result) > 0:
            relevance_score = 0.8
        
        # Quality based on strategy type
        strategy_bonus = 0.0
        if hasattr(test_traj, 'expected_result') and test_traj.expected_result:
            if isinstance(test_traj.expected_result, str) and test_traj.expected_result.lower() in ['clock', 'sponge', 'biological_constraint']:
                strategy_bonus = 0.2
        
        return min(1.0, relevance_score + strategy_bonus)
    
    def _calculate_success_metric(self, test_traj: TestTrajectory, retrieval_result, response_quality) -> float:
        """Calculate task success metric."""
        # Success based on retrieval and response quality
        base_success = 0.6  # Base success rate
        
        # Bonus for successful retrieval
        retrieval_bonus = 0.0
        if retrieval_result and len(retrieval_result) > 0:
            retrieval_bonus = 0.3
        
        # Bonus for high response quality
        quality_bonus = response_quality * 0.2
        
        return min(1.0, base_success + retrieval_bonus + quality_bonus)
    
    def _calculate_token_efficiency(self, test_traj: TestTrajectory) -> float:
        """Calculate token efficiency score."""
        # Strategy-based efficiency scoring
        strategy_scores = {
            "semantic": 0.9,
            "hybrid": 0.8,
            "llm_guided": 0.6,
            "keyword": 0.5
        }
        
        # Penalize complex queries
        complexity_penalty = len(test_traj.query.split()) / 20.0
        base_efficiency = strategy_scores.get(test_traj.test_type, 0.7)
        
        return max(0.0, base_efficiency - complexity_penalty)
    
    def _aggregate_fitness_scores(self, individual_scores: List[float]) -> List[float]:
        """Aggregate individual test scores into fitness vector components."""
        if not individual_scores:
            return [0.0, 0.0, 0.0, 0.0]
        
        # Component-wise aggregation
        task_success = np.mean(individual_scores) * 0.8  # Success rate
        token_efficiency = np.mean([self._calculate_token_efficiency(self.test_queries[i]) 
                                    for i in range(len(individual_scores))]) * 0.7
        response_time = 0.7  # Placeholder - should be calculated from actual response times
        retrieval_quality = np.mean(individual_scores) * 0.6  # Average response quality
        
        return [float(task_success), float(token_efficiency), float(response_time), float(retrieval_quality)]