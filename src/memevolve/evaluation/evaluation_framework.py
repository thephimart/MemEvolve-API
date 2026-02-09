"""
MemEvolve Benchmark Evaluation Framework

This module provides comprehensive evaluation capabilities for memory system architectures
across multiple AI agent benchmarks including GAIA, WebWalkerQA, xBench, and TaskCraft.
"""

import json
import logging
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..evolution.genotype import GenotypeFactory, MemoryGenotype
from ...utils.logging_manager import LoggingManager


@dataclass
class BenchmarkResult:
    """Result from running a benchmark with a specific memory architecture."""
    benchmark_name: str
    architecture_name: str
    genotype_id: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    raw_scores: List[float] = field(default_factory=list)
    execution_time: float = 0.0
    memory_usage: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    error_details: List[str] = field(default_factory=list)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this benchmark run."""
        summary = {
            "benchmark": self.benchmark_name,
            "architecture": self.architecture_name,
            "genotype_id": self.genotype_id,
            "execution_time": self.execution_time,
            "error_rate": self.error_count / max(1, len(self.raw_scores)) if self.raw_scores else 0,
        }

        if self.raw_scores:
            summary.update({
                "mean_score": statistics.mean(self.raw_scores),
                "median_score": statistics.median(self.raw_scores),
                "std_dev": statistics.stdev(self.raw_scores) if len(self.raw_scores) > 1 else 0,
                "min_score": min(self.raw_scores),
                "max_score": max(self.raw_scores),
                "sample_size": len(self.raw_scores)
            })

        summary.update(self.metrics)
        return summary


class BenchmarkEvaluator(ABC):
    """Abstract base class for benchmark evaluators."""

    def __init__(self, name: str):
        self.name = name
        self.logger = LoggingManager.get_logger(f"{__name__}.{name}")

    @abstractmethod
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the benchmark dataset."""
        pass

    @abstractmethod
    def evaluate_sample(self, sample: Dict[str, Any], memory_system) -> Dict[str, Any]:
        """Evaluate a single sample using the memory system."""
        pass

    @abstractmethod
    def validate_result(self, result: Dict[str, Any], ground_truth: Any) -> float:
        """Validate and score a result against ground truth."""
        pass

    def run_evaluation(self, memory_system, max_samples: Optional[int] = None) -> BenchmarkResult:
        """Run full evaluation on the benchmark."""
        start_time = time.time()

        dataset = self.load_dataset()
        if max_samples:
            dataset = dataset[:max_samples]

        results = []
        errors = []

        self.logger.info(
            f"Starting evaluation of {len(dataset)} samples on {self.name}")

        for i, sample in enumerate(dataset):
            if i % 10 == 0:
                self.logger.info(f"Processed {i}/{len(dataset)} samples")

            try:
                result = self.evaluate_sample(sample, memory_system)
                score = self.validate_result(result, sample.get(
                    "answer", sample.get("golden_answer")))
                results.append(score)
            except Exception as e:
                self.logger.warning(f"Error evaluating sample {i}: {str(e)}")
                errors.append(str(e))
                results.append(0.0)  # Assign 0 score for failed evaluations

        execution_time = time.time() - start_time

        # Get memory architecture info
        arch_name = getattr(memory_system, 'architecture_name', 'unknown')
        genotype_id = getattr(memory_system, 'genotype_id', 'unknown')

        return BenchmarkResult(
            benchmark_name=self.name,
            architecture_name=arch_name,
            genotype_id=genotype_id,
            raw_scores=results,
            execution_time=execution_time,
            error_count=len(errors),
            error_details=errors
        )


class EvaluationRunner:
    """Manages running evaluations across multiple benchmarks and architectures."""

    def __init__(self):
        self.benchmarks: Dict[str, BenchmarkEvaluator] = {}
        self.logger = LoggingManager.get_logger(__name__)

    def register_benchmark(self, benchmark: BenchmarkEvaluator):
        """Register a benchmark evaluator."""
        self.benchmarks[benchmark.name] = benchmark
        self.logger.info(f"Registered benchmark: {benchmark.name}")

    def get_reference_architectures(self) -> List[Dict[str, Any]]:
        """Get the four reference memory architectures."""
        return [
            {
                "name": "AgentKB",
                "genotype": GenotypeFactory.create_agentkb_genotype(),
                "description": "Static baseline with lesson-based storage"
            },
            {
                "name": "Lightweight",
                "genotype": GenotypeFactory.create_lightweight_genotype(),
                "description": "Trajectory-based with JSON storage and auto-pruning"
            },
            {
                "name": "Riva",
                "genotype": GenotypeFactory.create_riva_genotype(),
                "description": "Agent-centric with vector storage and hybrid retrieval"
            },
            {
                "name": "Cerebra",
                "genotype": GenotypeFactory.create_cerebra_genotype(),
                "description": "Tool distillation with semantic graphs"
            }
        ]

    def run_baseline_evaluation(
            self, max_samples_per_benchmark: Optional[int] = None) -> Dict[str, Any]:
        """Run baseline evaluation across all benchmarks and reference architectures."""
        results = {
            "timestamp": time.time(),
            "benchmarks": list(self.benchmarks.keys()),
            "architectures": [],
            "results": {}
        }

        architectures = self.get_reference_architectures()
        results["architectures"] = [arch["name"] for arch in architectures]

        for arch_info in architectures:
            arch_name = arch_info["name"]
            genotype = arch_info["genotype"]

            self.logger.info(f"Evaluating architecture: {arch_name}")
            results["results"][arch_name] = {}

            # Here we would create a MemorySystem from the genotype
            # For now, we'll create a mock evaluation
            memory_system = self._create_memory_system_from_genotype(
                genotype, arch_name)

            for benchmark_name, benchmark in self.benchmarks.items():
                self.logger.info(f"Running {benchmark_name} on {arch_name}")
                try:
                    benchmark_result = benchmark.run_evaluation(
                        memory_system, max_samples_per_benchmark)
                    results["results"][arch_name][benchmark_name] = benchmark_result.get_summary(
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to run {benchmark_name} on {arch_name}: {str(e)}")
                    results["results"][arch_name][benchmark_name] = {
                        "error": str(e),
                        "benchmark": benchmark_name,
                        "architecture": arch_name
                    }

        return results

    def _create_memory_system_from_genotype(self, genotype: MemoryGenotype, arch_name: str):
        """Create a memory system instance from a genotype."""
        from ..utils.config import load_config
        from .genotype_translator import create_memory_system_from_genotype

        config = load_config()
        memory_system = create_memory_system_from_genotype(genotype, config)

        # Override architecture name for reference architectures using wrapper
        class MemorySystemWithOverride:
            def __init__(self, memory_system, arch_name):
                self._memory_system = memory_system
                self.architecture_name = arch_name
                self.genotype_id = getattr(memory_system, 'genotype_id', 'unknown')

            def __getattr__(self, name):
                return getattr(self._memory_system, name)

        return MemorySystemWithOverride(memory_system, arch_name)

    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save evaluation results to file."""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Results saved to {filepath}")

    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load evaluation results from file."""
        with open(filepath, 'r') as f:
            return json.load(f)
