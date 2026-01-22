"""
Experiment Runner for MemEvolve Benchmark Evaluation

Automates running memory architecture evaluations across multiple benchmarks.
"""

import json
import logging
import argparse
import os
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from .evaluation_framework import EvaluationRunner
from .gaia_evaluator import GAIAEvaluator
from .webwalkerqa_evaluator import WebWalkerQAEvaluator
from .xbench_evaluator import XBenchEvaluator
from .taskcraft_evaluator import TaskCraftEvaluator


class MemEvolveExperimentRunner:
    """Experiment runner for comprehensive MemEvolve evaluation."""

    def __init__(self, output_dir: str = "experiments"):
        """
        Initialize experiment runner.

        Args:
            output_dir: Directory to save experiment results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.runner = EvaluationRunner()
        self._setup_benchmarks()
        self._setup_logging()

    def _setup_benchmarks(self):
        """Set up all benchmark evaluators."""
        # GAIA benchmark
        gaia_evaluator = GAIAEvaluator()
        self.runner.register_benchmark(gaia_evaluator)

        # WebWalkerQA benchmarks (English and Chinese)
        webwalkerqa_en = WebWalkerQAEvaluator(language="en")
        webwalkerqa_zh = WebWalkerQAEvaluator(language="zh")
        self.runner.register_benchmark(webwalkerqa_en)
        self.runner.register_benchmark(webwalkerqa_zh)

        # xBench domains
        xbench_recruitment = XBenchEvaluator(domain="recruitment")
        xbench_marketing = XBenchEvaluator(domain="marketing")
        xbench_all = XBenchEvaluator(domain="all")
        self.runner.register_benchmark(xbench_recruitment)
        self.runner.register_benchmark(xbench_marketing)
        self.runner.register_benchmark(xbench_all)

        # TaskCraft task types
        taskcraft_atomic = TaskCraftEvaluator(task_type="atomic")
        taskcraft_multihop = TaskCraftEvaluator(task_type="multihop")
        taskcraft_all = TaskCraftEvaluator(task_type="all")
        self.runner.register_benchmark(taskcraft_atomic)
        self.runner.register_benchmark(taskcraft_multihop)
        self.runner.register_benchmark(taskcraft_all)

    def _setup_logging(self):
        """Set up logging for experiments."""
        experiment_enable = os.getenv('MEMEVOLVE_LOG_EXPERIMENT_ENABLE', 'false').lower() == 'true'
        logs_dir = os.getenv('MEMEVOLVE_LOGS_DIR', './logs')
        experiment_dir = os.getenv('MEMEVOLVE_LOG_EXPERIMENT_DIR', logs_dir)

        handlers = [logging.StreamHandler()]

        if experiment_enable:
            os.makedirs(experiment_dir, exist_ok=True)
            log_file = os.path.join(experiment_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            handlers.insert(0, logging.FileHandler(log_file))

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        self.logger = logging.getLogger(__name__)

    def run_baseline_experiments(self, max_samples_per_benchmark: Optional[int] = None) -> str:
        """
        Run baseline experiments with all reference architectures on all benchmarks.

        Args:
            max_samples_per_benchmark: Maximum samples to evaluate per benchmark (None for all)

        Returns:
            Path to results file
        """
        self.logger.info("Starting MemEvolve baseline experiments")
        self.logger.info(
            f"Max samples per benchmark: {max_samples_per_benchmark}")

        # Run the evaluation
        results = self.runner.run_baseline_evaluation(
            max_samples_per_benchmark)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"baseline_results_{timestamp}.json"

        self.runner.save_results(results, str(results_file))

        self.logger.info(
            f"Baseline experiments completed. Results saved to {results_file}")

        # Generate summary report
        summary_file = self._generate_summary_report(results, timestamp)
        self.logger.info(f"Summary report saved to {summary_file}")

        return str(results_file)

    def run_single_experiment(self, architecture_name: str, benchmark_name: str,
                              max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a single experiment with specific architecture and benchmark.

        Args:
            architecture_name: Name of architecture (AgentKB, Lightweight, Riva, Cerebra)
            benchmark_name: Name of benchmark
            max_samples: Maximum samples to evaluate

        Returns:
            Experiment results
        """
        self.logger.info(
            f"Running single experiment: {architecture_name} on {benchmark_name}")

        architectures = self.runner.get_reference_architectures()
        arch_info = next(
            (a for a in architectures if a["name"] == architecture_name), None)
        if not arch_info:
            raise ValueError(f"Unknown architecture: {architecture_name}")

        benchmark = self.runner.benchmarks.get(benchmark_name)
        if not benchmark:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        # Create memory system from genotype
        memory_system = self._create_memory_system_from_genotype(
            arch_info["genotype"], architecture_name
        )

        # Run evaluation
        result = benchmark.run_evaluation(memory_system, max_samples)

        self.logger.info(
            f"Single experiment completed: {result.get_summary()}")
        return result.get_summary()

    def _create_memory_system_from_genotype(self, genotype, arch_name: str):
        """Create a memory system instance from a genotype."""
        # This would need to be properly implemented to translate genotype to MemorySystemConfig
        # For now, return a mock
        class MockMemorySystem:
            def __init__(self, genotype, arch_name):
                self.architecture_name = arch_name
                self.genotype_id = genotype.get_genome_id() if genotype else "mock"
                self.arch_name = arch_name  # Store for use in query_memory

            def query_memory(self, query: str, top_k: int = 5) -> list:
                """Mock memory query implementation."""
                # Return mock memory results based on architecture
                arch = self.arch_name
                if arch == "AgentKB":
                    return [
                        {"type": "lesson",
                            "content": f"Basic lesson for {query[:30]}"},
                        {"type": "skill",
                            "content": f"Simple skill for {query[:30]}"}
                    ]
                elif arch == "Lightweight":
                    return [
                        {"type": "lesson",
                            "content": f"Trajectory lesson for {query[:30]}"},
                        {"type": "skill",
                            "content": f"Trajectory skill for {query[:30]}"},
                        {"type": "abstraction",
                            "content": f"Trajectory abstraction for {query[:30]}"}
                    ]
                elif arch == "Riva":
                    return [
                        {"type": "lesson",
                            "content": f"Vector lesson for {query[:30]}"},
                        {"type": "skill",
                            "content": f"Vector skill for {query[:30]}"},
                        {"type": "abstraction",
                            "content": f"Vector abstraction for {query[:30]}"},
                        {"type": "tool",
                            "content": f"Vector tool for {query[:30]}"}
                    ]
                elif arch == "Cerebra":
                    return [
                        {"type": "tool",
                            "content": f"Cerebra tool for {query[:30]}"},
                        {"type": "abstraction",
                            "content": f"Cerebra abstraction for {query[:30]}"},
                        {"type": "skill",
                            "content": f"Cerebra skill for {query[:30]}"},
                        {"type": "lesson",
                            "content": f"Cerebra lesson for {query[:30]}"},
                        {"type": "tool",
                            "content": f"Additional Cerebra tool for {query[:30]}"}
                    ]
                else:
                    return [{"type": "lesson", "content": f"Mock result for {query[:30]}"}]

        return MockMemorySystem(genotype, arch_name)

    def _generate_summary_report(self, results: Dict[str, Any], timestamp: str) -> str:
        """Generate a human-readable summary report."""
        summary = {
            "experiment_timestamp": timestamp,
            "summary": {
                "total_architectures": len(results.get("architectures", [])),
                "total_benchmarks": len(results.get("benchmarks", [])),
                "architecture_performance": {},
                "benchmark_performance": {}
            }
        }

        arch_results = results.get("results", {})

        # Calculate per-architecture averages
        for arch_name, benchmarks in arch_results.items():
            arch_scores = []
            for benchmark_name, benchmark_result in benchmarks.items():
                if isinstance(benchmark_result, dict) and "mean_score" in benchmark_result:
                    arch_scores.append(benchmark_result["mean_score"])
                elif isinstance(benchmark_result, dict) and "error" in benchmark_result:
                    arch_scores.append(0.0)  # Failed experiments get 0

            if arch_scores:
                summary["summary"]["architecture_performance"][arch_name] = {
                    "average_score": sum(arch_scores) / len(arch_scores),
                    "benchmarks_completed": len([s for s in arch_scores if s > 0]),
                    "total_benchmarks": len(arch_scores)
                }

        # Calculate per-benchmark averages
        benchmark_averages = {}
        for arch_name, benchmarks in arch_results.items():
            for benchmark_name, benchmark_result in benchmarks.items():
                if benchmark_name not in benchmark_averages:
                    benchmark_averages[benchmark_name] = []

                if isinstance(benchmark_result, dict) and "mean_score" in benchmark_result:
                    benchmark_averages[benchmark_name].append(
                        benchmark_result["mean_score"])
                else:
                    benchmark_averages[benchmark_name].append(0.0)

        for benchmark_name, scores in benchmark_averages.items():
            valid_scores = [s for s in scores if s > 0]
            summary["summary"]["benchmark_performance"][benchmark_name] = {
                "average_score": sum(valid_scores) / len(valid_scores) if valid_scores else 0.0,
                "architectures_completed": len(valid_scores),
                "total_architectures": len(scores)
            }

        # Save summary
        summary_file = self.output_dir / f"experiment_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Also create a text summary
        text_summary_file = self.output_dir / \
            f"experiment_summary_{timestamp}.txt"
        with open(text_summary_file, 'w') as f:
            f.write("MemEvolve Baseline Experiment Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(
                f"Architectures Tested: {len(results.get('architectures', []))}\n")
            f.write(
                f"Benchmarks Tested: {len(results.get('benchmarks', []))}\n\n")

            f.write("Architecture Performance:\n")
            for arch, perf in summary["summary"]["architecture_performance"].items():
                f.write(".3f")
                f.write(
                    f"   (Completed {perf['benchmarks_completed']}/{perf['total_benchmarks']} benchmarks)\n")

            f.write("\nBenchmark Performance:\n")
            for bench, perf in summary["summary"]["benchmark_performance"].items():
                f.write(".3f")
                f.write(
                    f"   (Completed {perf['architectures_completed']}/{perf['total_architectures']} architectures)\n")

        return str(summary_file)


def main():
    """Command-line interface for running experiments."""
    parser = argparse.ArgumentParser(
        description="Run MemEvolve benchmark experiments")
    parser.add_argument(
        "--experiment-type",
        choices=["baseline", "single"],
        default="baseline",
        help="Type of experiment to run"
    )
    parser.add_argument(
        "--architecture",
        choices=["AgentKB", "Lightweight", "Riva", "Cerebra"],
        help="Architecture to test (for single experiments)"
    )
    parser.add_argument(
        "--benchmark",
        help="Benchmark to test (for single experiments)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per benchmark"
    )
    parser.add_argument(
        "--output-dir",
        default="experiments",
        help="Output directory for results"
    )

    args = parser.parse_args()

    runner = MemEvolveExperimentRunner(args.output_dir)

    if args.experiment_type == "baseline":
        results_file = runner.run_baseline_experiments(args.max_samples)
        print(f"Baseline experiments completed. Results: {results_file}")

    elif args.experiment_type == "single":
        if not args.architecture or not args.benchmark:
            parser.error(
                "--architecture and --benchmark required for single experiments")
        result = runner.run_single_experiment(
            args.architecture, args.benchmark, args.max_samples
        )
        print(f"Single experiment result: {result}")

    else:
        parser.error("Invalid experiment type")


if __name__ == "__main__":
    main()
