"""
Experiment Runner for MemEvolve Benchmark Evaluation

Automates running memory architecture evaluations across multiple benchmarks.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..evolution.genotype import MemoryGenotype
from ..utils.config import MemEvolveConfig, load_config
from ..utils.logging_manager import LoggingManager
from .evaluation_framework import EvaluationRunner
from .gaia_evaluator import GAIAEvaluator
from .genotype_translator import create_memory_system_from_genotype
from .taskcraft_evaluator import TaskCraftEvaluator
from .webwalkerqa_evaluator import WebWalkerQAEvaluator
from .xbench_evaluator import XBenchEvaluator


class MemEvolveExperimentRunner:
    """Experiment runner for comprehensive MemEvolve evaluation."""

    def __init__(self,
                 genotype_paths: Optional[List[str]] = None,
                 config: Optional[MemEvolveConfig] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize experiment runner.

        Args:
            genotype_paths: List of paths to genotype JSON files to evaluate
            config: MemEvolve configuration object
            output_dir: Override output directory (for testing)
        """
        self.config = config or load_config()

        # Results go to data/evaluations/ unless overridden
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(self.config.data_dir) / "evaluations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load genotype files if provided
        self.genotype_paths = genotype_paths or []
        self.loaded_genotypes = {}
        self._load_genotypes()

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

    def _load_genotypes(self):
        """Load genotype files from provided paths."""
        for path in self.genotype_paths:
            try:
                with open(path, 'r') as f:
                    genotype_data = json.load(f)
                    genotype = MemoryGenotype.from_dict(genotype_data)
                    self.loaded_genotypes[path] = genotype
            except Exception as e:
                self.logger.warning(f"Failed to load genotype from {path}: {e}")

    def _create_memory_system_from_genotype(self, genotype: MemoryGenotype, arch_name: str):
        """Create a real MemorySystem instance from a genotype."""
        return create_memory_system_from_genotype(genotype, self.config)

    def _setup_logging(self):
        """Set up logging for experiments using centralized LoggingManager."""
        self.logger = LoggingManager.get_logger(__name__)

    def run_genotype_experiments(self, max_samples_per_benchmark: Optional[int] = None) -> str:
        """
        Run experiments on loaded genotypes across all benchmarks.

        Args:
            max_samples_per_benchmark: Maximum samples to evaluate per benchmark (None for all)

        Returns:
            Path to results file
        """
        self.logger.info(
            f"Starting genotype experiments with {len(self.loaded_genotypes)} genotypes")

        if not self.loaded_genotypes:
            raise ValueError("No genotypes loaded. Provide genotype_paths to constructor.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "timestamp": timestamp,
            "experiment_type": "genotype_evaluation",
            "genotypes": [],
            "benchmarks": list(self.runner.benchmarks.keys()),
            "results": {}
        }

        # Add genotype information
        for path, genotype in self.loaded_genotypes.items():
            results["genotypes"].append({
                "path": path,
                "genotype_id": genotype.get_genome_id(),
                "name": f"Genotype-{genotype.get_genome_id()[:8]}"
            })
            results["results"][genotype.get_genome_id()] = {}

        # Run evaluation for each genotype
        for genotype_path, genotype in self.loaded_genotypes.items():
            self.logger.info(f"Evaluating genotype: {genotype.get_genome_id()}")

            # Create memory system from genotype (mock for now)
            memory_system = self._create_memory_system_from_genotype(
                genotype, f"Genotype-{genotype.get_genome_id()[:8]}"
            )

            # Run all benchmarks
            for benchmark_name, benchmark in self.runner.benchmarks.items():
                self.logger.info(f"Running {benchmark_name} on {genotype.get_genome_id()[:8]}")
                try:
                    benchmark_result = benchmark.run_evaluation(
                        memory_system, max_samples_per_benchmark)
                    results["results"][genotype.get_genome_id(
                    )][benchmark_name] = benchmark_result.get_summary()
                except Exception as e:
                    self.logger.error(
                        f"Failed to run {benchmark_name} on {
                            genotype.get_genome_id()[
                                :8]}: {e}")
                    results["results"][genotype.get_genome_id()][benchmark_name] = {"error": str(e)}

        # Save results
        results_file = self.output_dir / f"genotype_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Genotype experiments completed. Results saved to {results_file}")

        # Generate summary report
        summary_file = self._generate_summary_report(results, timestamp)
        self.logger.info(f"Summary report saved to {summary_file}")

        return str(results_file)

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
        choices=["baseline", "single", "genotype"],
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
        "--genotype-files",
        nargs="*",
        help="Paths to genotype JSON files (for genotype experiments)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per benchmark"
    )

    args = parser.parse_args()

    if args.experiment_type == "genotype":
        if not args.genotype_files:
            parser.error("--genotype-files required for genotype experiments")
        runner = MemEvolveExperimentRunner(genotype_paths=args.genotype_files)
        results_file = runner.run_genotype_experiments(args.max_samples)
        print(f"Genotype experiments completed. Results: {results_file}")
    else:
        runner = MemEvolveExperimentRunner()

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
