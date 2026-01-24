from memevolve.evaluation.experiment_runner import MemEvolveExperimentRunner
import sys
import pytest
import tempfile
import json
from pathlib import Path

# sys.path.insert(0, 'src')  # No longer needed with package structure


def test_experiment_runner_initialization():
    """Test that experiment runner initializes correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = MemEvolveExperimentRunner(temp_dir)

        # Check that benchmarks are registered
        assert len(runner.runner.benchmarks) > 0
        assert "GAIA" in runner.runner.benchmarks
        assert "WebWalkerQA-en" in runner.runner.benchmarks
        assert "xBench-recruitment" in runner.runner.benchmarks
        assert "TaskCraft-all" in runner.runner.benchmarks


def test_baseline_experiment():
    """Test running baseline experiments."""
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = MemEvolveExperimentRunner(temp_dir)

        # Run with very limited samples for testing
        results_file = runner.run_baseline_experiments(
            max_samples_per_benchmark=2)

        # Check that results file was created
        assert Path(results_file).exists()

        # Load and verify results structure
        with open(results_file, 'r') as f:
            results = json.load(f)

        assert "timestamp" in results
        assert "architectures" in results
        assert "benchmarks" in results
        assert "results" in results

        # Check that we have results for reference architectures
        arch_results = results["results"]
        expected_architectures = ["AgentKB", "Lightweight", "Riva", "Cerebra"]
        for arch in expected_architectures:
            assert arch in arch_results


def test_single_experiment():
    """Test running a single experiment."""
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = MemEvolveExperimentRunner(temp_dir)

        # Run a single experiment
        result = runner.run_single_experiment(
            architecture_name="AgentKB",
            benchmark_name="GAIA",
            max_samples=1
        )

        # Check result structure
        assert "benchmark" in result
        assert "architecture" in result
        assert result["benchmark"] == "GAIA"
        assert result["architecture"] == "AgentKB"


def test_experiment_summary_generation():
    """Test that experiment summary is generated correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = MemEvolveExperimentRunner(temp_dir)

        # Create mock results
        mock_results = {
            "timestamp": 1234567890,
            "architectures": ["AgentKB", "Lightweight"],
            "benchmarks": ["GAIA-all", "TaskCraft-all"],
            "results": {
                "AgentKB": {
                    "GAIA-all": {"mean_score": 0.8, "sample_size": 10},
                    "TaskCraft-all": {"mean_score": 0.7, "sample_size": 5}
                },
                "Lightweight": {
                    "GAIA-all": {"mean_score": 0.9, "sample_size": 10},
                    "TaskCraft-all": {"mean_score": 0.6, "sample_size": 5}
                }
            }
        }

        # Generate summary
        summary_file = runner._generate_summary_report(
            mock_results, "20231201_120000")

        # Check that summary file was created
        assert Path(summary_file).exists()

        # Load and verify summary structure
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        assert "summary" in summary
        assert "architecture_performance" in summary["summary"]
        assert "benchmark_performance" in summary["summary"]

        # Check architecture averages
        arch_perf = summary["summary"]["architecture_performance"]
        assert "AgentKB" in arch_perf
        assert "Lightweight" in arch_perf
        assert abs(arch_perf["AgentKB"]["average_score"] -
                   0.75) < 0.01  # (0.8 + 0.7) / 2
        # (0.9 + 0.6) / 2
        assert abs(arch_perf["Lightweight"]["average_score"] - 0.75) < 0.01


def test_get_reference_architectures():
    """Test getting reference architectures."""
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = MemEvolveExperimentRunner(temp_dir)

        architectures = runner.runner.get_reference_architectures()

        assert len(architectures) == 4
        arch_names = [arch["name"] for arch in architectures]
        assert "AgentKB" in arch_names
        assert "Lightweight" in arch_names
        assert "Riva" in arch_names
        assert "Cerebra" in arch_names

        # Check that each has required fields
        for arch in architectures:
            assert "name" in arch
            assert "genotype" in arch
            assert "description" in arch
