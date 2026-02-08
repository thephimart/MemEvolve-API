"""
GAIA Benchmark Integration for MemEvolve

GAIA (General AI Assistants) benchmark evaluates AI systems on complex questions
requiring various levels of tooling and autonomy.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evaluation_framework import BenchmarkEvaluator


class GAIAEvaluator(BenchmarkEvaluator):
    """Evaluator for the GAIA benchmark dataset."""

    def __init__(self, data_path: Optional[str] = None, level: str = "all"):
        """
        Initialize GAIA evaluator.

        Args:
            data_path: Path to GAIA dataset. If None, will try to download from HF.
            level: Difficulty level to evaluate ('level1', 'level2', 'level3', or 'all')
        """
        super().__init__("GAIA")
        self.data_path = data_path or os.path.join(os.getcwd(), "data", "gaia")
        self.level = level
        self.dataset = None

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load GAIA dataset from Hugging Face or local cache."""
        if self.dataset is not None:
            return self.dataset

        try:
            # Try to load from Hugging Face
            from datasets import load_dataset

            self.logger.info("Loading GAIA dataset from Hugging Face...")

            if self.level == "all":
                # Load all levels
                dataset = []
                for level in ["2023_level1", "2023_level2", "2023_level3"]:
                    try:
                        level_data = load_dataset(
                            "gaia-benchmark/GAIA",
                            level,
                            split="test",
                            cache_dir=self.data_path
                        )
                        # Convert to list of dicts
                        level_samples = []
                        for sample in level_data:
                            sample_dict = dict(sample)
                            sample_dict["level"] = level.replace("2023_", "")
                            level_samples.append(sample_dict)
                        dataset.extend(level_samples)
                        self.logger.info(
                            f"Loaded {len(level_samples)} samples from {level}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load {level}: {e}")
            else:
                # Load specific level
                config_name = f"2023_{self.level}"
                hf_dataset = load_dataset(
                    "gaia-benchmark/GAIA",
                    config_name,
                    split="test",
                    cache_dir=self.data_path
                )
                dataset = []
                for sample in hf_dataset:
                    sample_dict = dict(sample)
                    sample_dict["level"] = self.level
                    dataset.append(sample_dict)
                self.logger.info(
                    f"Loaded {len(dataset)} samples from {config_name}")

        except ImportError:
            self.logger.error(
                "datasets library not available. Please install with: pip install datasets")
            dataset = self._load_from_local_files()
        except Exception as e:
            self.logger.error(f"Failed to load GAIA dataset: {e}")
            dataset = self._load_from_local_files()

        self.dataset = dataset
        self.logger.info(f"GAIA dataset loaded: {len(dataset)} samples")
        return dataset

    def _load_from_local_files(self) -> List[Dict[str, Any]]:
        """Fallback method to load from local JSON files."""
        dataset = []

        # Look for local GAIA files
        gaia_dir = Path(self.data_path)
        if gaia_dir.exists():
            for json_file in gaia_dir.glob("*.jsonl"):
                try:
                    with open(json_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                sample = json.loads(line.strip())
                                dataset.append(sample)
                except Exception as e:
                    self.logger.warning(f"Failed to load {json_file}: {e}")

        if not dataset:
            self.logger.warning(
                "No GAIA data found locally. Please download from Hugging Face.")
            # Return a minimal mock dataset for testing
            dataset = [
                {
                    "task_id": "mock_1",
                    "Question": "What is 2 + 2?",
                    "Level": "level1",
                    "Final answer": "4",
                    "level": "level1"
                }
            ]

        return dataset

    def evaluate_sample(self, sample: Dict[str, Any], memory_system) -> Dict[str, Any]:
        """Evaluate a single GAIA sample using the memory system."""
        question = sample.get("Question", "")
        task_id = sample.get("task_id", "unknown")

        # Check if there are file attachments
        file_path = sample.get("file_path")
        if file_path and os.path.exists(os.path.join(self.data_path, file_path)):
            # Include file context in the query
            full_path = os.path.join(self.data_path, file_path)
            question += f"\n\nReference file: {full_path}"

        # Use memory system to help answer the question
        # In a real implementation, this would involve an agent loop
        # For now, we'll simulate by querying memory and getting a response

        try:
            # Query memory for relevant information
            memory_results = memory_system.query_memory(
                query=f"Help answer: {question}",
                top_k=5
            )

            # Simulate LLM reasoning with memory context
            context = "\n".join([r.get("content", "") for r in memory_results])

            # This would normally involve calling an LLM with the question and context
            # For now, return a mock result
            result = {
                "question": question,
                "memory_context": context,
                "memory_results_count": len(memory_results),
                "task_id": task_id,
                "level": sample.get("level", "unknown")
            }

        except Exception as e:
            self.logger.error(f"Error evaluating GAIA sample {task_id}: {e}")
            result = {
                "question": question,
                "error": str(e),
                "task_id": task_id,
                "level": sample.get("level", "unknown")
            }

        return result

    def validate_result(self, result: Dict[str, Any], ground_truth: Any) -> float:
        """Validate and score a result against ground truth."""
        if "error" in result:
            return 0.0

        # In a real implementation, this would involve sophisticated answer validation
        # For GAIA, answers can be text, numbers, or complex responses
        # For now, we'll use a simple string matching approach

        predicted_answer = result.get("predicted_answer", "")
        true_answer = str(ground_truth).lower().strip()

        # Simple exact match scoring
        if predicted_answer.lower().strip() == true_answer:
            return 1.0

        # Partial credit for containing the answer
        if true_answer in predicted_answer.lower():
            return 0.5

        return 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded dataset."""
        if not self.dataset:
            self.load_dataset()

        if not self.dataset:
            return {"error": "No dataset loaded"}

        stats = {
            "total_samples": len(self.dataset),
            "levels": {}
        }

        for sample in self.dataset:
            level = sample.get("level", "unknown")
            if level not in stats["levels"]:
                stats["levels"][level] = 0
            stats["levels"][level] += 1

        return stats
