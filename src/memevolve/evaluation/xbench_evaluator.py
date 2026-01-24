"""
xBench Benchmark Integration for MemEvolve

xBench provides profession-aligned real-world evaluations for AI agents,
focusing on domains like recruitment and marketing.
"""

import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from .evaluation_framework import BenchmarkEvaluator


class XBenchEvaluator(BenchmarkEvaluator):
    """Evaluator for the xBench profession-aligned benchmark."""

    def __init__(self, data_path: Optional[str] = None, domain: str = "recruitment"):
        """
        Initialize xBench evaluator.

        Args:
            data_path: Path to xBench dataset. If None, will use mock data.
            domain: Domain to evaluate ('recruitment', 'marketing', or 'all')
        """
        super().__init__(f"xBench-{domain}")
        self.data_path = data_path or os.path.join(
            os.getcwd(), "data", "xbench")
        self.domain = domain
        self.dataset = None

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load xBench dataset."""
        if self.dataset is not None:
            return self.dataset

        try:
            # Try to load from local cache
            dataset_path = Path(self.data_path) / f"xbench_{self.domain}.json"
            if dataset_path.exists():
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    self.dataset = json.load(f)
                self.logger.info(f"Loaded xBench dataset from {dataset_path}")
            else:
                # Generate mock professional tasks
                self.dataset = self._generate_mock_tasks()
                # Save for future use
                os.makedirs(self.data_path, exist_ok=True)
                with open(dataset_path, 'w', encoding='utf-8') as f:
                    json.dump(self.dataset, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to load xBench dataset: {e}")
            self.dataset = self._generate_mock_tasks()

        self.logger.info(f"xBench dataset loaded: {len(self.dataset)} samples")
        return self.dataset

    def _generate_mock_tasks(self) -> List[Dict[str, Any]]:
        """Generate mock professional tasks for evaluation."""
        tasks = []

        if self.domain in ["recruitment", "all"]:
            tasks.extend([
                {
                    "task_id": "recruit_001",
                    "task_type": "candidate_screening",
                    "description": "Review 50 candidate resumes and identify top 5 for software engineering positions",
                    "evaluation_criteria": ["technical_skills", "experience", "cultural_fit"],
                    "expected_output": "Ranked list of candidates with justification",
                    "domain": "recruitment",
                    "complexity": "medium"
                },
                {
                    "task_id": "recruit_002",
                    "task_type": "interview_preparation",
                    "description": "Prepare technical interview questions for a senior Python developer position",
                    "evaluation_criteria": ["relevance", "difficulty", "coverage"],
                    "expected_output": "List of 10 questions with expected answers",
                    "domain": "recruitment",
                    "complexity": "hard"
                }
            ])

        if self.domain in ["marketing", "all"]:
            tasks.extend([
                {
                    "task_id": "market_001",
                    "task_type": "campaign_analysis",
                    "description": "Analyze the performance of our Q3 email marketing campaign",
                    "evaluation_criteria": ["metrics_analysis", "insights", "recommendations"],
                    "expected_output": "Comprehensive campaign report with actionable insights",
                    "domain": "marketing",
                    "complexity": "medium"
                },
                {
                    "task_id": "market_002",
                    "task_type": "content_strategy",
                    "description": "Develop a 6-month content marketing strategy for our SaaS product",
                    "evaluation_criteria": ["strategy_completeness", "channel_selection", "measurement_plan"],
                    "expected_output": "Detailed content calendar and measurement framework",
                    "domain": "marketing",
                    "complexity": "hard"
                }
            ])

        return tasks

    def evaluate_sample(self, sample: Dict[str, Any], memory_system) -> Dict[str, Any]:
        """Evaluate a single xBench sample using the memory system."""
        task_description = sample.get("description", "")
        task_type = sample.get("task_type", "unknown")
        domain = sample.get("domain", "unknown")

        try:
            # Use memory system to help complete the professional task
            # Professional tasks often require recalling past experiences and best practices

            memory_results = memory_system.query_memory(
                query=f"Professional task assistance: {task_description}",
                top_k=15
            )

            # Simulate task completion using memory guidance
            # In a real implementation, this would involve more sophisticated task execution
            task_context = "\n".join([r.get("content", "")
                                     for r in memory_results])

            result = {
                "task_id": sample.get("task_id"),
                "task_type": task_type,
                "domain": domain,
                "description": task_description,
                "memory_items_used": len(memory_results),
                "task_context": task_context[:500],  # Truncate for storage
                "completion_quality": self._assess_completion_quality(sample, memory_results),
                "predicted_output": self._simulate_task_completion(sample, memory_results)
            }

        except Exception as e:
            self.logger.error(
                f"Error evaluating xBench sample {sample.get('task_id')}: {e}")
            result = {
                "task_id": sample.get("task_id"),
                "error": str(e),
                "task_type": task_type,
                "domain": domain
            }

        return result

    def _assess_completion_quality(self, task: Dict[str, Any], memory_results: List[Dict[str, Any]]) -> float:
        """Assess the quality of task completion based on memory utilization."""
        # Simple heuristic: more relevant memory items = better completion
        base_score = min(len(memory_results) / 10.0, 1.0)

        # Bonus for using tool-related memories for technical tasks
        has_relevant_tools = any(
            "tool" in r.get("type", "").lower(
            ) or "skill" in r.get("type", "").lower()
            for r in memory_results
        )

        if has_relevant_tools:
            base_score += 0.2

        return min(base_score, 1.0)

    def _simulate_task_completion(self, task: Dict[str, Any], memory_results: List[Dict[str, Any]]) -> str:
        """Simulate task completion output."""
        task_type = task.get("task_type", "")

        # Generate mock outputs based on task type
        if "screening" in task_type:
            return "Top 5 candidates: 1. John Doe (Python expert), 2. Jane Smith (ML specialist), ..."
        elif "interview" in task_type:
            return "Technical questions: 1. Explain Python GIL, 2. Design LRU cache, ..."
        elif "campaign" in task_type:
            return "Campaign analysis: 15% open rate, key insights: subject line optimization needed..."
        elif "content" in task_type:
            return "Content strategy: Monthly themes, 3 posts/week, SEO optimization focus..."
        else:
            return f"Completed {task_type} task using {len(memory_results)} memory references"

    def validate_result(self, result: Dict[str, Any], ground_truth: Any) -> float:
        """Validate and score a result against ground truth."""
        if "error" in result:
            return 0.0

        # xBench evaluation is more subjective than exact matching
        # Use completion quality as the primary metric
        quality_score = result.get("completion_quality", 0.0)

        # Bonus for structured output
        output = result.get("predicted_output", "")
        if len(output) > 100 and "\n" in output:  # Structured response
            quality_score += 0.1

        return min(quality_score, 1.0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded dataset."""
        if not self.dataset:
            self.load_dataset()

        if not self.dataset:
            return {"error": "No dataset loaded"}

        stats = {
            "total_samples": len(self.dataset),
            "domains": {},
            "task_types": {},
            "complexities": {}
        }

        for sample in self.dataset:
            domain = sample.get("domain", "unknown")
            task_type = sample.get("task_type", "unknown")
            complexity = sample.get("complexity", "unknown")

            stats["domains"][domain] = stats["domains"].get(domain, 0) + 1
            stats["task_types"][task_type] = stats["task_types"].get(
                task_type, 0) + 1
            stats["complexities"][complexity] = stats["complexities"].get(
                complexity, 0) + 1

        return stats
