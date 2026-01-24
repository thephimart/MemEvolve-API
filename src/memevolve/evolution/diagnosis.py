from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
from enum import Enum


class FailureType(Enum):
    """Types of failures that can be diagnosed."""
    MEMORY_GAP = "memory_gap"
    RETRIEVAL_FAILURE = "retrieval_failure"
    KNOWLEDGE_CONFLICT = "knowledge_conflict"
    OUTDATED_INFO = "outdated_info"
    ABSENT_ABSTRACTION = "absent_abstraction"
    POOR_QUALITY = "poor_quality"


@dataclass
class FailureAnalysis:
    """Analysis of a specific failure."""

    failure_type: FailureType
    description: str
    severity: float
    memory_units_involved: List[str] = field(default_factory=list)
    suggested_fixes: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class TrajectoryStep:
    """Single step in an agent trajectory."""

    step_id: int
    task_description: str
    action_taken: str
    observation: str
    memory_accessed: List[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    timestamp: Optional[float] = None


@dataclass
class MemoryGapsAnalysis:
    """Analysis of memory gaps identified from trajectory analysis."""

    missing_lessons: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    missing_abstractions: List[str] = field(default_factory=list)
    outdated_units: List[str] = field(default_factory=list)
    coverage_score: float = 0.0


@dataclass
class DiagnosisReport:
    """Comprehensive diagnosis report from trajectory analysis."""

    trajectory_id: str
    success: bool
    failure_analysis: Optional[FailureAnalysis] = None
    memory_gaps: Optional[MemoryGapsAnalysis] = None
    retrieval_issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnosis report to dictionary."""
        return {
            "trajectory_id": self.trajectory_id,
            "success": self.success,
            "failure_analysis": (
                {
                    "failure_type": self.failure_analysis.failure_type.value,
                    "description": self.failure_analysis.description,
                    "severity": self.failure_analysis.severity,
                    "memory_units_involved": self.failure_analysis.memory_units_involved,
                    "suggested_fixes": self.failure_analysis.suggested_fixes,
                    "confidence": self.failure_analysis.confidence
                } if self.failure_analysis else None
            ),
            "memory_gaps": (
                {
                    "missing_lessons": self.memory_gaps.missing_lessons,
                    "missing_skills": self.memory_gaps.missing_skills,
                    "missing_abstractions": self.memory_gaps.missing_abstractions,
                    "outdated_units": self.memory_gaps.outdated_units,
                    "coverage_score": self.memory_gaps.coverage_score
                } if self.memory_gaps else None
            ),
            "retrieval_issues": self.retrieval_issues,
            "suggestions": self.suggestions,
            "confidence": self.confidence
        }


class DiagnosisEngine:
    """Diagnoses memory issues from agent trajectories."""

    def __init__(self):
        self.retrieval_patterns: Dict[str, List[str]] = {}
        self.failure_patterns: Dict[str, List[FailureAnalysis]] = {}

    def analyze_trajectory(
        self,
        trajectory: List[TrajectoryStep],
        memory_system: Any
    ) -> DiagnosisReport:
        """Analyze a complete agent trajectory for memory issues.

        Args:
            trajectory: List of trajectory steps
            memory_system: Memory system instance to query

        Returns:
            DiagnosisReport with analysis results
        """
        trajectory_id = self._generate_trajectory_id(trajectory)

        overall_success = all(step.success for step in trajectory)

        failure_analysis = None
        if not overall_success:
            failure_analysis = self._diagnose_failure(
                trajectory, memory_system)

        memory_gaps = self._analyze_memory_gaps(trajectory, memory_system)

        retrieval_issues = self._analyze_retrieval_issues(
            trajectory, memory_system)

        suggestions = self._generate_suggestions(
            failure_analysis, memory_gaps, retrieval_issues
        )

        confidence = self._calculate_confidence(
            failure_analysis, memory_gaps, retrieval_issues
        )

        return DiagnosisReport(
            trajectory_id=trajectory_id,
            success=overall_success,
            failure_analysis=failure_analysis,
            memory_gaps=memory_gaps,
            retrieval_issues=retrieval_issues,
            suggestions=suggestions,
            confidence=confidence
        )

    def _generate_trajectory_id(self, trajectory: List[TrajectoryStep]) -> str:
        """Generate unique ID for trajectory."""
        import hashlib
        trajectory_str = json.dumps([
            {
                "step_id": step.step_id,
                "task": step.task_description,
                "action": step.action_taken
            }
            for step in trajectory
        ], sort_keys=True)
        return hashlib.md5(trajectory_str.encode()).hexdigest()[:12]

    def _diagnose_failure(
        self,
        trajectory: List[TrajectoryStep],
        memory_system: Any
    ) -> Optional[FailureAnalysis]:
        """Diagnose the root cause of failure in trajectory.

        Args:
            trajectory: Failed trajectory steps
            memory_system: Memory system to query

        Returns:
            FailureAnalysis if failure detected
        """
        failed_steps = [step for step in trajectory if not step.success]

        if not failed_steps:
            return None

        failed_step = failed_steps[0]

        if failed_step.error_message:
            error_lower = failed_step.error_message.lower()

            if "not found" in error_lower or "no memory" in error_lower:
                return FailureAnalysis(
                    failure_type=FailureType.MEMORY_GAP,
                    description=f"Memory gap detected: {failed_step.error_message}",
                    severity=0.8,
                    memory_units_involved=failed_step.memory_accessed,
                    suggested_fixes=[
                        "Add more memory units for this task type",
                        "Improve retrieval strategy",
                        "Enable more aggressive memory encoding"
                    ],
                    confidence=0.7
                )

            elif "conflict" in error_lower or "contradiction" in error_lower:
                return FailureAnalysis(
                    failure_type=FailureType.KNOWLEDGE_CONFLICT,
                    description=f"Knowledge conflict detected: {failed_step.error_message}",
                    severity=0.9,
                    memory_units_involved=failed_step.memory_accessed,
                    suggested_fixes=[
                        "Implement conflict resolution strategy",
                        "Add versioning to memory units",
                        "Prioritize more recent or higher-quality information"
                    ],
                    confidence=0.6
                )

            elif "outdated" in error_lower or "stale" in error_lower:
                return FailureAnalysis(
                    failure_type=FailureType.OUTDATED_INFO,
                    description=f"Outdated information detected: {failed_step.error_message}",
                    severity=0.7,
                    memory_units_involved=failed_step.memory_accessed,
                    suggested_fixes=[
                        "Enable automatic memory pruning",
                        "Add timestamp-based retrieval",
                        "Implement forgetting mechanisms"
                    ],
                    confidence=0.8
                )

        if len(failed_step.memory_accessed) == 0:
            return FailureAnalysis(
                failure_type=FailureType.MEMORY_GAP,
                description="No memory was accessed during failed step",
                severity=0.9,
                memory_units_involved=[],
                suggested_fixes=[
                    "Increase memory retrieval scope",
                    "Lower similarity threshold",
                    "Enable more encoding strategies"
                ],
                confidence=0.5
            )

        return FailureAnalysis(
            failure_type=FailureType.RETRIEVAL_FAILURE,
            description=f"Retrieval failure: {failed_step.error_message}",
            severity=0.8,
            memory_units_involved=failed_step.memory_accessed,
            suggested_fixes=[
                "Improve embedding quality",
                "Adjust retrieval parameters",
                "Add hybrid retrieval strategy"
            ],
            confidence=0.6
        )

    def _analyze_memory_gaps(
        self,
        trajectory: List[TrajectoryStep],
        memory_system: Any
    ) -> Optional[MemoryGapsAnalysis]:
        """Analyze trajectory for memory coverage gaps.

        Args:
            trajectory: Trajectory steps to analyze
            memory_system: Memory system to query

        Returns:
            MemoryGapsAnalysis with identified gaps
        """
        if not memory_system:
            return None

        try:
            all_memories = []

            try:
                if hasattr(memory_system, 'store') and hasattr(memory_system.store, 'retrieve_all'):
                    all_memories = memory_system.store.retrieve_all()
                elif hasattr(memory_system, 'retrieve_all'):
                    all_memories = memory_system.retrieve_all()
            except Exception:
                all_memories = []

            memory_types = {}
            for mem in all_memories:
                mem_type = mem.get('type', 'unknown')
                if mem_type not in memory_types:
                    memory_types[mem_type] = []
                memory_types[mem_type].append(mem)

            trajectory_topics = self._extract_topics(trajectory)

            missing_lessons = []
            missing_skills = []
            missing_abstractions = []

            for topic in trajectory_topics:
                has_lesson = any(
                    topic.lower() in mem.get('content', '').lower()
                    for mem in memory_types.get('lesson', [])
                )
                if not has_lesson:
                    missing_lessons.append(topic)

                has_skill = any(
                    topic.lower() in mem.get('content', '').lower()
                    for mem in memory_types.get('skill', [])
                )
                if not has_skill:
                    missing_skills.append(topic)

                has_abstraction = any(
                    topic.lower() in mem.get('content', '').lower()
                    for mem in memory_types.get('abstraction', [])
                )
                if not has_abstraction:
                    missing_abstractions.append(topic)

            total_possible = (
                len(trajectory_topics) * 3
            )
            actual = (
                len(memory_types.get('lesson', [])) +
                len(memory_types.get('skill', [])) +
                len(memory_types.get('abstraction', []))
            )
            coverage_score = min(actual / max(total_possible, 1), 1.0)

            return MemoryGapsAnalysis(
                missing_lessons=missing_lessons,
                missing_skills=missing_skills,
                missing_abstractions=missing_abstractions,
                outdated_units=[],
                coverage_score=coverage_score
            )
        except Exception:
            return None

    def _extract_topics(self, trajectory: List[TrajectoryStep]) -> List[str]:
        """Extract topics from trajectory for gap analysis."""
        topics = set()

        for step in trajectory:
            words = step.task_description.split()
            for word in words:
                if len(word) > 4 and word.isalpha():
                    topics.add(word.lower())

        return list(topics)[:10]

    def _analyze_retrieval_issues(
        self,
        trajectory: List[TrajectoryStep],
        memory_system: Any
    ) -> List[str]:
        """Analyze trajectory for retrieval issues.

        Args:
            trajectory: Trajectory steps to analyze
            memory_system: Memory system to query

        Returns:
            List of identified retrieval issues
        """
        issues = []

        for step in trajectory:
            if len(step.memory_accessed) == 0 and step.success:
                issues.append(
                    f"Step {step.step_id} succeeded with no memory access"
                )

            if len(step.memory_accessed) > 10:
                issues.append(
                    f"Step {step.step_id} accessed excessive memory "
                    f"({len(step.memory_accessed)} units)"
                )

        return issues

    def _generate_suggestions(
        self,
        failure_analysis: Optional[FailureAnalysis],
        memory_gaps: Optional[MemoryGapsAnalysis],
        retrieval_issues: List[str]
    ) -> List[str]:
        """Generate improvement suggestions from analysis.

        Args:
            failure_analysis: Failure analysis results
            memory_gaps: Memory gaps analysis
            retrieval_issues: Retrieval issues list

        Returns:
            List of suggestions
        """
        suggestions = []

        if failure_analysis:
            suggestions.extend(failure_analysis.suggested_fixes)

        if memory_gaps and memory_gaps.coverage_score < 0.5:
            suggestions.append(
                "Memory coverage is low - enable more encoding strategies"
            )

        if memory_gaps and len(memory_gaps.missing_abstractions) > 3:
            suggestions.append(
                "Consider enabling abstractions to reduce memory gaps"
            )

        if len(retrieval_issues) > 3:
            suggestions.append(
                "Adjust retrieval parameters to optimize memory access patterns"
            )

        return suggestions[:10]

    def _calculate_confidence(
        self,
        failure_analysis: Optional[FailureAnalysis],
        memory_gaps: Optional[MemoryGapsAnalysis],
        retrieval_issues: List[str]
    ) -> float:
        """Calculate overall confidence in diagnosis.

        Args:
            failure_analysis: Failure analysis results
            memory_gaps: Memory gaps analysis
            retrieval_issues: Retrieval issues list

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.0

        if failure_analysis:
            confidence += 0.4 * failure_analysis.confidence

        if memory_gaps:
            confidence += 0.3 * (1.0 - memory_gaps.coverage_score)

        if retrieval_issues:
            confidence += 0.3 * min(len(retrieval_issues) / 10, 1.0)

        return min(confidence, 1.0)

    def batch_analyze(
        self,
        trajectories: List[List[TrajectoryStep]],
        memory_system: Any
    ) -> List[DiagnosisReport]:
        """Analyze multiple trajectories in batch.

        Args:
            trajectories: List of trajectories to analyze
            memory_system: Memory system to query

        Returns:
            List of diagnosis reports
        """
        reports = []

        for trajectory in trajectories:
            report = self.analyze_trajectory(trajectory, memory_system)
            reports.append(report)

        return reports

    def get_summary_statistics(
        self,
        reports: List[DiagnosisReport]
    ) -> Dict[str, Any]:
        """Get summary statistics from multiple diagnosis reports.

        Args:
            reports: List of diagnosis reports

        Returns:
            Dictionary with summary statistics
        """
        total = len(reports)
        if total == 0:
            return {}

        successful = sum(1 for r in reports if r.success)
        failed = total - successful

        failure_types = {}
        for report in reports:
            if report.failure_analysis:
                ftype = report.failure_analysis.failure_type.value
                failure_types[ftype] = failure_types.get(ftype, 0) + 1

        avg_confidence = sum(r.confidence for r in reports) / total

        return {
            "total_trajectories": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total,
            "failure_types": failure_types,
            "average_confidence": avg_confidence
        }
