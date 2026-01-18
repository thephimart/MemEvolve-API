import sys

sys.path.insert(0, 'src')

import pytest
from evolution.genotype import MemoryGenotype, GenotypeFactory
from evolution.diagnosis import (
    DiagnosisEngine,
    TrajectoryStep,
    FailureType,
    MemoryGapsAnalysis,
    DiagnosisReport
)


def test_trajectory_step_creation():
    """Test trajectory step creation."""
    step = TrajectoryStep(
        step_id=1,
        task_description="Complete task A",
        action_taken="Execute action",
        observation="Result observed",
        memory_accessed=["mem1", "mem2"],
        success=True
    )

    assert step.step_id == 1
    assert step.success is True
    assert len(step.memory_accessed) == 2


def test_diagnosis_engine_initialization():
    """Test diagnosis engine initialization."""
    engine = DiagnosisEngine()

    assert engine is not None
    assert engine.retrieval_patterns == {}
    assert engine.failure_patterns == {}


def test_analyze_successful_trajectory():
    """Test analysis of successful trajectory."""
    engine = DiagnosisEngine()

    trajectory = [
        TrajectoryStep(
            step_id=1,
            task_description="Task 1",
            action_taken="Action 1",
            observation="Observed 1",
            memory_accessed=["mem1"],
            success=True
        ),
        TrajectoryStep(
            step_id=2,
            task_description="Task 2",
            action_taken="Action 2",
            observation="Observed 2",
            memory_accessed=["mem2"],
            success=True
        )
    ]

    report = engine.analyze_trajectory(trajectory, None)

    assert isinstance(report, DiagnosisReport)
    assert report.success is True
    assert report.failure_analysis is None


def test_analyze_failed_trajectory():
    """Test analysis of failed trajectory."""
    engine = DiagnosisEngine()

    trajectory = [
        TrajectoryStep(
            step_id=1,
            task_description="Task 1",
            action_taken="Action 1",
            observation="Observed 1",
            memory_accessed=["mem1"],
            success=True
        ),
        TrajectoryStep(
            step_id=2,
            task_description="Task 2",
            action_taken="Action 2",
            observation="Error: information not found",
            memory_accessed=[],
            success=False,
            error_message="Error: information not found"
        )
    ]

    report = engine.analyze_trajectory(trajectory, None)

    assert isinstance(report, DiagnosisReport)
    assert report.success is False
    assert report.failure_analysis is not None
    assert report.failure_analysis.failure_type in [
        FailureType.MEMORY_GAP,
        FailureType.RETRIEVAL_FAILURE
    ]


def test_failure_type_detection():
    """Test detection of different failure types."""
    engine = DiagnosisEngine()

    memory_not_found_trajectory = [
        TrajectoryStep(
            step_id=1,
            task_description="Task",
            action_taken="Action",
            observation="Error: memory not found",
            memory_accessed=[],
            success=False,
            error_message="Error: memory not found"
        )
    ]

    report = engine.analyze_trajectory(memory_not_found_trajectory, None)

    assert report.failure_analysis.failure_type == FailureType.MEMORY_GAP


def test_diagnosis_report_to_dict():
    """Test conversion of diagnosis report to dictionary."""
    engine = DiagnosisEngine()

    trajectory = [
        TrajectoryStep(
            step_id=1,
            task_description="Task",
            action_taken="Action",
            observation="Observed",
            memory_accessed=["mem1"],
            success=True
        )
    ]

    report = engine.analyze_trajectory(trajectory, None)
    report_dict = report.to_dict()

    assert isinstance(report_dict, dict)
    assert "trajectory_id" in report_dict
    assert "success" in report_dict
    assert "suggestions" in report_dict


def test_batch_analyze():
    """Test batch analysis of multiple trajectories."""
    engine = DiagnosisEngine()

    trajectory1 = [
        TrajectoryStep(
            step_id=1,
            task_description="Task 1",
            action_taken="Action 1",
            observation="Observed 1",
            memory_accessed=["mem1"],
            success=True
        )
    ]

    trajectory2 = [
        TrajectoryStep(
            step_id=1,
            task_description="Task 2",
            action_taken="Action 2",
            observation="Error",
            memory_accessed=[],
            success=False,
            error_message="Error: not found"
        )
    ]

    reports = engine.batch_analyze([trajectory1, trajectory2], None)

    assert len(reports) == 2
    assert all(isinstance(r, DiagnosisReport) for r in reports)


def test_get_summary_statistics():
    """Test summary statistics from multiple reports."""
    engine = DiagnosisEngine()

    trajectory1 = [
        TrajectoryStep(
            step_id=1,
            task_description="Task 1",
            action_taken="Action 1",
            observation="Observed 1",
            memory_accessed=["mem1"],
            success=True
        )
    ]

    trajectory2 = [
        TrajectoryStep(
            step_id=1,
            task_description="Task 2",
            action_taken="Action 2",
            observation="Error",
            memory_accessed=[],
            success=False,
            error_message="Error: not found"
        )
    ]

    reports = engine.batch_analyze([trajectory1, trajectory2], None)
    summary = engine.get_summary_statistics(reports)

    assert "total_trajectories" in summary
    assert "successful" in summary
    assert "failed" in summary
    assert "success_rate" in summary
    assert summary["total_trajectories"] == 2
    assert summary["successful"] == 1
    assert summary["failed"] == 1


def test_confidence_calculation():
    """Test confidence score calculation."""
    engine = DiagnosisEngine()

    trajectory = [
        TrajectoryStep(
            step_id=1,
            task_description="Task",
            action_taken="Action",
            observation="Error: not found",
            memory_accessed=[],
            success=False,
            error_message="Error: not found"
        )
    ]

    report = engine.analyze_trajectory(trajectory, None)

    assert 0.0 <= report.confidence <= 1.0


def test_trajectory_id_generation():
    """Test unique trajectory ID generation."""
    engine = DiagnosisEngine()

    trajectory1 = [
        TrajectoryStep(
            step_id=1,
            task_description="Task A",
            action_taken="Action A",
            observation="Observed A",
            memory_accessed=["mem1"],
            success=True
        )
    ]

    trajectory2 = [
        TrajectoryStep(
            step_id=1,
            task_description="Task B",
            action_taken="Action B",
            observation="Observed B",
            memory_accessed=["mem2"],
            success=True
        )
    ]

    report1 = engine.analyze_trajectory(trajectory1, None)
    report2 = engine.analyze_trajectory(trajectory2, None)

    assert report1.trajectory_id != report2.trajectory_id
    assert len(report1.trajectory_id) == 12
