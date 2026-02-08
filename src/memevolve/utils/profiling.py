"""
Performance profiling tools for MemEvolve memory systems.

This module provides tools for profiling memory system operations,
analyzing performance bottlenecks, and generating optimization reports.
"""

import cProfile
import io
import pstats
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .logging import get_logger


@dataclass
class ProfileResult:
    """Result of a profiling operation."""

    operation_name: str
    start_time: str
    end_time: str
    duration_seconds: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "metadata": self.metadata
        }


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""

    report_id: str
    timestamp: str
    total_operations: int
    total_duration: float
    average_operation_time: float
    operation_breakdown: Dict[str, Dict[str, Any]]
    bottlenecks: List[str]
    recommendations: List[str]
    raw_results: List[ProfileResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "total_operations": self.total_operations,
            "total_duration": self.total_duration,
            "average_operation_time": self.average_operation_time,
            "operation_breakdown": self.operation_breakdown,
            "bottlenecks": self.bottlenecks,
            "recommendations": self.recommendations,
            "raw_results": [r.to_dict() for r in self.raw_results]
        }


class MemoryProfiler:
    """Profiler for memory system operations."""

    def __init__(self):
        self.logger = get_logger("memory_profiler")
        self.results: List[ProfileResult] = []
        self._profiler = cProfile.Profile()

    @contextmanager
    def profile_operation(self, operation_name: str, **metadata):
        """Context manager for profiling a specific operation.

        Args:
            operation_name: Name of the operation being profiled
            **metadata: Additional metadata to store with the profile
        """
        start_time = time.time()
        start_timestamp = datetime.now(timezone.utc).isoformat() + "Z"

        # Start profiling
        self._profiler.enable()

        try:
            yield
        finally:
            # Stop profiling
            self._profiler.disable()

            end_time = time.time()
            end_timestamp = datetime.now(timezone.utc).isoformat() + "Z"
            duration = end_time - start_time

            # Get memory usage if available
            memory_usage = self._get_memory_usage()

            # Get CPU usage if available
            cpu_usage = self._get_cpu_usage()

            result = ProfileResult(
                operation_name=operation_name,
                start_time=start_timestamp,
                end_time=end_timestamp,
                duration_seconds=duration,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                metadata=metadata
            )

            self.results.append(result)
            self.logger.debug(f"Profiled {operation_name}: {duration:.4f}s")

    def profile_function(
        self,
        func: Callable,
        *args,
        operation_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Profile a function call.

        Args:
            func: Function to profile
            *args: Arguments to pass to the function
            operation_name: Name for the profiling operation
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Return value of the function
        """
        op_name = operation_name or f"{func.__name__}"

        with self.profile_operation(op_name):
            return func(*args, **kwargs)

    def profile_memory_system_operation(
        self,
        memory_system,
        operation: str,
        *args,
        **kwargs
    ) -> Any:
        """Profile a memory system operation.

        Args:
            memory_system: MemorySystem instance
            operation: Operation name (add_experience, query_memory, etc.)
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Return value of the operation
        """
        if not hasattr(memory_system, operation):
            raise ValueError(
                f"Memory system does not have operation: {operation}")

        op_func = getattr(memory_system, operation)

        with self.profile_operation(f"memory_system.{operation}", operation=operation):
            return op_func(*args, **kwargs)

    def generate_report(self, report_id: Optional[str] = None) -> PerformanceReport:
        """Generate a comprehensive performance report.

        Args:
            report_id: Optional custom report ID

        Returns:
            PerformanceReport with analysis
        """
        if not self.results:
            return PerformanceReport(
                report_id=report_id or "empty_report",
                timestamp=datetime.now(timezone.utc).isoformat() + "Z",
                total_operations=0,
                total_duration=0.0,
                average_operation_time=0.0,
                operation_breakdown={},
                bottlenecks=[],
                recommendations=["No profiling data available"]
            )

        # Calculate basic statistics
        total_duration = sum(r.duration_seconds for r in self.results)
        avg_operation_time = total_duration / len(self.results)

        # Group by operation type
        operation_breakdown = {}
        for result in self.results:
            op_name = result.operation_name
            if op_name not in operation_breakdown:
                operation_breakdown[op_name] = {
                    "count": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0,
                    "min_time": float('inf'),
                    "max_time": 0.0
                }

            stats = operation_breakdown[op_name]
            stats["count"] += 1
            stats["total_time"] += result.duration_seconds
            stats["min_time"] = min(stats["min_time"], result.duration_seconds)
            stats["max_time"] = max(stats["max_time"], result.duration_seconds)
            stats["avg_time"] = stats["total_time"] / stats["count"]

        # Identify bottlenecks
        bottlenecks = []
        threshold = avg_operation_time * 2  # Operations taking 2x average

        for op_name, stats in operation_breakdown.items():
            if stats["avg_time"] > threshold:
                bottlenecks.append(
                    f"{op_name}: {stats['avg_time']:.4f}s avg "
                    f"({stats['count']} operations)"
                )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            operation_breakdown, bottlenecks)

        report = PerformanceReport(
            report_id=report_id or f"report_{int(time.time())}",
            timestamp=datetime.now(timezone.utc).isoformat() + "Z",
            total_operations=len(self.results),
            total_duration=total_duration,
            average_operation_time=avg_operation_time,
            operation_breakdown=operation_breakdown,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            raw_results=self.results.copy()
        )

        return report

    def _generate_recommendations(
        self,
        operation_breakdown: Dict[str, Dict[str, Any]],
        bottlenecks: List[str]
    ) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        if bottlenecks:
            recommendations.append(
                f"Address {len(bottlenecks)} performance bottleneck(s) identified"
            )

        # Check for operation distribution
        total_ops = sum(stats["count"]
                        for stats in operation_breakdown.values())
        for op_name, stats in operation_breakdown.items():
            percentage = (stats["count"] / total_ops) * 100
            if percentage > 70:
                recommendations.append(
                    f"Operation '{op_name}' dominates {percentage:.1f}% of total operations - "
                    "consider optimizing this operation"
                )

        # Check for high variance operations
        for op_name, stats in operation_breakdown.items():
            if stats["count"] > 5:  # Need enough samples
                variance = stats["max_time"] - stats["min_time"]
                if variance > stats["avg_time"] * 3:
                    recommendations.append(
                        f"Operation '{op_name}' shows high variance "
                        f"({variance:.4f}s range) - investigate consistency"
                    )

        if not recommendations:
            recommendations.append(
                "Performance looks good - no major issues identified")

        return recommendations

    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return None

    def _get_cpu_usage(self) -> Optional[float]:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return None

    def clear_results(self):
        """Clear all profiling results."""
        self.results.clear()
        self.logger.info("Profiling results cleared")

    def export_profile_stats(self, filepath: Union[str, Path]) -> bool:
        """Export detailed profiling statistics to file.

        Args:
            filepath: Path to save the profile statistics

        Returns:
            True if export successful, False otherwise
        """
        try:
            filepath = Path(filepath)

            # Create string stream for stats
            stream = io.StringIO()
            stats = pstats.Stats(self._profiler, stream=stream)
            stats.sort_stats('cumulative').print_stats()

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(stream.getvalue())

            self.logger.info(f"Profile statistics exported to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export profile stats: {str(e)}")
            return False

    def run_benchmark(
        self,
        memory_system,
        operations: List[Dict[str, Any]],
        iterations: int = 1
    ) -> PerformanceReport:
        """Run a benchmark suite on the memory system.

        Args:
            memory_system: MemorySystem instance to benchmark
            operations: List of operation specifications
            iterations: Number of times to run each operation

        Returns:
            PerformanceReport with benchmark results
        """
        self.clear_results()

        for op_spec in operations:
            op_name = op_spec.get("name", "unknown")
            op_type = op_spec.get("type", "custom")
            op_args = op_spec.get("args", [])
            op_kwargs = op_spec.get("kwargs", {})

            for i in range(iterations):
                if op_type == "add_experience":
                    self.profile_memory_system_operation(
                        memory_system, "add_experience", *op_args, **op_kwargs
                    )
                elif op_type == "query_memory":
                    self.profile_memory_system_operation(
                        memory_system, "query_memory", *op_args, **op_kwargs
                    )
                elif op_type == "custom":
                    # For custom operations, assume the spec contains a callable
                    if "func" in op_spec:
                        self.profile_function(
                            op_spec["func"], *op_args, **op_kwargs,
                            operation_name=op_name
                        )
                else:
                    self.logger.warning(f"Unknown operation type: {op_type}")

        return self.generate_report(f"benchmark_{int(time.time())}")


# Convenience functions
def profile_memory_operation(
    memory_system,
    operation: str,
    *args,
    profiler: Optional[MemoryProfiler] = None,
    **kwargs
) -> Any:
    """Convenience function to profile a memory system operation.

    Args:
        memory_system: MemorySystem instance
        operation: Operation name
        *args: Arguments for the operation
        profiler: Optional MemoryProfiler instance (creates one if None)
        **kwargs: Keyword arguments for the operation

    Returns:
        Return value of the operation
    """
    if profiler is None:
        profiler = MemoryProfiler()

    return profiler.profile_memory_system_operation(
        memory_system, operation, *args, **kwargs
    )


def benchmark_memory_system(
    memory_system,
    operations: List[Dict[str, Any]],
    iterations: int = 1,
    export_path: Optional[Union[str, Path]] = None
) -> PerformanceReport:
    """Convenience function to run benchmarks on a memory system.

    Args:
        memory_system: MemorySystem instance to benchmark
        operations: List of operation specifications
        iterations: Number of iterations per operation
        export_path: Optional path to export profile statistics

    Returns:
        PerformanceReport with results
    """
    profiler = MemoryProfiler()

    report = profiler.run_benchmark(memory_system, operations, iterations)

    if export_path:
        profiler.export_profile_stats(export_path)

    return report
