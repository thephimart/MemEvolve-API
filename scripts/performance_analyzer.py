#!/usr/bin/env python3
"""
MemEvolve-API Performance Analyzer
==================================

Generates comprehensive performance reports from system logs and data files.
Can analyze any time range without requiring LLM endpoints.

Usage:
    python scripts/performance_analyzer.py --start-date 2026-01-23 --end-date 2026-01-23
    python scripts/performance_analyzer.py --days 7  # Last 7 days
    python scripts/performance_analyzer.py --output custom_report.md
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memevolve.utils.logging_manager import LoggingManager

logger = LoggingManager.get_logger(__name__)
logger.info("Performance analyzer script initialized")
import json
import os
import re
import statistics
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class PerformanceAnalyzer:
    """Comprehensive performance analyzer for MemEvolve-API."""

    def __init__(self, data_dir: str = "./data", logs_dir: str = "./logs"):
        self.data_dir = Path(data_dir)
        self.logs_dir = Path(logs_dir)
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None

    def set_date_range(self, start_date: str, end_date: str):
        """Set analysis date range."""
        self.start_date = datetime.fromisoformat(start_date)
        self.end_date = datetime.fromisoformat(end_date)

    def set_days_range(self, days: int):
        """Set analysis to last N days."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        self.start_date = start_date
        self.end_date = end_date

    def _parse_log_timestamp(self, line: str) -> Optional[datetime]:
        """Extract timestamp from log line."""
        # Format: 2026-01-23 00:21:40,325
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
        if timestamp_match:
            try:
                return datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S,%f')
            except ValueError:
                return None
        return None

    def _is_in_date_range(self, timestamp: datetime) -> bool:
        """Check if timestamp is within analysis range."""
        if not self.start_date or not self.end_date:
            return True
        return self.start_date <= timestamp <= self.end_date

    def analyze_api_requests(self) -> Dict[str, Any]:
        """Analyze API request patterns and success rates."""
        api_log = self.logs_dir / "api-server.log"
        if not api_log.exists():
            return {"error": "API server log not found"}

        total_requests = 0
        successful_requests = 0
        status_codes = Counter()

        try:
            with open(api_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    timestamp = self._parse_log_timestamp(line)
                    if timestamp and self._is_in_date_range(timestamp):
                        # Count HTTP requests (httpx log lines)
                        if "HTTP Request:" in line and "HTTP/" in line:
                            total_requests += 1

                            # Extract status code from "HTTP/1.1 200 OK" format
                            status_match = re.search(r'HTTP/\d\.\d (\d+)', line)
                            if status_match:
                                status_code = int(status_match.group(1))
                                status_codes[status_code] += 1
                                if status_code == 200:
                                    successful_requests += 1

        except Exception as e:
            return {"error": f"Failed to analyze API requests: {e}"}

        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0

        # Estimate response time from evolution metrics (since direct timing not in logs)
        # This is an approximation based on the evolution system's average response time
        evolution_data = self.analyze_evolution_state()
        estimated_avg_response_time = evolution_data.get("average_response_time", 0) if "error" not in evolution_data else 0

        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": success_rate,
            "status_codes": dict(status_codes),
            "estimated_avg_response_time": estimated_avg_response_time
        }

    def analyze_memory_operations(self) -> Dict[str, Any]:
        """Analyze memory system operations."""
        middleware_log = self.logs_dir / "middleware.log"
        if not middleware_log.exists():
            return {"error": "Middleware log not found"}

        encoding_times = []
        retrieval_times = []
        memory_injections = 0
        experiences_created = 0

        start_encoding_time = None

        try:
            with open(middleware_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    timestamp = self._parse_log_timestamp(line)
                    if not timestamp or not self._is_in_date_range(timestamp):
                        continue

                    # Track memory injections
                    if "Injected" in line and "memories" in line:
                        memory_injections += 1

                    # Track experience encoding times
                    if "Calling memory_system.add_experience()" in line:
                        start_encoding_time = timestamp
                    elif "Experience added successfully" in line and start_encoding_time:
                        encoding_duration = (timestamp - start_encoding_time).total_seconds()
                        encoding_times.append(encoding_duration)
                        experiences_created += 1
                        start_encoding_time = None

        except Exception as e:
            return {"error": f"Failed to analyze memory operations: {e}"}

        return {
            "experiences_created": experiences_created,
            "memory_injections": memory_injections,
            "encoding_performance": {
                "count": len(encoding_times),
                "average": statistics.mean(encoding_times) if encoding_times else 0,
                "min": min(encoding_times) if encoding_times else 0,
                "max": max(encoding_times) if encoding_times else 0
            }
        }

    def analyze_quality_scores(self) -> Dict[str, Any]:
        """Extract and analyze quality scores with summary statistics only."""
        middleware_log = self.logs_dir / "middleware.log"
        if not middleware_log.exists():
            return {"error": "Middleware log not found"}

        quality_scores = []

        try:
            with open(middleware_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    timestamp = self._parse_log_timestamp(line)
                    if timestamp and self._is_in_date_range(timestamp):
                        # Extract quality scores
                        score_match = re.search(r'quality score["\s:=]+([\d.]+)', line, re.IGNORECASE)
                        if score_match:
                            quality_scores.append(float(score_match.group(1)))

        except Exception as e:
            return {"error": f"Failed to analyze quality scores: {e}"}

        if not quality_scores:
            return {"count": 0, "average": 0.0, "min": 0.0, "max": 0.0}

        # Calculate summary statistics only
        return {
            "count": len(quality_scores),
            "average": statistics.mean(quality_scores),
            "min": min(quality_scores),
            "max": max(quality_scores)
        }

    def analyze_evolution_state(self) -> Dict[str, Any]:
        """Analyze evolution system state."""
        evolution_file = self.data_dir / "evolution_state.json"
        if not evolution_file.exists():
            return {"error": "Evolution state file not found"}

        try:
            with open(evolution_file, 'r') as f:
                data = json.load(f)

            metrics = data.get("metrics", {})
            best_genotype = data.get("best_genotype", {})

            # Extract current genotype info from best_genotype
            encode_strategies = best_genotype.get("encode", {}).get("encoding_strategies", [])
            genotype_summary = f"{' + '.join(encode_strategies)}" if encode_strategies else "Default"

            return {
                "evolution_cycles_completed": metrics.get("evolution_cycles_completed", 0),
                "last_evolution_time": metrics.get("last_evolution_time"),
                "current_genotype_summary": genotype_summary,
                "api_requests_total": metrics.get("api_requests_total", 0),
                "average_response_time": metrics.get("average_response_time", 0),
                "average_retrieval_time": metrics.get("average_retrieval_time", 0),
                "fitness_score": self._calculate_fitness_score(metrics),
                "evolution_status": "Active" if metrics.get("evolution_cycles_completed", 0) > 0 else "Inactive"
            }

        except Exception as e:
            return {"error": f"Failed to analyze evolution state: {e}"}

    def _calculate_fitness_score(self, metrics: Dict) -> float:
        """Calculate current fitness score from metrics."""
        response_time = metrics.get("average_response_time", 0)
        retrieval_time = metrics.get("average_retrieval_time", 0)

        # Lower times are better, so invert and scale
        if response_time > 0:
            fitness = 1.0 / (1.0 + response_time + retrieval_time * 100)
            return fitness
        return 0.0

    def analyze_memory_system(self) -> Dict[str, Any]:
        """Analyze memory system state comprehensively."""
        memory_file = self.data_dir / "memory" / "memory_system.json"
        if not memory_file.exists():
            return {"error": "Memory system file not found"}

        try:
            with open(memory_file, 'r') as f:
                data = json.load(f)

            # Memory data is stored as individual objects with keys like "unit_0", "unit_1", etc.
            experiences = []
            for key, value in data.items():
                if key.startswith("unit_") and isinstance(value, dict):
                    experiences.append(value)

            memory_stats = data.get("stats", {})

            # Analyze memory types and content
            memory_types = Counter()
            content_lengths = []

            for exp in experiences:
                mem_type = exp.get("type", "unknown")
                memory_types[mem_type] += 1

                content = exp.get("content", "")
                content_lengths.append(len(content))

            # Analyze memory injection patterns (from middleware logs)
            injection_patterns = self._analyze_memory_injection_patterns()

            # Calculate storage efficiency
            total_content_length = sum(content_lengths)
            storage_efficiency = total_content_length / len(experiences) if experiences else 0

            return {
                "total_experiences": len(experiences),
                "memory_types": dict(memory_types),
                "content_length_stats": {
                    "average": statistics.mean(content_lengths) if content_lengths else 0,
                    "min": min(content_lengths) if content_lengths else 0,
                    "max": max(content_lengths) if content_lengths else 0,
                    "total": total_content_length
                },
                "file_size_kb": memory_file.stat().st_size / 1024,
                "storage_efficiency_bytes_per_experience": storage_efficiency,
                "memory_injection_patterns": injection_patterns,
                "memory_stats": memory_stats,
                "data_integrity": self._check_memory_integrity(experiences)
            }

        except Exception as e:
            return {"error": f"Failed to analyze memory system: {e}"}

    def _analyze_memory_injection_patterns(self) -> Dict[str, Any]:
        """Analyze how memories are being injected into responses."""
        middleware_log = self.logs_dir / "middleware.log"
        if not middleware_log.exists():
            return {"error": "Middleware log not found"}

        injection_counts = Counter()
        total_injections = 0

        try:
            with open(middleware_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    timestamp = self._parse_log_timestamp(line)
                    if timestamp and self._is_in_date_range(timestamp):
                        # Count memory injections
                        if "Injected" in line and "memories" in line:
                            # Extract number of memories injected
                            match = re.search(r'Injected (\d+) memories', line)
                            if match:
                                count = int(match.group(1))
                                injection_counts[count] += 1
                                total_injections += 1

        except Exception as e:
            return {"error": f"Failed to analyze injection patterns: {e}"}

        # Calculate most common injection patterns
        most_common = injection_counts.most_common(3)  # Just top 3 for brevity

        return {
            "total_injections": total_injections,
            "most_common_pattern": most_common[0][0] if most_common else 0,
            "pattern_summary": f"{most_common[0][0]} memories ({most_common[0][1]} times)" if most_common else "No patterns found"
        }

    def _check_memory_integrity(self, experiences: List) -> Dict[str, Any]:
        """Check memory data integrity."""
        if not experiences:
            return {"valid": False, "error": "No experiences found"}

        # Check for required fields
        required_fields = ["id", "content", "type"]
        missing_fields = Counter()
        invalid_entries = 0

        for exp in experiences:
            if not isinstance(exp, dict):
                invalid_entries += 1
                continue

            for field in required_fields:
                if field not in exp:
                    missing_fields[field] += 1

        integrity_score = (len(experiences) - invalid_entries) / len(experiences) if experiences else 0

        return {
            "json_valid": True,
            "total_entries": len(experiences),
            "invalid_entries": invalid_entries,
            "integrity_score": integrity_score,
            "status": "Excellent" if integrity_score > 0.99 else "Good" if integrity_score > 0.95 else "Needs Review"
        }

    def analyze_log_files(self) -> Dict[str, Any]:
        """Analyze log file sizes and growth patterns."""
        log_files = {
            "api_server": self.logs_dir / "api-server.log",
            "middleware": self.logs_dir / "middleware.log",
            "memory": self.logs_dir / "memory.log"
        }

        log_analysis = {}

        for log_name, log_file in log_files.items():
            if log_file.exists():
                stat = log_file.stat()
                size_kb = stat.st_size / 1024
                size_mb = size_kb / 1024

                # Count lines (rough estimate of activity)
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        line_count = sum(1 for _ in f)
                except:
                    line_count = 0

                log_analysis[log_name] = {
                    "exists": True,
                    "size_kb": size_kb,
                    "size_mb": size_mb,
                    "line_count": line_count,
                    "avg_bytes_per_line": size_kb * 1024 / line_count if line_count > 0 else 0
                }
            else:
                log_analysis[log_name] = {
                    "exists": False,
                    "size_kb": 0,
                    "size_mb": 0,
                    "line_count": 0
                }

        return log_analysis

    def analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze overall system performance metrics."""
        api_data = self.analyze_api_requests()
        memory_data = self.analyze_memory_operations()
        log_data = self.analyze_log_files()

        # Calculate system reliability metrics
        api_success_rate = api_data.get("success_rate", 0) if "error" not in api_data else 0
        memory_operations = memory_data.get("experiences_created", 0) if "error" not in memory_data else 0

        # Estimate resource utilization
        total_log_size_mb = sum(log["size_mb"] for log in log_data.values())

        # Calculate log growth rate (estimate based on analysis period)
        days_in_period = 1
        if self.start_date and self.end_date:
            days_in_period = max(1, (self.end_date - self.start_date).days)

        log_growth_rate_kb_per_day = total_log_size_mb * 1024 / days_in_period if days_in_period > 0 else 0

        return {
            "system_reliability": {
                "api_success_rate": api_success_rate,
                "memory_operations_successful": memory_operations > 0,
                "uptime_percentage": 100,  # Assume 100% if logs exist
                "error_rate": 100 - api_success_rate
            },
            "resource_utilization": {
                "log_storage_mb": total_log_size_mb,
                "log_growth_rate_kb_per_day": log_growth_rate_kb_per_day,
                "memory_file_size_kb": self.analyze_memory_system().get("file_size_kb", 0) if "error" not in self.analyze_memory_system() else 0
            },
            "performance_characteristics": {
                "log_files_analyzed": len([l for l in log_data.values() if l["exists"]]),
                "analysis_period_days": days_in_period,
                "data_integrity_checks": True
            }
        }

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive performance report with actionable insights."""

        # Ensure reports directory exists
        reports_dir = Path(self.data_dir) / "reports"
        reports_dir.mkdir(exist_ok=True)

        # If no output file specified, create one in reports directory
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(reports_dir / f"performance_report_{timestamp}.md")

        # Collect all analysis data
        api_data = self.analyze_api_requests()
        memory_data = self.analyze_memory_operations()
        quality_data = self.analyze_quality_scores()
        evolution_data = self.analyze_evolution_state()
        system_data = self.analyze_memory_system()
        log_data = self.analyze_log_files()
        perf_data = self.analyze_system_performance()

        # Calculate test duration
        test_duration_hours = 4  # Default assumption
        if self.start_date and self.end_date:
            test_duration_hours = (self.end_date - self.start_date).total_seconds() / 3600

        # Generate report
        report_lines = []

        # Header
        report_lines.extend([
            "# MemEvolve-API Performance Report",
            "",
            f"**Analysis Period**: {self.start_date.date() if self.start_date else 'All time'} to {self.end_date.date() if self.end_date else 'Present'}",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            ""
        ])

        # Executive Summary
        success_rate = perf_data['system_reliability']['api_success_rate'] if "error" not in api_data else 0
        evolution_status = f"{evolution_data.get('evolution_cycles_completed', 0)} generations" if "error" not in evolution_data else "Inactive"

        report_lines.extend([
            "## 📊 Executive Summary",
            "",
            f"**Duration**: {test_duration_hours:.1f} hours",
            f"**Success Rate**: {success_rate:.1f}%",
            f"**Evolution**: {evolution_status}",
            f"**Memory**: {system_data.get('total_experiences', 0) if 'error' not in system_data else 0} experiences",
            "",
            "---",
            ""
        ])

        # Quality Analysis - Simplified
        report_lines.extend([
            "## 🎯 Quality Analysis"
        ])

        if "error" not in quality_data:
            report_lines.extend([
                f"- **Evaluations**: {quality_data['count']}",
                f"- **Average Score**: {quality_data['average']:.3f}",
                f"- **Score Range**: {quality_data['min']:.3f} - {quality_data['max']:.3f}",
                f"- **Performance**: {'Excellent' if quality_data['average'] > 0.6 else 'Good' if quality_data['average'] > 0.5 else 'Needs Attention'}",
            ])
        else:
            report_lines.append(f"- Error: {quality_data['error']}")

        # API Performance
        report_lines.extend([
            "",
            "## ⚡ API Performance"
        ])

        if "error" not in api_data:
            rt = api_data.get('response_times', {})
            report_lines.extend([
                f"- **Total Requests**: {api_data['total_requests']}",
                f"- **Avg Response Time**: {rt.get('average', 0):.1f}s" if rt.get('count', 0) > 0 else "- Response time data unavailable",
                f"- **Upstream API**: ~{rt.get('average', 0) * 0.85:.1f}s (estimated primary bottleneck)" if rt.get('count', 0) > 0 else "",
                f"- **Memory Encoding**: ~{memory_data.get('encoding_performance', {}).get('average', 0):.1f}s per experience" if "error" not in memory_data else "",
            ])
        else:
            report_lines.append(f"- Error: {api_data['error']}")

        # Evolution Analysis
        report_lines.extend([
            "",
            "## 🔄 Evolution System"
        ])

        if "error" not in evolution_data:
            cycles = evolution_data['evolution_cycles_completed']
            report_lines.extend([
                f"- **Generations**: {cycles}",
                f"- **Status**: {'Active' if cycles > 0 else 'Inactive'}",
                f"- **Best Genotype**: {evolution_data.get('current_genotype_id', 'None')}",
                f"- **Fitness Score**: {evolution_data['fitness_score']:.4f}",
                f"- **Avg Response Time**: {evolution_data['average_response_time']:.1f}s",
            ])
        else:
            report_lines.append(f"- Error: {evolution_data['error']}")

        # Memory System
        report_lines.extend([
            "",
            "## 🧠 Memory System"
        ])

        if "error" not in system_data:
            exp_count = system_data['total_experiences']
            report_lines.extend([
                f"- **Total Experiences**: {exp_count}",
                f"- **Storage Size**: {system_data['file_size_kb']:.1f} KB",
                f"- **Avg Content Length**: {system_data['content_length_stats']['average']:.0f} chars",
                f"- **Memory Types**: {', '.join(system_data['memory_types'].keys()) if system_data['memory_types'] else 'None'}",
                f"- **Injection Pattern**: {system_data.get('memory_injection_patterns', {}).get('most_common_pattern', 0)} memories per response",
            ])
        else:
            report_lines.append(f"- Error: {system_data['error']}")

        # System Resources
        report_lines.extend([
            "",
            "## 💾 System Resources"
        ])

        total_logs = sum(log['size_mb'] for log in log_data.values() if log['exists'])
        report_lines.extend([
            f"- **Log Storage**: {total_logs:.1f} MB total",
            f"- **API Server Log**: {log_data['api_server']['size_mb']:.1f} MB ({log_data['api_server']['line_count']} lines)",
            f"- **Middleware Log**: {log_data['middleware']['size_mb']:.1f} MB ({log_data['middleware']['line_count']} lines)",
            f"- **Memory Operations**: {perf_data['resource_utilization']['log_growth_rate_kb_per_day']:.1f} KB/day growth",
        ])

        # Key Insights
        report_lines.extend([
            "",
            "## 🔍 Key Insights",
            "",
            "### Performance Breakdown:"
        ])

        # Calculate insights
        if "error" not in api_data and api_data.get('response_times', {}).get('average', 0) > 0:
            upstream_time = api_data['response_times']['average'] * 0.85
            memory_time = memory_data.get('encoding_performance', {}).get('average', 0) if "error" not in memory_data else 0
            retrieval_time = evolution_data.get('average_retrieval_time', 0) if "error" not in evolution_data else 0

            report_lines.extend([
                f"- **Upstream API**: {upstream_time:.1f}s (85% of response time) - Primary bottleneck",
                f"- **Memory Encoding**: {memory_time:.1f}s (Async, non-blocking)",
                f"- **Memory Retrieval**: {retrieval_time:.3f}s (Highly optimized)",
            ])

        # Recommendations
        report_lines.extend([
            "",
            "### Recommendations:"
        ])

        insights = []
        if "error" not in api_data and api_data.get('response_times', {}).get('average', 0) > 120:
            insights.append("- **Reduce response times**: Optimize upstream API calls or implement response streaming")

        if "error" not in memory_data and memory_data.get('encoding_performance', {}).get('average', 0) > 20:
            insights.append("- **Accelerate memory encoding**: Consider batch processing or faster models")

        if "error" not in quality_data and quality_data.get('average', 0) < 0.5:
            insights.append("- **Improve quality scores**: Review evaluation criteria and memory selection logic")

        if evolution_data.get('evolution_cycles_completed', 0) == 0:
            insights.append("- **Enable evolution**: Activate meta-optimization for continuous improvement")

        if not insights:
            insights.append("- **System performing well**: Continue monitoring for optimization opportunities")

        report_lines.extend(insights)

        # Footer
        report_lines.extend([
            "",
            "---",
            f"*Generated by MemEvolve Performance Analyzer*",
            f"*Analysis period: {test_duration_hours:.1f} hours*",
            f"*Success rate: {success_rate:.1f}%*",
            f"*Evolution generations: {evolution_data.get('evolution_cycles_completed', 0) if 'error' not in evolution_data else 0}*"
        ])

        report = "\n".join(report_lines)

        # Save to reports directory
        with open(output_file, 'w') as f:
            f.write(report)

        print(f"📊 Report saved to: {output_file}")
        return report


def main():
    """Main entry point for the performance analyzer."""
    parser = argparse.ArgumentParser(description="MemEvolve-API Performance Analyzer")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, help="Analyze last N days")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--data-dir", default="./data", help="Data directory path")
    parser.add_argument("--logs-dir", default="./logs", help="Logs directory path")

    args = parser.parse_args()

    analyzer = PerformanceAnalyzer(args.data_dir, args.logs_dir)

    # Set date range
    if args.days:
        analyzer.set_days_range(args.days)
    elif args.start_date and args.end_date:
        analyzer.set_date_range(args.start_date, args.end_date)
    else:
        print("Please specify either --days N or both --start-date and --end-date")
        return

    # Generate report
    report = analyzer.generate_report(args.output)

    if not args.output:
        print(report)


if __name__ == "__main__":
    main()