"""
Debugging utilities for MemEvolve memory systems.

This module provides tools for inspecting, analyzing, and debugging memory system
components and operations.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime, timezone

from .logging import get_logger

try:
    from memory_system import MemorySystem
except ImportError:
    # Handle case where memory_system might not be available during testing
    MemorySystem = None


class MemoryInspector:
    """Inspector for memory system components and data."""

    def __init__(self, memory_system: MemorySystem):
        self.memory_system = memory_system
        self.logger = get_logger("memory_inspector")

    def get_system_overview(self) -> Dict[str, Any]:
        """Get a comprehensive overview of the memory system state."""
        overview = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "system_info": {
                "type": type(self.memory_system).__name__,
                "components_initialized": {
                    "encoder": hasattr(self.memory_system, 'encoder') and self.memory_system.encoder is not None,
                    "retriever": hasattr(self.memory_system, 'retriever') and self.memory_system.retriever is not None,
                    "storage": hasattr(self.memory_system, 'storage') and self.memory_system.storage is not None,
                    "manager": hasattr(self.memory_system, 'memory_manager') and self.memory_system.memory_manager is not None
                }
            }
        }

        # Add health metrics if available
        try:
            health = self.memory_system.get_health_metrics()
        except Exception as e:
            self.logger.warning(f"Failed to get health metrics: {str(e)}")
            health = None

        if health:
            overview["health_metrics"] = {
                "total_units": health.total_units,
                "total_size_bytes": health.total_size_bytes,
                "average_unit_size": health.average_unit_size,
                "duplicate_count": health.duplicate_count,
                "unit_types_distribution": health.unit_types_distribution,
                "last_operation": health.last_operation,
                "last_operation_time": health.last_operation_time
            }

        # Add operation log summary
        if hasattr(self.memory_system, 'operation_log'):
            operations = self.memory_system.operation_log
            overview["operation_log"] = {
                "total_operations": len(operations),
                "recent_operations": operations[-5:] if operations else []
            }

        return overview

    def inspect_memory_contents(
        self,
        limit: int = 50,
        unit_type: Optional[str] = None,
        include_content: bool = False
    ) -> Dict[str, Any]:
        """Inspect memory contents with filtering and pagination."""
        try:
            storage = getattr(self.memory_system, 'storage', None)
            if not storage or not hasattr(storage, 'retrieve_all'):
                return {"error": "Storage backend not accessible"}

            all_units = storage.retrieve_all()

            # Filter by type if specified
            if unit_type:
                filtered_units = [
                    u for u in all_units if u.get("type") == unit_type]
            else:
                filtered_units = all_units

            # Sort by creation time (newest first)
            filtered_units.sort(
                key=lambda u: u.get("metadata", {}).get("created_at", ""),
                reverse=True
            )

            # Limit results
            limited_units = filtered_units[:limit]

            # Prepare summary
            summary = {
                "total_units": len(all_units),
                "filtered_units": len(filtered_units),
                "returned_units": len(limited_units),
                "unit_types": {}
            }

            # Count unit types
            for unit in all_units:
                unit_type_count = unit.get("type", "unknown")
                summary["unit_types"][unit_type_count] = summary["unit_types"].get(
                    unit_type_count, 0) + 1

            # Prepare unit details
            units_detail = []
            for unit in limited_units:
                unit_info = {
                    "id": unit.get("id"),
                    "type": unit.get("type"),
                    "tags": unit.get("tags", []),
                    "created_at": unit.get("metadata", {}).get("created_at"),
                    "size_bytes": len(json.dumps(unit, default=str).encode('utf-8'))
                }

                if include_content:
                    unit_info["content"] = unit.get("content", "")
                    unit_info["full_data"] = unit

                units_detail.append(unit_info)

            summary["units"] = units_detail

            return summary

        except Exception as e:
            self.logger.error(f"Failed to inspect memory contents: {str(e)}")
            return {"error": str(e)}

    def analyze_component_states(self) -> Dict[str, Any]:
        """Analyze the state of all memory system components."""
        analysis = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "components": {}
        }

        # Analyze encoder
        encoder_info = {"initialized": False, "type": None}
        if hasattr(self.memory_system, 'encoder'):
            encoder = getattr(self.memory_system, 'encoder', None)
            encoder_info["initialized"] = encoder is not None
            encoder_info["type"] = type(encoder).__name__ if encoder else None

            if encoder and hasattr(encoder, 'get_metrics'):
                try:
                    metrics = encoder.get_metrics()
                    encoder_info["metrics"] = {
                        "total_encodings": getattr(metrics, 'total_encodings', 0),
                        "successful_encodings": getattr(metrics, 'successful_encodings', 0),
                        "failed_encodings": getattr(metrics, 'failed_encodings', 0),
                        "success_rate": getattr(metrics, 'success_rate', 0.0),
                        "average_time": getattr(metrics, 'average_encoding_time', 0.0)
                    }
                except Exception as e:
                    encoder_info["metrics_error"] = str(e)

        analysis["components"]["encoder"] = encoder_info

        # Analyze retriever
        retriever_info = {"initialized": False, "type": None}
        if hasattr(self.memory_system, 'retriever'):
            retriever = getattr(self.memory_system, 'retriever', None)
            retriever_info["initialized"] = retriever is not None
            retriever_info["type"] = type(
                retriever).__name__ if retriever else None

            if retriever and hasattr(retriever, 'get_metrics'):
                try:
                    metrics = retriever.get_metrics()
                    retriever_info["metrics"] = {
                        "total_retrievals": getattr(metrics, 'total_retrievals', 0),
                        "successful_retrievals": getattr(metrics, 'successful_retrievals', 0),
                        "failed_retrievals": getattr(metrics, 'failed_retrievals', 0),
                        "success_rate": getattr(metrics, 'calculate_success_rate', lambda: 0.0)(),
                        "average_time": getattr(metrics, 'average_retrieval_time', 0.0)
                    }
                except Exception as e:
                    retriever_info["metrics_error"] = str(e)

        analysis["components"]["retriever"] = retriever_info

        # Analyze storage
        if hasattr(self.memory_system, 'storage'):
            storage_info = {
                "initialized": self.memory_system.storage is not None,
                "type": type(self.memory_system.storage).__name__ if self.memory_system.storage else None
            }

            if self.memory_system.storage:
                try:
                    storage_info["unit_count"] = self.memory_system.storage.count()
                except Exception as e:
                    storage_info["count_error"] = str(e)

            analysis["components"]["storage"] = storage_info

        # Analyze manager
        if hasattr(self.memory_system, 'memory_manager'):
            manager_info = {
                "initialized": self.memory_system.memory_manager is not None,
                "type": type(self.memory_system.memory_manager).__name__ if self.memory_system.memory_manager else None
            }

            if self.memory_system.memory_manager and hasattr(self.memory_system.memory_manager, 'get_health_metrics'):
                try:
                    health = self.memory_system.memory_manager.get_health_metrics()
                    manager_info["health_metrics"] = {
                        "total_units": health.total_units,
                        "duplicate_count": health.duplicate_count,
                        "last_operation": health.last_operation
                    }
                except Exception as e:
                    manager_info["health_error"] = str(e)

            analysis["components"]["manager"] = manager_info

        return analysis

    def find_similar_units(self, target_unit_id: str, limit: int = 10) -> Dict[str, Any]:
        """Find units similar to a target unit."""
        try:
            storage = getattr(self.memory_system, 'storage', None)
            if not storage:
                return {"error": "Storage backend not accessible"}

            # Get target unit
            target_unit = storage.retrieve(target_unit_id)
            if not target_unit:
                return {"error": f"Unit {target_unit_id} not found"}

            # Get all units for comparison
            all_units = storage.retrieve_all()
            if len(all_units) <= 1:
                return {"error": "Insufficient units for similarity analysis"}

            # Simple similarity based on content overlap (can be enhanced with embeddings)
            target_content = target_unit.get("content", "").lower()
            target_tags = set(target_unit.get("tags", []))
            target_type = target_unit.get("type", "")

            similarities = []

            for unit in all_units:
                if unit.get("id") == target_unit_id:
                    continue

                score = 0
                reasons = []

                # Type match
                if unit.get("type") == target_type:
                    score += 0.3
                    reasons.append("same_type")

                # Tag overlap
                unit_tags = set(unit.get("tags", []))
                tag_overlap = len(target_tags & unit_tags)
                if tag_overlap > 0:
                    score += min(tag_overlap * 0.2, 0.4)
                    reasons.append(f"tag_overlap_{tag_overlap}")

                # Content similarity (simple word overlap)
                unit_content = unit.get("content", "").lower()
                target_words = set(target_content.split())
                unit_words = set(unit_content.split())

                if target_words and unit_words:
                    word_overlap = len(target_words & unit_words)
                    overlap_ratio = word_overlap / \
                        len(target_words | unit_words)
                    score += overlap_ratio * 0.3
                    if overlap_ratio > 0.1:
                        reasons.append(".2f")

                similarities.append({
                    "unit_id": unit.get("id"),
                    "unit_type": unit.get("type"),
                    "similarity_score": score,
                    "reasons": reasons,
                    "content_preview": unit.get("content", "")[:100] + "..." if len(unit.get("content", "")) > 100 else unit.get("content", "")
                })

            # Sort by similarity score
            similarities.sort(
                key=lambda x: x["similarity_score"], reverse=True)
            top_similar = similarities[:limit]

            return {
                "target_unit": {
                    "id": target_unit_id,
                    "type": target_type,
                    "content_preview": target_content[:100] + "..." if len(target_content) > 100 else target_content,
                    "tags": list(target_tags)
                },
                "similar_units": top_similar,
                "total_candidates": len(all_units) - 1
            }

        except Exception as e:
            self.logger.error(f"Failed to find similar units: {str(e)}")
            return {"error": str(e)}

    def export_debug_report(self, filepath: Union[str, Path]) -> bool:
        """Export a comprehensive debug report to file."""
        try:
            filepath = Path(filepath)

            report = {
                "debug_report": {
                    "generated_at": datetime.now(timezone.utc).isoformat() + "Z",
                    "system_overview": self.get_system_overview(),
                    "component_analysis": self.analyze_component_states(),
                    "memory_contents_summary": self.inspect_memory_contents(limit=10, include_content=False)
                }
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"Debug report exported to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export debug report: {str(e)}")
            return False


class MemoryDebugger:
    """Interactive debugging tools for memory systems."""

    def __init__(self):
        self.logger = get_logger("memory_debugger")
        self.inspectors: Dict[str, MemoryInspector] = {}

    def add_inspector(self, name: str, memory_system: MemorySystem):
        """Add a memory system inspector."""
        self.inspectors[name] = MemoryInspector(memory_system)
        self.logger.info(f"Added inspector for memory system: {name}")

    def get_inspector(self, name: str) -> Optional[MemoryInspector]:
        """Get a memory system inspector by name."""
        return self.inspectors.get(name)

    def compare_systems(self, system_names: List[str]) -> Dict[str, Any]:
        """Compare multiple memory systems."""
        if not all(name in self.inspectors for name in system_names):
            missing = [
                name for name in system_names if name not in self.inspectors]
            return {"error": f"Inspectors not found for systems: {missing}"}

        comparison = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "systems_compared": system_names,
            "comparison": {}
        }

        # Compare system overviews
        overviews = {}
        for name in system_names:
            overviews[name] = self.inspectors[name].get_system_overview()

        comparison["comparison"]["system_overviews"] = overviews

        # Compare component states
        component_states = {}
        for name in system_names:
            component_states[name] = self.inspectors[name].analyze_component_states(
            )

        comparison["comparison"]["component_states"] = component_states

        # Compare memory statistics
        memory_stats = {}
        for name in system_names:
            overview = overviews[name]
            health = overview.get("health_metrics", {})
            memory_stats[name] = {
                "total_units": health.get("total_units", 0),
                "total_size_bytes": health.get("total_size_bytes", 0),
                "duplicate_count": health.get("duplicate_count", 0)
            }

        comparison["comparison"]["memory_statistics"] = memory_stats

        return comparison

    def generate_system_report(self, system_name: str, include_details: bool = False) -> Dict[str, Any]:
        """Generate a detailed report for a specific system."""
        if system_name not in self.inspectors:
            return {"error": f"Inspector not found for system: {system_name}"}

        inspector = self.inspectors[system_name]

        report = {
            "system_name": system_name,
            "generated_at": datetime.now(timezone.utc).isoformat() + "Z",
            "system_overview": inspector.get_system_overview(),
            "component_analysis": inspector.analyze_component_states()
        }

        if include_details:
            report["memory_contents"] = inspector.inspect_memory_contents(
                limit=100, include_content=True)

        return report

    def list_systems(self) -> List[str]:
        """List all registered memory systems."""
        return list(self.inspectors.keys())


# Convenience functions
def inspect_memory_system(memory_system: MemorySystem, name: str = "default") -> MemoryInspector:
    """Create and return a memory inspector for the given system."""
    return MemoryInspector(memory_system)


def quick_debug_report(memory_system: MemorySystem, filepath: Union[str, Path]) -> bool:
    """Generate a quick debug report for a memory system."""
    inspector = MemoryInspector(memory_system)
    return inspector.export_debug_report(filepath)
