"""
Data export/import utilities for MemEvolve memory systems.

This module provides utilities for exporting memory data to various formats
and importing data back into memory systems.
"""

from typing import Dict, List, Any, Union
from pathlib import Path
import json
import csv
from datetime import datetime, timezone

from .logging import get_logger


class MemoryDataExporter:
    """Exporter for memory system data."""

    def __init__(self):
        self.logger = get_logger("memory_exporter")

    def export_to_json(
        self,
        memory_system,
        filepath: Union[str, Path],
        include_metadata: bool = True
    ) -> bool:
        """Export memory data to JSON format.

        Args:
            memory_system: MemorySystem instance to export from
            filepath: Path to save the JSON file
            include_metadata: Whether to include system metadata

        Returns:
            True if export successful, False otherwise
        """
        try:
            filepath = Path(filepath)

            # Get all memory units from storage
            if hasattr(memory_system, 'storage') and hasattr(memory_system.storage, 'retrieve_all'):
                units = memory_system.storage.retrieve_all()
            else:
                self.logger.error("Memory system does not have accessible storage")
                return False

            export_data = {
                "export_info": {
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                    "total_units": len(units),
                    "format": "json"
                },
                "memory_units": units
            }

            if include_metadata:
                # Add system metadata if available
                metadata = {}
                if hasattr(memory_system, 'get_health_metrics'):
                    metadata["health"] = memory_system.get_health_metrics()
                if hasattr(memory_system, 'get_operation_log'):
                    metadata["operation_log"] = memory_system.get_operation_log()

                export_data["metadata"] = metadata

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Exported {len(units)} memory units to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export memory data: {str(e)}")
            return False

    def export_to_csv(
        self,
        memory_system,
        filepath: Union[str, Path],
        include_metadata: bool = False
    ) -> bool:
        """Export memory data to CSV format.

        Args:
            memory_system: MemorySystem instance to export from
            filepath: Path to save the CSV file
            include_metadata: Whether to include metadata columns

        Returns:
            True if export successful, False otherwise
        """
        try:
            filepath = Path(filepath)

            # Get all memory units from storage
            if hasattr(memory_system, 'storage') and hasattr(memory_system.storage, 'retrieve_all'):
                units = memory_system.storage.retrieve_all()
            else:
                self.logger.error("Memory system does not have accessible storage")
                return False

            if not units:
                self.logger.warning("No memory units to export")
                return False

            # Determine CSV columns from the first unit
            columns = ["id", "type", "content"]

            # Add additional fields that exist in units
            extra_fields = set()
            for unit in units[:10]:  # Sample first 10 units
                extra_fields.update(unit.keys())

            extra_fields.discard("id")
            extra_fields.discard("type")
            extra_fields.discard("content")

            if include_metadata:
                columns.extend(sorted(extra_fields))
            else:
                # Only include common non-metadata fields
                common_fields = ["tags", "embedding", "score"]
                columns.extend([f for f in common_fields if f in extra_fields])

            # Write CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()

                for unit in units:
                    row = {}
                    for col in columns:
                        value = unit.get(col, "")
                        # Convert complex types to strings for CSV
                        if isinstance(value, (list, dict)):
                            row[col] = json.dumps(value, ensure_ascii=False)
                        else:
                            row[col] = str(value) if value is not None else ""
                    writer.writerow(row)

            self.logger.info(f"Exported {len(units)} memory units to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export memory data: {str(e)}")
            return False

    def export_by_type(
        self,
        memory_system,
        base_filepath: Union[str, Path],
        format: str = "json"
    ) -> Dict[str, str]:
        """Export memory data grouped by type.

        Args:
            memory_system: MemorySystem instance to export from
            base_filepath: Base path for export files (will append type names)
            format: Export format ("json" or "csv")

        Returns:
            Dictionary mapping types to exported file paths
        """
        try:
            base_path = Path(base_filepath)

            # Get all memory units from storage
            if hasattr(memory_system, 'storage') and hasattr(memory_system.storage, 'retrieve_all'):
                units = memory_system.storage.retrieve_all()
            else:
                self.logger.error("Memory system does not have accessible storage")
                return {}

            # Group units by type
            units_by_type = {}
            for unit in units:
                unit_type = unit.get("type", "unknown")
                if unit_type not in units_by_type:
                    units_by_type[unit_type] = []
                units_by_type[unit_type].append(unit)

            exported_files = {}

            for unit_type, type_units in units_by_type.items():
                if format.lower() == "json":
                    filename = f"{base_path.stem}_{unit_type}.json"
                    filepath = base_path.parent / filename
                    success = self._export_units_to_json(type_units, filepath)
                elif format.lower() == "csv":
                    filename = f"{base_path.stem}_{unit_type}.csv"
                    filepath = base_path.parent / filename
                    success = self._export_units_to_csv(type_units, filepath)
                else:
                    self.logger.error(f"Unsupported format: {format}")
                    continue

                if success:
                    exported_files[unit_type] = str(filepath)

            self.logger.info(f"Exported data for {len(exported_files)} types")
            return exported_files

        except Exception as e:
            self.logger.error(f"Failed to export by type: {str(e)}")
            return {}

    def _export_units_to_json(self, units: List[Dict[str, Any]], filepath: Path) -> bool:
        """Helper method to export units to JSON."""
        try:
            export_data = {
                "export_info": {
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                    "total_units": len(units),
                    "format": "json"
                },
                "memory_units": units
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            self.logger.error(f"Failed to export units to {filepath}: {str(e)}")
            return False

    def _export_units_to_csv(self, units: List[Dict[str, Any]], filepath: Path) -> bool:
        """Helper method to export units to CSV."""
        try:
            if not units:
                return False

            # Determine columns
            columns = ["id", "type", "content"]
            extra_fields = set()
            for unit in units:
                extra_fields.update(unit.keys())
            extra_fields.discard("id")
            extra_fields.discard("type")
            extra_fields.discard("content")
            columns.extend(sorted(extra_fields))

            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()

                for unit in units:
                    row = {}
                    for col in columns:
                        value = unit.get(col, "")
                        if isinstance(value, (list, dict)):
                            row[col] = json.dumps(value, ensure_ascii=False)
                        else:
                            row[col] = str(value) if value is not None else ""
                    writer.writerow(row)
            return True
        except Exception as e:
            self.logger.error(f"Failed to export units to {filepath}: {str(e)}")
            return False


class MemoryDataImporter:
    """Importer for memory system data."""

    def __init__(self):
        self.logger = get_logger("memory_importer")

    def import_from_json(
        self,
        memory_system,
        filepath: Union[str, Path],
        skip_duplicates: bool = True
    ) -> Dict[str, int]:
        """Import memory data from JSON format.

        Args:
            memory_system: MemorySystem instance to import into
            filepath: Path to the JSON file
            skip_duplicates: Whether to skip units with existing IDs

        Returns:
            Dictionary with import statistics
        """
        try:
            filepath = Path(filepath)

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract memory units
            if "memory_units" in data:
                units = data["memory_units"]
            elif isinstance(data, list):
                units = data
            else:
                self.logger.error("Invalid JSON format: no memory_units found")
                return {"imported": 0, "skipped": 0, "errors": 1}

            return self._import_units(memory_system, units, skip_duplicates)

        except Exception as e:
            self.logger.error(f"Failed to import from JSON: {str(e)}")
            return {"imported": 0, "skipped": 0, "errors": 1}

    def import_from_csv(
        self,
        memory_system,
        filepath: Union[str, Path],
        skip_duplicates: bool = True
    ) -> Dict[str, int]:
        """Import memory data from CSV format.

        Args:
            memory_system: MemorySystem instance to import into
            filepath: Path to the CSV file
            skip_duplicates: Whether to skip units with existing IDs

        Returns:
            Dictionary with import statistics
        """
        try:
            filepath = Path(filepath)
            units = []

            with open(filepath, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    unit = {}
                    for key, value in row.items():
                        if value:
                            # Try to parse JSON strings back to objects
                            try:
                                unit[key] = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                unit[key] = value
                        else:
                            unit[key] = None
                    units.append(unit)

            return self._import_units(memory_system, units, skip_duplicates)

        except Exception as e:
            self.logger.error(f"Failed to import from CSV: {str(e)}")
            return {"imported": 0, "skipped": 0, "errors": 1}

    def import_from_directory(
        self,
        memory_system,
        directory: Union[str, Path],
        pattern: str = "*.json",
        skip_duplicates: bool = True
    ) -> Dict[str, int]:
        """Import memory data from all files in a directory.

        Args:
            memory_system: MemorySystem instance to import into
            directory: Directory containing data files
            pattern: File pattern to match (e.g., "*.json", "*.csv")
            skip_duplicates: Whether to skip units with existing IDs

        Returns:
            Dictionary with import statistics
        """
        try:
            directory = Path(directory)
            if not directory.is_dir():
                self.logger.error(f"Directory does not exist: {directory}")
                return {"imported": 0, "skipped": 0, "errors": 1}

            total_stats = {"imported": 0, "skipped": 0, "errors": 0, "files": 0}

            for filepath in directory.glob(pattern):
                total_stats["files"] += 1

                if filepath.suffix.lower() == ".json":
                    stats = self.import_from_json(memory_system, filepath, skip_duplicates)
                elif filepath.suffix.lower() == ".csv":
                    stats = self.import_from_csv(memory_system, filepath, skip_duplicates)
                else:
                    self.logger.warning(f"Unsupported file type: {filepath}")
                    continue

                total_stats["imported"] += stats.get("imported", 0)
                total_stats["skipped"] += stats.get("skipped", 0)
                total_stats["errors"] += stats.get("errors", 0)

            self.logger.info(
                f"Imported from {total_stats['files']} files: "
                f"{total_stats['imported']} units, "
                f"{total_stats['skipped']} skipped, "
                f"{total_stats['errors']} errors"
            )
            return total_stats

        except Exception as e:
            self.logger.error(f"Failed to import from directory: {str(e)}")
            return {"imported": 0, "skipped": 0, "errors": 1, "files": 0}

    def _import_units(
        self,
        memory_system,
        units: List[Dict[str, Any]],
        skip_duplicates: bool
    ) -> Dict[str, int]:
        """Helper method to import units into memory system."""
        imported = 0
        skipped = 0
        errors = 0

        for unit in units:
            try:
                unit_id = unit.get("id")

                # Check for duplicates if requested
                if (skip_duplicates and unit_id and
                        hasattr(memory_system, 'storage')):
                    if (hasattr(memory_system.storage, 'exists') and
                            memory_system.storage.exists(unit_id)):
                        skipped += 1
                        continue

                # Add the unit to memory system
                if hasattr(memory_system, 'add_experience'):
                    memory_system.add_experience(unit)
                elif hasattr(memory_system, 'storage') and hasattr(memory_system.storage, 'store'):
                    memory_system.storage.store(unit)
                else:
                    self.logger.error("Memory system does not support adding experiences")
                    errors += 1
                    continue

                imported += 1

            except Exception as e:
                self.logger.error(
                    f"Failed to import unit {unit.get('id', 'unknown')}: {str(e)}"
                )
                errors += 1

        self.logger.info(
            f"Import complete: {imported} imported, {skipped} skipped, {errors} errors"
        )
        return {"imported": imported, "skipped": skipped, "errors": errors}


# Convenience functions
def export_memory_data(
    memory_system,
    filepath: Union[str, Path],
    format: str = "json",
    **kwargs
) -> bool:
    """Convenience function to export memory data.

    Args:
        memory_system: MemorySystem instance to export from
        filepath: Path to save the exported data
        format: Export format ("json" or "csv")
        **kwargs: Additional arguments for the exporter

    Returns:
        True if export successful, False otherwise
    """
    exporter = MemoryDataExporter()

    if format.lower() == "json":
        return exporter.export_to_json(memory_system, filepath, **kwargs)
    elif format.lower() == "csv":
        return exporter.export_to_csv(memory_system, filepath, **kwargs)
    else:
        exporter.logger.error(f"Unsupported export format: {format}")
        return False


def import_memory_data(
    memory_system,
    filepath: Union[str, Path],
    **kwargs
) -> Dict[str, int]:
    """Convenience function to import memory data.

    Args:
        memory_system: MemorySystem instance to import into
        filepath: Path to the data file
        **kwargs: Additional arguments for the importer

    Returns:
        Dictionary with import statistics
    """
    importer = MemoryDataImporter()

    filepath = Path(filepath)
    if filepath.suffix.lower() == ".json":
        return importer.import_from_json(memory_system, filepath, **kwargs)
    elif filepath.suffix.lower() == ".csv":
        return importer.import_from_csv(memory_system, filepath, **kwargs)
    else:
        importer.logger.error(f"Unsupported import format: {filepath.suffix}")
        return {"imported": 0, "skipped": 0, "errors": 1}
