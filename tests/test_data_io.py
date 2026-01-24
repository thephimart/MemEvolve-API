from memevolve.utils.data_io import (
    MemoryDataExporter,
    MemoryDataImporter,
    export_memory_data,
    import_memory_data
)
import sys
import tempfile
import json
import csv
from pathlib import Path

# sys.path.insert(0, 'src')  # No longer needed with package structure


class MockStorage:
    """Mock storage backend for testing."""

    def __init__(self):
        self.data = {}

    def store(self, unit):
        unit_id = unit.get("id", f"unit_{len(self.data)}")
        if "id" not in unit:
            unit["id"] = unit_id
        self.data[unit_id] = unit
        return unit_id

    def retrieve_all(self):
        return list(self.data.values())

    def exists(self, unit_id):
        return unit_id in self.data


class MockMemorySystem:
    """Mock memory system for testing."""

    def __init__(self):
        self.storage = MockStorage()
        self.added_experiences = []

    def add_experience(self, experience):
        """Mock add experience method."""
        self.added_experiences.append(experience)
        return self.storage.store(experience)


def test_memory_data_exporter_initialization():
    """Test exporter initialization."""
    exporter = MemoryDataExporter()
    assert exporter is not None


def test_memory_data_importer_initialization():
    """Test importer initialization."""
    importer = MemoryDataImporter()
    assert importer is not None


def test_export_to_json():
    """Test JSON export functionality."""
    memory_system = MockMemorySystem()

    # Add some test data
    memory_system.add_experience({
        "id": "exp_001",
        "type": "lesson",
        "content": "Python basics",
        "tags": ["python", "programming"]
    })
    memory_system.add_experience({
        "id": "exp_002",
        "type": "skill",
        "content": "Java development",
        "tags": ["java", "development"]
    })

    exporter = MemoryDataExporter()

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "test_export.json"

        success = exporter.export_to_json(memory_system, filepath)
        assert success
        assert filepath.exists()

        # Verify content
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert "export_info" in data
        assert "memory_units" in data
        assert len(data["memory_units"]) == 2
        assert data["memory_units"][0]["type"] == "lesson"
        assert data["memory_units"][1]["type"] == "skill"


def test_export_to_csv():
    """Test CSV export functionality."""
    memory_system = MockMemorySystem()

    # Add some test data
    memory_system.add_experience({
        "id": "exp_001",
        "type": "lesson",
        "content": "Python basics",
        "tags": ["python", "programming"]
    })

    exporter = MemoryDataExporter()

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "test_export.csv"

        success = exporter.export_to_csv(memory_system, filepath)
        assert success
        assert filepath.exists()

        # Verify content
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["id"] == "exp_001"
        assert rows[0]["type"] == "lesson"
        assert rows[0]["content"] == "Python basics"


def test_export_by_type():
    """Test export by type functionality."""
    memory_system = MockMemorySystem()

    # Add test data of different types
    memory_system.add_experience({
        "id": "exp_001",
        "type": "lesson",
        "content": "Python basics"
    })
    memory_system.add_experience({
        "id": "exp_002",
        "type": "skill",
        "content": "Java development"
    })
    memory_system.add_experience({
        "id": "exp_003",
        "type": "lesson",
        "content": "Advanced Python"
    })

    exporter = MemoryDataExporter()

    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir) / "test_export"

        exported_files = exporter.export_by_type(
            memory_system, base_path, format="json")

        assert "lesson" in exported_files
        assert "skill" in exported_files
        assert len(exported_files) == 2

        # Verify lesson file
        with open(exported_files["lesson"], 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert len(data["memory_units"]) == 2

        # Verify skill file
        with open(exported_files["skill"], 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert len(data["memory_units"]) == 1


def test_import_from_json():
    """Test JSON import functionality."""
    memory_system = MockMemorySystem()
    importer = MemoryDataImporter()

    # Create test JSON data
    test_data = {
        "export_info": {
            "timestamp": "2024-01-19T12:00:00Z",
            "total_units": 2,
            "format": "json"
        },
        "memory_units": [
            {
                "id": "exp_001",
                "type": "lesson",
                "content": "Python basics",
                "tags": ["python"]
            },
            {
                "id": "exp_002",
                "type": "skill",
                "content": "Java development",
                "tags": ["java"]
            }
        ]
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "test_import.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)

        stats = importer.import_from_json(memory_system, filepath)

        assert stats["imported"] == 2
        assert stats["skipped"] == 0
        assert stats["errors"] == 0
        assert len(memory_system.added_experiences) == 2


def test_import_from_csv():
    """Test CSV import functionality."""
    memory_system = MockMemorySystem()
    importer = MemoryDataImporter()

    # Create test CSV data
    csv_data = [
        ["id", "type", "content", "tags"],
        ["exp_001", "lesson", "Python basics", '["python"]'],
        ["exp_002", "skill", "Java development", '["java"]']
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "test_import.csv"

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)

        stats = importer.import_from_csv(memory_system, filepath)

        assert stats["imported"] == 2
        assert stats["skipped"] == 0
        assert stats["errors"] == 0
        assert len(memory_system.added_experiences) == 2


def test_import_skip_duplicates():
    """Test import with duplicate skipping."""
    memory_system = MockMemorySystem()
    importer = MemoryDataImporter()

    # Add existing experience
    memory_system.add_experience({
        "id": "exp_001",
        "type": "lesson",
        "content": "Python basics"
    })

    # Create test data with duplicate
    test_data = {
        "memory_units": [
            {
                "id": "exp_001",  # Duplicate
                "type": "lesson",
                "content": "Python basics"
            },
            {
                "id": "exp_002",  # New
                "type": "skill",
                "content": "Java development"
            }
        ]
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "test_import.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)

        stats = importer.import_from_json(
            memory_system, filepath, skip_duplicates=True)

        assert stats["imported"] == 1  # Only the new one
        assert stats["skipped"] == 1  # The duplicate
        assert stats["errors"] == 0


def test_convenience_functions():
    """Test convenience export/import functions."""
    memory_system = MockMemorySystem()

    # Add test data
    memory_system.add_experience({
        "id": "exp_001",
        "type": "lesson",
        "content": "Python basics"
    })

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test export
        filepath = Path(temp_dir) / "test.json"
        success = export_memory_data(memory_system, filepath, format="json")
        assert success
        assert filepath.exists()

        # Test import into new system
        new_memory_system = MockMemorySystem()
        stats = import_memory_data(new_memory_system, filepath)
        assert stats["imported"] == 1
        assert len(new_memory_system.added_experiences) == 1


def test_import_from_directory():
    """Test importing from a directory."""
    memory_system = MockMemorySystem()
    importer = MemoryDataImporter()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create multiple JSON files
        file1_data = {
            "memory_units": [
                {"id": "exp_001", "type": "lesson", "content": "File 1 content"}
            ]
        }
        file2_data = {
            "memory_units": [
                {"id": "exp_002", "type": "skill", "content": "File 2 content"}
            ]
        }

        with open(temp_path / "file1.json", 'w', encoding='utf-8') as f:
            json.dump(file1_data, f)

        with open(temp_path / "file2.json", 'w', encoding='utf-8') as f:
            json.dump(file2_data, f)

        # Create a non-matching file
        with open(temp_path / "notes.txt", 'w', encoding='utf-8') as f:
            f.write("Not a JSON file")

        stats = importer.import_from_directory(
            memory_system, temp_path, pattern="*.json")

        assert stats["imported"] == 2
        assert stats["files"] == 2  # Only JSON files
        assert stats["errors"] == 0
