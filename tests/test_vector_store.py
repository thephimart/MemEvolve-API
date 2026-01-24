import pytest
import tempfile
import os
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from memevolve.components.store.vector_store import VectorStore


@pytest.fixture
def mock_embedding_function():
    """Mock embedding function for testing."""
    def embedding_func(text: str) -> np.ndarray:
        # Return a fixed-size embedding based on text length (for testing)
        dim = 768
        # Create deterministic but varying embeddings
        seed = hash(text) % 1000
        np.random.seed(seed)
        embedding = np.random.rand(dim).astype(np.float32)
        # Normalize to make it more realistic
        return embedding / np.linalg.norm(embedding)
    return embedding_func


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test files."""
    return tmp_path


@pytest.fixture
def vector_store(temp_dir, mock_embedding_function):
    """Create a VectorStore instance for testing."""
    index_file = str(temp_dir / "test_index")
    return VectorStore(
        index_file=index_file,
        embedding_function=mock_embedding_function,
        embedding_dim=768
    )


class TestVectorStoreInitialization:
    """Test VectorStore initialization and setup."""

    def test_initialization_creates_index(self, temp_dir, mock_embedding_function):
        """Test that initialization creates a new index when file doesn't exist."""
        index_file = str(temp_dir / "new_index")
        store = VectorStore(
            index_file=index_file,
            embedding_function=mock_embedding_function,
            embedding_dim=768
        )

        assert store.index is not None
        assert len(store.data) == 0
        assert not os.path.exists(index_file + ".index")
        assert not os.path.exists(index_file + ".data")

    def test_initialization_loads_existing_files(self, temp_dir, mock_embedding_function):
        """Test that initialization loads existing index and data files."""
        index_file = str(temp_dir / "existing_index")

        # Create some initial data
        initial_store = VectorStore(
            index_file=index_file,
            embedding_function=mock_embedding_function,
            embedding_dim=768
        )

        # Add some data
        unit1 = {"id": "test1", "content": "test content 1", "type": "lesson"}
        unit2 = {"id": "test2", "content": "test content 2", "type": "skill"}
        initial_store.store(unit1)
        initial_store.store(unit2)

        # Create new instance - should load existing data
        new_store = VectorStore(
            index_file=index_file,
            embedding_function=mock_embedding_function,
            embedding_dim=768
        )

        assert len(new_store.data) == 2
        assert "test1" in new_store.data
        assert "test2" in new_store.data

    def test_initialization_with_wrong_embedding_dim(self, temp_dir):
        """Test initialization with mismatched embedding dimensions."""
        index_file = str(temp_dir / "test_index")

        def wrong_dim_embedding(text: str) -> np.ndarray:
            return np.random.rand(256).astype(np.float32)  # Wrong dimension

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            store = VectorStore(
                index_file=index_file,
                embedding_function=wrong_dim_embedding,
                embedding_dim=768
            )

            # Trigger dimension check by trying to store something
            unit = {"content": "test", "type": "lesson"}
            store.store(unit)


class TestVectorStoreCRUD:
    """Test basic CRUD operations."""

    def test_store_single_unit(self, vector_store):
        """Test storing a single memory unit."""
        unit = {
            "content": "This is a test memory unit",
            "type": "lesson",
            "tags": ["test", "memory"]
        }

        unit_id = vector_store.store(unit)

        assert unit_id is not None
        assert unit_id in vector_store.data
        assert vector_store.data[unit_id]["content"] == "This is a test memory unit"
        assert vector_store.data[unit_id]["type"] == "lesson"
        assert vector_store.data[unit_id]["tags"] == ["test", "memory"]
        assert "metadata" in vector_store.data[unit_id]

    def test_store_unit_with_custom_id(self, vector_store):
        """Test storing a unit with a custom ID."""
        custom_id = "custom_unit_123"
        unit = {
            "id": custom_id,
            "content": "Custom ID unit",
            "type": "skill"
        }

        unit_id = vector_store.store(unit)

        assert unit_id == custom_id
        assert custom_id in vector_store.data

    def test_retrieve_existing_unit(self, vector_store):
        """Test retrieving an existing unit."""
        unit = {"content": "Test content", "type": "lesson"}
        unit_id = vector_store.store(unit)

        retrieved = vector_store.retrieve(unit_id)

        assert retrieved is not None
        assert retrieved["id"] == unit_id
        assert retrieved["content"] == "Test content"

    def test_retrieve_nonexistent_unit(self, vector_store):
        """Test retrieving a non-existent unit."""
        retrieved = vector_store.retrieve("nonexistent_id")
        assert retrieved is None

    def test_update_existing_unit(self, vector_store):
        """Test updating an existing unit."""
        unit = {"content": "Original content", "type": "lesson"}
        unit_id = vector_store.store(unit)

        updated_unit = {"content": "Updated content", "type": "skill"}
        success = vector_store.update(unit_id, updated_unit)

        assert success is True
        retrieved = vector_store.retrieve(unit_id)
        assert retrieved["content"] == "Updated content"
        assert retrieved["type"] == "skill"
        assert "updated_at" in retrieved["metadata"]

    def test_update_nonexistent_unit(self, vector_store):
        """Test updating a non-existent unit."""
        unit = {"content": "Test content", "type": "lesson"}
        success = vector_store.update("nonexistent_id", unit)
        assert success is False

    def test_delete_existing_unit(self, vector_store):
        """Test deleting an existing unit."""
        unit = {"content": "Test content", "type": "lesson"}
        unit_id = vector_store.store(unit)

        success = vector_store.delete(unit_id)

        assert success is True
        assert vector_store.retrieve(unit_id) is None
        assert not vector_store.exists(unit_id)

    def test_delete_nonexistent_unit(self, vector_store):
        """Test deleting a non-existent unit."""
        success = vector_store.delete("nonexistent_id")
        assert success is False

    def test_exists_check(self, vector_store):
        """Test checking if units exist."""
        unit = {"content": "Test content", "type": "lesson"}
        unit_id = vector_store.store(unit)

        assert vector_store.exists(unit_id) is True
        assert vector_store.exists("nonexistent_id") is False

    def test_count_units(self, vector_store):
        """Test counting stored units."""
        assert vector_store.count() == 0

        vector_store.store({"content": "Unit 1", "type": "lesson"})
        assert vector_store.count() == 1

        vector_store.store({"content": "Unit 2", "type": "skill"})
        assert vector_store.count() == 2


class TestVectorStoreBatchOperations:
    """Test batch operations."""

    def test_store_batch_units(self, vector_store):
        """Test storing multiple units at once."""
        units = [
            {"content": "Batch unit 1", "type": "lesson"},
            {"content": "Batch unit 2", "type": "skill"},
            {"content": "Batch unit 3", "type": "tool"}
        ]

        ids = vector_store.store_batch(units)

        assert len(ids) == 3
        assert vector_store.count() == 3

        for i, unit_id in enumerate(ids):
            retrieved = vector_store.retrieve(unit_id)
            assert retrieved is not None
            assert retrieved["content"] == f"Batch unit {i+1}"

    def test_retrieve_all_units(self, vector_store):
        """Test retrieving all stored units."""
        units = [
            {"content": "Unit 1", "type": "lesson"},
            {"content": "Unit 2", "type": "skill"}
        ]

        for unit in units:
            vector_store.store(unit)

        all_units = vector_store.retrieve_all()
        assert len(all_units) == 2

        contents = [unit["content"] for unit in all_units]
        assert "Unit 1" in contents
        assert "Unit 2" in contents


class TestVectorStoreSearch:
    """Test search functionality."""

    def test_search_with_results(self, vector_store):
        """Test searching for similar units."""
        # Add some test units
        units = [
            {"content": "Python programming tutorial",
                "type": "lesson", "tags": ["python", "programming"]},
            {"content": "Machine learning basics",
                "type": "skill", "tags": ["ml", "ai"]},
            {"content": "Debugging techniques", "type": "tool",
                "tags": ["debugging", "development"]}
        ]

        for unit in units:
            vector_store.store(unit)

        # Search for programming-related content
        results = vector_store.search("python development", top_k=2)

        assert len(results) == 2
        # Results should be tuples of (distance, unit_id)
        for distance, unit_id in results:
            assert isinstance(distance, float)
            assert isinstance(unit_id, str)
            assert vector_store.exists(unit_id)

    def test_search_empty_index(self, vector_store):
        """Test searching when no units are stored."""
        results = vector_store.search("any query")
        assert results == []

    def test_search_top_k_larger_than_available(self, vector_store):
        """Test search when requesting more results than available."""
        vector_store.store({"content": "Single unit", "type": "lesson"})

        results = vector_store.search("query", top_k=5)
        assert len(results) == 1


class TestVectorStoreMetadata:
    """Test metadata handling."""

    def test_get_metadata_existing_unit(self, vector_store):
        """Test getting metadata for an existing unit."""
        unit = {"content": "Test content", "type": "lesson"}
        unit_id = vector_store.store(unit)

        metadata = vector_store.get_metadata(unit_id)

        assert metadata is not None
        assert "created_at" in metadata

    def test_get_metadata_nonexistent_unit(self, vector_store):
        """Test getting metadata for a non-existent unit."""
        metadata = vector_store.get_metadata("nonexistent_id")
        assert metadata is None


class TestVectorStorePersistence:
    """Test data persistence."""

    def test_persistence_across_instances(self, temp_dir, mock_embedding_function):
        """Test that data persists across store instances."""
        index_file = str(temp_dir / "persistent_index")

        # First instance
        store1 = VectorStore(
            index_file=index_file,
            embedding_function=mock_embedding_function,
            embedding_dim=768
        )

        unit = {"content": "Persistent content", "type": "lesson"}
        unit_id = store1.store(unit)

        # Second instance should load the data
        store2 = VectorStore(
            index_file=index_file,
            embedding_function=mock_embedding_function,
            embedding_dim=768
        )

        retrieved = store2.retrieve(unit_id)
        assert retrieved is not None
        assert retrieved["content"] == "Persistent content"

    def test_clear_operation(self, vector_store):
        """Test clearing all data."""
        # Add some data
        vector_store.store({"content": "Unit 1", "type": "lesson"})
        vector_store.store({"content": "Unit 2", "type": "skill"})

        assert vector_store.count() == 2

        # Clear
        vector_store.clear()

        assert vector_store.count() == 0
        assert len(vector_store.data) == 0


class TestVectorStoreErrorHandling:
    """Test error handling and edge cases."""

    def test_faiss_import_error(self, temp_dir, mock_embedding_function):
        """Test behavior when FAISS is not available."""
        index_file = str(temp_dir / "test_index")

        with patch.dict('sys.modules', {'faiss': None}):
            with pytest.raises(RuntimeError, match="Failed to create index"):
                VectorStore(
                    index_file=index_file,
                    embedding_function=mock_embedding_function,
                    embedding_dim=768
                )

    def test_index_load_failure(self, temp_dir, mock_embedding_function):
        """Test graceful handling of index load failures (falls back to new index)."""
        with patch('faiss.read_index') as mock_read_index:
            mock_read_index.side_effect = Exception("Load failed")

            index_file = str(temp_dir / "test_index")
            # Create fake index file
            Path(index_file + ".index").touch()

            # Should gracefully fall back to creating new index instead of raising error
            store = VectorStore(
                index_file=index_file,
                embedding_function=mock_embedding_function,
                embedding_dim=768
            )

            assert store.index is not None  # New index created
            assert len(store.data) == 0  # Empty data

    def test_data_load_failure(self, temp_dir, mock_embedding_function):
        """Test graceful handling of data load failures (starts with empty data)."""
        index_file = str(temp_dir / "test_index")

        # Create corrupted data file
        with open(index_file + ".data", 'wb') as f:
            f.write(b'corrupted data')

        # Should gracefully handle corrupted data by starting with empty data
        store = VectorStore(
            index_file=index_file,
            embedding_function=mock_embedding_function,
            embedding_dim=768
        )

        assert store.index is not None  # Index created
        assert len(store.data) == 0  # Empty data due to corruption


class TestVectorStoreInternalMethods:
    """Test internal helper methods."""

    def test_unit_to_text_conversion(self, vector_store):
        """Test converting units to text for embedding."""
        unit = {
            "content": "Main content",
            "tags": ["tag1", "tag2"],
            "type": "lesson"
        }

        text = vector_store._unit_to_text(unit)
        assert "Main content" in text
        assert "tag1" in text
        assert "tag2" in text
        assert "lesson" in text

    def test_unit_to_text_missing_fields(self, vector_store):
        """Test text conversion with missing fields."""
        unit = {"content": "Just content"}

        text = vector_store._unit_to_text(unit)
        assert text == "Just content"

    def test_rebuild_index(self, vector_store):
        """Test index rebuilding after data changes."""
        # Add some data
        unit1 = {"content": "Unit 1", "type": "lesson"}
        unit2 = {"content": "Unit 2", "type": "skill"}

        vector_store.store(unit1)
        vector_store.store(unit2)

        initial_count = vector_store.index.ntotal
        assert initial_count == 2

        # Delete one unit (should trigger rebuild)
        vector_store.delete(list(vector_store.data.keys())[0])

        # Index should be rebuilt with remaining unit
        assert vector_store.index.ntotal == 1
