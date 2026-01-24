"""
Simple graph storage backend tests that work without Neo4j dependencies.

Tests GraphStorageBackend with NetworkX fallback functionality.
"""

import pytest
from memevolve.components.store import GraphStorageBackend


class TestGraphStorageBasic:
    """Test basic GraphStorageBackend functionality (NetworkX fallback)."""
    
    def test_graph_storage_basic_operations(self):
        """Test basic graph storage operations."""
        # Force NetworkX mode for testing (avoid Neo4j connection attempts)
        import os
        os.environ['MEMEVOLVE_GRAPH_DISABLE_NEO4J'] = 'true'
        
        # Should initialize with NetworkX fallback (no Neo4j connection)
        store = GraphStorageBackend()
        
        # Test storing a unit
        unit = {"id": "test1", "type": "lesson", "content": "test content for graph"}
        stored_id = store.store(unit)
        
        assert stored_id == "test1"
        
        # Test retrieval
        retrieved = store.retrieve("test1")
        assert retrieved is not None
        assert retrieved["id"] == "test1"
        assert retrieved["content"] == "test content for graph"
    
    def test_graph_storage_batch_operations(self):
        """Test batch operations in graph storage."""
        store = GraphStorageBackend()
        
        # Store multiple units
        units = [
            {"id": "test1", "type": "lesson", "content": "first content"},
            {"id": "test2", "type": "skill", "content": "second content"},
            {"id": "test3", "type": "tool", "content": "third content"}
        ]
        
        stored_ids = store.store_batch(units)
        
        # Should store all units
        assert len(stored_ids) == 3
        
        # Should be able to retrieve all
        all_units = store.retrieve_all()
        assert len(all_units) == 3
        
        # Check content
        unit_contents = {unit["id"] for unit in all_units}
        expected_contents = {"first content", "second content", "third content"}
        assert unit_contents == expected_contents
    
    def test_graph_storage_delete_operations(self):
        """Test delete operations in graph storage."""
        store = GraphStorageBackend()
        
        # Store and verify unit exists
        unit = {"id": "test_delete", "type": "lesson", "content": "to be deleted"}
        store.store(unit)
        
        assert store.exists("test_delete")
        
        # Delete unit
        success = store.delete("test_delete")
        assert success is True
        
        # Should not exist after deletion
        assert not store.exists("test_delete")
        
        # Retrieval should return None
        retrieved = store.retrieve("test_delete")
        assert retrieved is None
    
    def test_graph_storage_update_operations(self):
        """Test update operations in graph storage."""
        store = GraphStorageBackend()
        
        # Store initial unit
        unit = {"id": "test_update", "type": "lesson", "content": "original content"}
        store.store(unit)
        
        # Update unit
        updated_unit = {"id": "test_update", "type": "skill", "content": "updated content"}
        success = store.update("test_update", updated_unit)
        assert success is True
        
        # Retrieve and verify update
        retrieved = store.retrieve("test_update")
        assert retrieved is not None
        assert retrieved["content"] == "updated content"
        assert retrieved["type"] == "skill"
    
    def test_graph_storage_count_and_clear(self):
        """Test count and clear operations."""
        store = GraphStorageBackend()
        
        # Initially should be empty
        assert store.count() == 0
        
        # Add some units
        units = [
            {"id": "count1", "type": "lesson", "content": "first"},
            {"id": "count2", "type": "skill", "content": "second"}
        ]
        store.store_batch(units)
        
        # Should count correctly
        assert store.count() == 2
        
        # Clear storage
        store.clear()
        assert store.count() == 0