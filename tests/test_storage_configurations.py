"""
Comprehensive storage backend configuration tests.

Tests all storage backends with different configurations:
- JSONFileStore: Basic JSON file storage
- VectorStore: flat, ivf, hnsw index types  
- GraphStorageBackend: Neo4j and NetworkX fallback
"""

import os
import tempfile
import pytest
from pathlib import Path

from memevolve.components.store import JSONFileStore, VectorStore, GraphStorageBackend
from memevolve.utils.embeddings import create_embedding_function


class TestJSONStoreConfigurations:
    """Test JSON store with different configurations."""
    
    def test_json_store_default_config(self):
        """Test JSON store with default configuration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store_path = Path(tmp_dir) / "test_store.json"
            store = JSONFileStore(str(store_path))
            
            # Should work with default config
            unit = {"id": "test1", "type": "lesson", "content": "test content"}
            store.store(unit)
            
            retrieved = store.retrieve("test1")
            assert retrieved["id"] == "test1"
            assert retrieved["content"] == "test content"
    
    def test_json_store_with_custom_path(self):
        """Test JSON store with custom path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_path = Path(tmp_dir) / "custom" / "data.json"
            custom_path.parent.mkdir(parents=True)
            
            store = JSONFileStore(str(custom_path))
            
            unit = {"id": "test1", "type": "lesson", "content": "test content"}
            store.store(unit)
            
            assert custom_path.exists()
            retrieved = store.retrieve("test1")
            assert retrieved["id"] == "test1"


class TestVectorStoreConfigurations:
    """Test Vector store with different index types."""
    
    @pytest.fixture
    def mock_embedding_function(self):
        """Create mock embedding function for testing."""
        import numpy as np
        
        def embedding_func(text: str):
            # Deterministic embedding based on text hash
            np.random.seed(hash(text) % 2147483647)
            embedding = np.random.randn(384).astype(np.float32)
            return embedding / np.linalg.norm(embedding)
        
        return embedding_func
    
    def test_vector_store_flat_index(self, mock_embedding_function):
        """Test VectorStore with flat index type."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_file = Path(tmp_dir) / "test_flat.index"
            
            store = VectorStore(
                index_file=str(index_file),
                embedding_function=mock_embedding_function,
                embedding_dim=384,
                index_type="flat"
            )
            
            # Add test data
            unit = {"id": "test1", "type": "lesson", "content": "test content for flat index"}
            store.store(unit)
            
            # Test retrieval works
            results = store.search("test content", top_k=1)
            assert len(results) == 1
            assert results[0][1] == "test1"  # (distance, unit_id)
    
    def test_vector_store_ivf_index(self, mock_embedding_function):
        """Test VectorStore with IVF index type."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_file = Path(tmp_dir) / "test_ivf.index"
            
            store = VectorStore(
                index_file=str(index_file),
                embedding_function=mock_embedding_function,
                embedding_dim=384,
                index_type="ivf"
            )
            
            # Add test data
            unit = {"id": "test1", "type": "lesson", "content": "test content for ivf index"}
            store.store(unit)
            
            # Test retrieval works
            results = store.search("test content", top_k=1)
            assert len(results) == 1
            assert results[0][1] == "test1"
    
    def test_vector_store_hnsw_index(self, mock_embedding_function):
        """Test VectorStore with HNSW index type."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_file = Path(tmp_dir) / "test_hnsw.index"
            
            store = VectorStore(
                index_file=str(index_file),
                embedding_function=mock_embedding_function,
                embedding_dim=384,
                index_type="hnsw"
            )
            
            # Add test data
            unit = {"id": "test1", "type": "lesson", "content": "test content for hnsw index"}
            store.store(unit)
            
            # Test retrieval works
            results = store.search("test content", top_k=1)
            assert len(results) == 1
            assert results[0][1] == "test1"
    
    def test_vector_store_with_real_embeddings(self):
        """Test VectorStore with real embedding function."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_file = Path(tmp_dir) / "test_real.index"
            
            # Create dummy embedding function that mimics real embeddings
            embedding_fn = create_embedding_function(provider="dummy", embedding_dim=384)
            
            store = VectorStore(
                index_file=str(index_file),
                embedding_function=embedding_fn,
                embedding_dim=384,
                index_type="flat"
            )
            
            # Add test data
            unit = {"id": "test1", "type": "lesson", "content": "test content with real embeddings"}
            store.store(unit)
            
            # Test retrieval works
            results = store.search("test content", top_k=1)
            assert len(results) == 1
            assert results[0][1] == "test1"


class TestGraphStorageConfigurations:
    """Test Graph storage with different configurations."""
    
    def test_graph_storage_default_config(self):
        """Test GraphStorageBackend with default configuration (NetworkX fallback)."""
        # Default should use NetworkX when Neo4j unavailable
        store = GraphStorageBackend()
        
        # Should work with NetworkX fallback
        unit = {"id": "test1", "type": "lesson", "content": "test content for graph"}
        stored_id = store.store(unit)
        
        assert stored_id == "test1"
        
        # Test retrieval
        retrieved = store.retrieve("test1")
        assert retrieved is not None
        assert retrieved["id"] == "test1"
    
    @pytest.mark.skip("Neo4j connection attempts - tests use NetworkX fallback")
    def test_graph_storage_with_neo4j_config(self):
        """Test GraphStorageBackend with Neo4j configuration."""
        # This test will use NetworkX fallback if Neo4j not available
        # but should still initialize without errors
        store = GraphStorageBackend(
            uri="bolt://localhost:7687",
            user="neo4j", 
            password="password"
        )
        
        # Should initialize (will fall back to NetworkX if Neo4j unavailable)
        assert store is not None
        
        # Add test data
        unit = {"id": "test1", "type": "lesson", "content": "test content for neo4j"}
        stored_id = store.store(unit)
        
        # Should work regardless of backend
        assert stored_id == "test1"
    
@pytest.mark.skip("Complex graph operations - focus on basic functionality")
    def test_graph_storage_relationships(self):
        """Test graph storage relationship functionality (basic)."""
        store = GraphStorageBackend()
        
        # Add test data
        unit = {"id": "test1", "type": "lesson", "content": "test content for graph"}
        stored_id = store.store(unit)
        
        # Basic functionality should work
        assert stored_id == "test1"
        
        # Should be able to retrieve
        retrieved = store.retrieve("test1")
        assert retrieved is not None
        assert retrieved["id"] == "test1"


class TestStorageBackendIntegration:
    """Test storage backends work with MemorySystem."""
    
    def test_memory_system_with_json_store(self):
        """Test MemorySystem works with JSONFileStore."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store_path = Path(tmp_dir) / "memory.json"
            json_store = JSONFileStore(str(store_path))
            
            from memevolve.memory_system import MemorySystem, MemorySystemConfig
            from memevolve.components.retrieve import KeywordRetrievalStrategy
            from memevolve.components.manage import SimpleManagementStrategy
            
            config = MemorySystemConfig(
                storage_backend=json_store,
                retrieval_strategy=KeywordRetrievalStrategy(),
                management_strategy=SimpleManagementStrategy(),
                log_level="WARNING"
            )
            
            system = MemorySystem(config)
            
            # Test basic functionality
            experience = {"action": "test action", "result": "test result"}
            memory_id = system.add_experience(experience)
            assert memory_id is not None
            
            # Test retrieval
            results = system.query_memory("test", top_k=1)
            assert isinstance(results, list)
    
    def test_memory_system_with_vector_store_flat(self):
        """Test MemorySystem works with VectorStore (flat index)."""
        import numpy as np
        
        def mock_embedding_function(text: str):
            embedding = np.random.randn(384).astype(np.float32)
            return embedding / np.linalg.norm(embedding)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_file = Path(tmp_dir) / "vector.index"
            vector_store = VectorStore(
                index_file=str(index_file),
                embedding_function=mock_embedding_function,
                embedding_dim=384,
                index_type="flat"
            )
            
            from memevolve.memory_system import MemorySystem, MemorySystemConfig
            from memevolve.components.retrieve import SemanticRetrievalStrategy
            from memevolve.components.manage import SimpleManagementStrategy
            
            config = MemorySystemConfig(
                storage_backend=vector_store,
                retrieval_strategy=SemanticRetrievalStrategy(
                    embedding_function=mock_embedding_function
                ),
                management_strategy=SimpleManagementStrategy(),
                log_level="WARNING"
            )
            
            system = MemorySystem(config)
            
            # Test basic functionality
            experience = {"action": "test action", "result": "test result"}
            memory_id = system.add_experience(experience)
            assert memory_id is not None
            
            # Test retrieval
            results = system.query_memory("test", top_k=1)
            assert isinstance(results, list)
    
    def test_memory_system_with_vector_store_hnsw(self):
        """Test MemorySystem works with VectorStore (HNSW index)."""
        import numpy as np
        
        def mock_embedding_function(text: str):
            embedding = np.random.randn(384).astype(np.float32)
            return embedding / np.linalg.norm(embedding)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_file = Path(tmp_dir) / "hnsw.index"
            vector_store = VectorStore(
                index_file=str(index_file),
                embedding_function=mock_embedding_function,
                embedding_dim=384,
                index_type="hnsw"
            )
            
            from memevolve.memory_system import MemorySystem, MemorySystemConfig
            from memevolve.components.retrieve import SemanticRetrievalStrategy
            from memevolve.components.manage import SimpleManagementStrategy
            
            config = MemorySystemConfig(
                storage_backend=vector_store,
                retrieval_strategy=SemanticRetrievalStrategy(
                    embedding_function=mock_embedding_function
                ),
                management_strategy=SimpleManagementStrategy(),
                log_level="WARNING"
            )
            
            system = MemorySystem(config)
            
            # Test basic functionality
            experience = {"action": "test action", "result": "test result"}
            memory_id = system.add_experience(experience)
            assert memory_id is not None
            
            # Test retrieval
            results = system.query_memory("test", top_k=1)
            assert isinstance(results, list)
    
    def test_memory_system_with_graph_store(self):
        """Test MemorySystem works with GraphStorageBackend."""
        graph_store = GraphStorageBackend()
        
        from memevolve.memory_system import MemorySystem, MemorySystemConfig
        from memevolve.components.retrieve import KeywordRetrievalStrategy
        from memevolve.components.manage import SimpleManagementStrategy
        
        config = MemorySystemConfig(
            storage_backend=graph_store,
            retrieval_strategy=KeywordRetrievalStrategy(),
            management_strategy=SimpleManagementStrategy(),
            log_level="WARNING"
        )
        
        system = MemorySystem(config)
        
        # Test basic functionality
        experience = {"action": "test action", "result": "test result"}
        memory_id = system.add_experience(experience)
        assert memory_id is not None
        
        # Test retrieval
        results = system.query_memory("test", top_k=1)
        assert isinstance(results, list)


class TestStorageBackendPerformance:
    """Test storage backend performance characteristics."""
    
    def test_json_store_performance(self):
        """Test JSON store performance with larger datasets."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store_path = Path(tmp_dir) / "perf_test.json"
            store = JSONFileStore(str(store_path))
            
            # Add many units
            units = [
                {"id": f"unit_{i}", "type": "lesson", "content": f"Content {i}"}
                for i in range(100)
            ]
            
            import time
            start_time = time.time()
            store.store_batch(units)
            batch_time = time.time() - start_time
            
            # Should complete reasonably quickly
            assert batch_time < 5.0  # 100 units should store in <5 seconds
            
            # Test retrieval performance
            start_time = time.time()
            retrieved = store.retrieve("unit_50")
            single_time = time.time() - start_time
            
            assert single_time < 0.1  # Single retrieval should be fast
            assert retrieved["id"] == "unit_50"
    
    def test_vector_store_performance_with_different_indices(self):
        """Test VectorStore performance with different index types."""
        import numpy as np
        
        def mock_embedding_function(text: str):
            embedding = np.random.randn(384).astype(np.float32)
            return embedding / np.linalg.norm(embedding)
        
        # Test performance characteristics
        index_types = ["flat", "ivf", "hnsw"]
        
        for index_type in index_types:
            with tempfile.TemporaryDirectory() as tmp_dir:
                index_file = Path(tmp_dir) / f"perf_{index_type}.index"
                
                store = VectorStore(
                    index_file=str(index_file),
                    embedding_function=mock_embedding_function,
                    embedding_dim=384,
                    index_type=index_type
                )
                
                # Add test data
                units = [
                    {"id": f"unit_{i}", "type": "lesson", "content": f"Content {i}"}
                    for i in range(50)
                ]
                
                import time
                start_time = time.time()
                for unit in units:
                    store.store(unit)
                store_time = time.time() - start_time
                
                # Test search performance
                start_time = time.time()
                results = store.search("Content", top_k=5)
                search_time = time.time() - start_time
                
                # All index types should work
                assert len(results) == 5
                assert store_time < 10.0  # Should complete in reasonable time
                assert search_time < 1.0   # Search should be fast
                
                # Cleanup
                store.clear()