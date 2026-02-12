#!/usr/bin/env python3
"""Direct test of vector store persistence"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables
os.environ['MEMEVOLVE_STORAGE_BACKEND_TYPE'] = 'vector'
os.environ['MEMEVOLVE_EMBEDDING_PROVIDER'] = 'dummy'

def test_vector_store_persistence():
    """Test vector store directly to isolate the issue"""
    try:
        from memevolve.components.store import VectorStore
        from memevolve.utils.embeddings import DummyEmbeddingProvider
        
        # Create test directory
        test_dir = Path("/tmp/test_vector_persistence")
        test_dir.mkdir(exist_ok=True)
        
        index_file = str(test_dir / "test_vector")
        
        print(f"Creating VectorStore with index_file: {index_file}")
        
        # Create vector store with dummy embedding function
        embedding_provider = DummyEmbeddingProvider(embedding_dim=384)
        vector_store = VectorStore(
            index_file=index_file,
            embedding_function=embedding_provider.get_embedding,
            embedding_dim=384,  # Dummy provider dimension
            index_type="ivf_flat"
        )
        
        # Test storing a memory unit
        test_memory = {
            'id': 'test_1',
            'type': 'lesson',
            'content': 'This is a test memory about persistence',
            'tags': ['test', 'persistence'],
            'metadata': {
                'created_at': '2026-02-12T17:00:00Z',
                'category': 'test',
                'encoding_method': 'test',
                'quality_score': 0.8
            }
        }
        
        print("Storing test memory...")
        vector_store.store(test_memory)
        
        # Check if files were created
        expected_index_file = index_file + ".index"
        expected_data_file = index_file + ".data"
        
        print(f"Checking for files:")
        print(f"  Index file: {expected_index_file}")
        print(f"  Data file: {expected_data_file}")
        
        index_exists = os.path.exists(expected_index_file)
        data_exists = os.path.exists(expected_data_file)
        
        print(f"  Index file exists: {index_exists}")
        print(f"  Data file exists: {data_exists}")
        
        if index_exists and data_exists:
            print("✅ Vector store persistence working!")
            
            # Test retrieval
            print("Testing retrieval...")
            results = vector_store.search("test memory", top_k=5)
            print(f"Retrieved {len(results)} results")
            
            return True
        else:
            print("❌ Vector store persistence failed!")
            
            # List directory contents
            if test_dir.exists():
                print(f"Directory contents: {list(test_dir.glob('*'))}")
            
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if 'test_dir' in locals() and test_dir.exists():
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    success = test_vector_store_persistence()
    sys.exit(0 if success else 1)