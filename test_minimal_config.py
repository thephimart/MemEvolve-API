#!/usr/bin/env python3
"""Minimal test for vector store persistence after config unification"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Mock environment variables
os.environ['MEMEVOLVE_API_ENABLE'] = 'true'
os.environ['MEMEVOLVE_API_HOST'] = 'localhost'
os.environ['MEMEVOLVE_API_PORT'] = '8000'
os.environ['MEMEVOLVE_API_MEMORY_INTEGRATION'] = 'true'
os.environ['MEMEVOLVE_DEFAULT_TOP_K'] = '5'
os.environ['MEMEVOLVE_STORAGE_BACKEND_TYPE'] = 'vector'
os.environ['MEMEVOLVE_EMBEDDING_PROVIDER'] = 'dummy'

def test_config_unification():
    """Test that configuration is unified without APIConfig"""
    try:
        # Import just the config classes
        from memevolve.utils.config import MemEvolveConfig, UpstreamConfig
        
        print("Testing configuration unification...")
        
        # Test 1: MemEvolveConfig has direct API fields
        config = MemEvolveConfig()
        
        assert hasattr(config, 'api_enable'), "Missing api_enable field"
        assert hasattr(config, 'api_host'), "Missing api_host field"
        assert hasattr(config, 'api_port'), "Missing api_port field"
        assert hasattr(config, 'api_memory_integration'), "Missing api_memory_integration field"
        assert hasattr(config, 'memory_retrieval_limit'), "Missing memory_retrieval_limit field"
        
        print("✅ Direct API fields exist in MemEvolveConfig")
        
        # Test 2: API fields load from environment
        assert config.api_enable == True, "api_enable not loaded from env"
        assert config.api_host == 'localhost', "api_host not loaded from env"
        assert config.api_port == 8000, "api_port not loaded from env"
        assert config.api_memory_integration == True, "api_memory_integration not loaded from env"
        
        print("✅ API fields load from environment variables")
        
        # Test 3: memory_retrieval_limit uses default_top_k
        assert config.memory_retrieval_limit == config.default_top_k, "memory_retrieval_limit should use default_top_k"
        
        print("✅ memory_retrieval_limit uses default_top_k as default")
        
        # Test 4: No APIConfig dependency
        try:
            from memevolve.utils.config import APIConfig
            print("❌ APIConfig still exists - should be removed")
            return False
        except ImportError:
            print("✅ APIConfig successfully removed")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config_unification()
    if success:
        print("\n🎉 Configuration unification completed successfully!")
        print("   - APIConfig eliminated")
        print("   - API fields moved to MemEvolveConfig")
        print("   - Environment variables working")
        print("   - memory_retrieval_limit uses default_top_k")
        print("   - Ready for vector store persistence testing")
    
    sys.exit(0 if success else 1)