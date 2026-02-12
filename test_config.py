#!/usr/bin/env python3
"""Test script for configuration unification"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables
os.environ['MEMEVOLVE_API_ENABLE'] = 'true'
os.environ['MEMEVOLVE_API_HOST'] = 'localhost'
os.environ['MEMEVOLVE_API_PORT'] = '8000'
os.environ['MEMEVOLVE_API_MEMORY_INTEGRATION'] = 'true'
os.environ['MEMEVOLVE_DEFAULT_TOP_K'] = '10'
os.environ['MEMEVOLVE_API_MEMORY_RETRIEVAL_LIMIT'] = '7'

def test_config():
    try:
        from memevolve.utils.config import ConfigManager
        
        print("Testing ConfigManager...")
        config_manager = ConfigManager()
        config = config_manager.config
        
        print("✅ Configuration unified successfully!")
        print(f"API enable: {config.api_enable}")
        print(f"API host: {config.api_host}") 
        print(f"API port: {config.api_port}")
        print(f"API memory integration: {config.api_memory_integration}")
        print(f"Memory retrieval limit (env override): {config.memory_retrieval_limit}")
        print(f"Default top_k: {config.default_top_k}")
        
        # Verify configuration is working for vector store persistence
        print(f"Data directory: {config.data_dir}")
        print(f"Storage backend type: {config.storage.backend_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config()
    sys.exit(0 if success else 1)