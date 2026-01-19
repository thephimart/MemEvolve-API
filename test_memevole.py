#!/usr/bin/env python3
"""
MemEvolve Local Test Scenario
=============================

This script provides a complete test scenario for MemEvolve using mock data
and minimal LLM requirements. It demonstrates:
1. Basic memory system setup
2. Adding experiences
3. Querying memories
4. Memory management

Requirements:
- Python 3.8+
- LLM API (Ollama recommended for local testing)
- Set environment variables below

Run with: python test_memevole.py
"""

import os
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, using existing environment variables")
    print("   Install with: pip install python-dotenv")

from memory_system import MemorySystem, MemorySystemConfig
from components.retrieve import KeywordRetrievalStrategy

def main():
    print("🧠 MemEvolve Local Test Scenario")
    print("=" * 50)

    # Check environment setup
    required_vars = ["MEMEVOLVE_LLM_BASE_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nCreate a .env file in the project root with:")
        print("MEMEVOLVE_LLM_BASE_URL=http://localhost:11434/v1")
        print("MEMEVOLVE_LLM_API_KEY=ollama")
        print("\nOr set them manually with:")
        print("export MEMEVOLVE_LLM_BASE_URL='http://localhost:11434/v1'  # For Ollama")
        print("export MEMEVOLVE_LLM_API_KEY='ollama'                   # Or your API key")
        print("\nFor local testing with Ollama:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Run: ollama serve")
        print("3. Pull a model: ollama pull llama2")
        print("4. Install python-dotenv: pip install python-dotenv")
        return 1

    print("✅ Environment variables configured")

    try:
        # 1. Configure memory system with minimal settings
        print("\n📋 Step 1: Configuring Memory System")
        config = MemorySystemConfig(
            default_retrieval_top_k=3,
            retrieval_strategy=KeywordRetrievalStrategy(),  # Use keyword retrieval for JSON storage
            enable_auto_management=False,  # Disable for testing
            log_level="INFO"
        )

        memory = MemorySystem(config)
        print("✅ Memory system initialized")

        # 2. Add test experiences (software development scenario)
        print("\n📝 Step 2: Adding Test Experiences")
        test_experiences = [
            {
                "action": "debug authentication error",
                "result": "found JWT token expired, implemented refresh logic",
                "context": "React frontend calling Node.js API",
                "tags": ["authentication", "jwt", "frontend", "api"]
            },
            {
                "action": "optimize slow database query",
                "result": "added composite index on user_id + created_at, reduced query time from 2.5s to 0.15s",
                "context": "PostgreSQL database with 1M+ user records",
                "tags": ["database", "optimization", "indexing", "performance"]
            },
            {
                "action": "fix memory leak in Python service",
                "result": "identified circular references in callback handlers, added weakref.WeakMethod",
                "context": "Long-running async Python microservice",
                "tags": ["python", "memory", "async", "weakref"]
            },
            {
                "action": "implement API rate limiting",
                "result": "used Redis sliding window counter, prevented abuse while maintaining performance",
                "context": "REST API serving 1000+ requests/second",
                "tags": ["api", "rate-limiting", "redis", "scalability"]
            },
            {
                "action": "resolve CSS layout bug",
                "result": "fixed flexbox overflow issue with min-width constraint",
                "context": "Responsive web application layout",
                "tags": ["css", "layout", "flexbox", "responsive"]
            }
        ]

        memory_ids = []
        for i, exp in enumerate(test_experiences, 1):
            print(f"   Adding experience {i}/5...")
            memory_id = memory.add_experience(exp)
            memory_ids.append(memory_id)

        print(f"✅ Added {len(memory_ids)} experiences")

        # 3. Test different queries
        print("\n🔍 Step 3: Testing Memory Queries")

        test_queries = [
            "database performance issues",
            "authentication problems",
            "memory management in Python",
            "API security and rate limiting",
            "CSS layout debugging"
        ]

        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            results = memory.query_memory(query, top_k=2)

            if results:
                for i, result in enumerate(results, 1):
                    content = result.get('content', '')[:100] + "..." if len(result.get('content', '')) > 100 else result.get('content', '')
                    print(f"   {i}. {result.get('type', 'unknown').title()}: {content}")
                    print(".3f")
            else:
                print("   No relevant memories found")

        # 4. Test memory statistics
        print("\n📊 Step 4: Memory Statistics")
        try:
            stats = memory.get_memory_stats()
            print("Current memory status:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        except AttributeError:
            print("Memory statistics not available (expected in some configurations)")

        # 5. Test memory management
        print("\n🧹 Step 5: Memory Management")
        try:
            # Test different management operations
            operations = ["prune", "consolidate", "deduplicate"]
            for operation in operations:
                try:
                    result = memory.manage_memory(operation)
                    print(f"✅ {operation.capitalize()} operation completed")
                    if result and 'removed' in result:
                        print(f"   Removed {result['removed']} memories")
                except Exception as op_error:
                    print(f"⚠️  {operation.capitalize()} operation failed: {op_error}")

            print("✅ Memory management testing completed")
        except AttributeError:
            print("Memory management not available (expected in some configurations)")

        print("\n🎉 MemEvolve test scenario completed successfully!")
        print("\nNext steps:")
        print("- Try different storage backends (VectorStore, GraphStorageBackend)")
        print("- Experiment with custom retrieval strategies")
        print("- Run the benchmark evaluation framework")
        print("- Check the logs for detailed operation information")

        return 0

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("\nTroubleshooting:")
        print("1. Verify your LLM API is running and accessible")
        print("2. Check that the API endpoint accepts the expected request format")
        print("3. Ensure the model can handle the encoding prompts")
        print("4. Try with a smaller model if encoding fails")
        print("\nFor more detailed errors, set: export MEMEVOLVE_LOG_LEVEL=DEBUG")
        return 1

if __name__ == "__main__":
    exit(main())