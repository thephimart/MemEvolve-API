#!/usr/bin/env python3
"""
Basic MemEvolve Usage Example

This script demonstrates the fundamental operations of MemEvolve:
1. Setting up a memory system
2. Adding experiences
3. Querying memories
4. Basic memory management

Run this script to see MemEvolve in action:

    python examples/basic_usage.py
"""

import os
import sys
from pathlib import Path

# No longer needed with package structure - memevolve is installed as a package
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memevolve.memory_system import MemorySystem, MemorySystemConfig


def main():
    print("🧠 MemEvolve Basic Usage Example")
    print("=" * 40)

    # Check for required environment variables
    if not os.getenv("MEMEVOLVE_MEMORY_BASE_URL"):
        print("❌ Please set MEMEVOLVE_MEMORY_BASE_URL environment variable")
        print("Example: export MEMEVOLVE_MEMORY_BASE_URL='http://localhost:8080/v1'")
        return

    if not os.getenv("MEMEVOLVE_MEMORY_API_KEY"):
        print("❌ Please set MEMEVOLVE_MEMORY_API_KEY environment variable")
        return

    try:
        # 1. Configure the memory system
        print("\n📋 Step 1: Configuring Memory System")
        config = MemorySystemConfig(
            default_retrieval_top_k=3,
            enable_auto_management=True
        )

        # Create memory system (uses environment variables for LLM config)
        memory = MemorySystem(config)
        print("✅ Memory system initialized")

        # 2. Add some programming experiences
        print("\n📝 Step 2: Adding Programming Experiences")
        experiences = [
            {
                "action": "debug segmentation fault",
                "result": "found null pointer dereference in C code",
                "context": "Linux system programming",
                "tags": ["debugging", "c", "pointers"]
            },
            {
                "action": "optimize slow database query",
                "result": "added composite index, reduced query time from 2s to 0.1s",
                "context": "PostgreSQL database with 500k records",
                "tags": ["database", "optimization", "indexing"]
            },
            {
                "action": "implement authentication middleware",
                "result": "created JWT-based auth with refresh tokens",
                "context": "Node.js Express API",
                "tags": ["authentication", "jwt", "security"]
            },
            {
                "action": "fix memory leak in Python application",
                "result": "identified circular references in object graph",
                "context": "Long-running Python service",
                "tags": ["python", "memory", "garbage-collection"]
            }
        ]

        memory_ids = []
        for i, exp in enumerate(experiences, 1):
            memory_id = memory.add_experience(exp)
            memory_ids.append(memory_id)
            print(f"✅ Added experience {i}: {memory_id}")

        print(f"\n📊 Total experiences stored: {len(memory_ids)}")

        # 3. Query the memory system
        print("\n🔍 Step 3: Querying Memories")
        queries = [
            "debugging techniques",
            "database performance optimization",
            "web application security",
            "memory management in Python"
        ]

        for query in queries:
            print(f"\n🔍 Query: '{query}'")
            results = memory.query_memory(query, top_k=2)

            if results:
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result['type'].title()}: {result['content'][:80]}...")
                    print(".3f")
            else:
                print("  No relevant memories found")

        # 4. Check memory statistics
        print("\n📈 Step 4: Memory Statistics")
        try:
            stats = memory.get_memory_stats()
            print("Current memory status:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        except AttributeError:
            # get_memory_stats might not be implemented yet
            print("Memory statistics not available in this version")

        # 5. Demonstrate memory management
        print("\n🧹 Step 5: Memory Management")
        try:
            memory.manage_memory()
            print("✅ Memory management completed")
        except AttributeError:
            print("Memory management not available in this version")

        print("\n🎉 Basic MemEvolve example completed successfully!")
        print("\nNext steps:")
        print("- Try the advanced_patterns.py example")
        print("- Explore different storage backends (graph_store.py)")
        print("- Run the evaluation framework (see evaluation/ directory)")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your LLM API is running and accessible")
        print("2. Check that MEMEVOLVE_MEMORY_BASE_URL and MEMEVOLVE_MEMORY_API_KEY are set")
        print("3. Verify the API endpoint accepts the expected request format")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())