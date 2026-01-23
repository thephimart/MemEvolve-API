#!/usr/bin/env python3
"""
Graph Storage Backend Example

This script demonstrates MemEvolve's graph storage capabilities:
1. Using Neo4j for relationship-aware memory storage
2. Automatic relationship creation between similar memories
3. Graph traversal queries for finding related memories
4. Fallback to NetworkX when Neo4j is unavailable

Run this script to explore graph-based memory operations:

    python examples/graph_store_example.py
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory_system import MemorySystem, MemorySystemConfig
from components.store import GraphStorageBackend


def main():
    print("🕸️  MemEvolve Graph Storage Example")
    print("=" * 40)

    try:
        # 1. Configure graph storage backend
        print("\n📋 Step 1: Configuring Graph Storage")
        graph_backend = GraphStorageBackend(
            uri="bolt://localhost:7687",  # Neo4j bolt URL (will fallback if unavailable)
            create_relationships=True      # Enable automatic relationship creation
        )

        config = MemorySystemConfig(
            storage_backend=graph_backend,
            default_retrieval_top_k=3
        )

        memory = MemorySystem(config)
        print("✅ Graph-based memory system initialized")

        # 2. Add interconnected experiences
        print("\n📝 Step 2: Adding Interconnected Experiences")
        experiences = [
            # Database-related experiences (should be linked)
            {
                "action": "design normalized database schema",
                "result": "achieved 3NF with proper relationships",
                "context": "E-commerce platform with complex product catalog",
                "tags": ["database", "design", "normalization", "relationships"]
            },
            {
                "action": "optimize slow SQL query",
                "result": "added composite index, query time reduced from 3s to 0.1s",
                "context": "PostgreSQL database with user analytics",
                "tags": ["database", "optimization", "indexing", "performance"]
            },
            {
                "action": "implement database connection pooling",
                "result": "eliminated connection timeouts under high load",
                "context": "Python web application with high concurrency",
                "tags": ["database", "performance", "connection-pooling", "scalability"]
            },

            # Debugging experiences (different domain, should not be strongly linked)
            {
                "action": "debug race condition in multithreaded code",
                "result": "added proper mutex locks, eliminated data corruption",
                "context": "C++ networking library",
                "tags": ["debugging", "concurrency", "threading", "mutex"]
            },
            {
                "action": "trace memory leak in Python application",
                "result": "found circular references in callback handlers",
                "context": "Long-running web service",
                "tags": ["debugging", "memory", "python", "garbage-collection"]
            },

            # Algorithm experiences (some overlap with database)
            {
                "action": "implement efficient sorting algorithm",
                "result": "used Timsort variant, O(n log n) performance",
                "context": "Data processing pipeline",
                "tags": ["algorithm", "sorting", "performance", "optimization"]
            }
        ]

        memory_ids = []
        for i, exp in enumerate(experiences, 1):
            memory_id = memory.add_experience(exp)
            memory_ids.append(memory_id)
            print(f"✅ Added experience {i}: {memory_id[:16]}...")

        print(f"\n📊 Total experiences stored: {len(memory_ids)}")

        # 3. Query and explore relationships
        print("\n🔍 Step 3: Exploring Memory Relationships")

        # Query for database-related memories
        db_results = memory.query_memory("database optimization techniques", top_k=5)
        print(f"\n🔍 Database query results: {len(db_results)} memories")
        for i, result in enumerate(db_results, 1):
            print(f"  {i}. {result['content'][:60]}...")
            print(f"     Tags: {', '.join(result.get('tags', []))}")

        # Explore graph relationships if available
        if hasattr(memory.storage, 'query_related') and db_results:
            print("
🕸️  Exploring Graph Relationships")

            # Get the first database memory and find related ones
            first_db_memory = db_results[0]
            memory_id = first_db_memory['id']

            related = memory.storage.query_related(
                memory_id,
                max_depth=2,
                limit=5
            )

            print(f"Memories related to: '{first_db_memory['content'][:40]}...'")
            print(f"Found {len(related)} related memories:")

            for rel in related:
                mem = rel['unit']
                print(f"  • {mem['content'][:50]}... (depth: {rel['depth']})")
                if 'relationship' in rel and 'weight' in rel['relationship']:
                    print(".3f")

        # 4. Demonstrate different query types
        print("\n🔍 Step 4: Different Query Patterns")

        queries = [
            "debugging strategies",
            "performance optimization",
            "data structure algorithms",
            "web application development"  # Should have fewer results
        ]

        for query in queries:
            results = memory.query_memory(query, top_k=2)
            print(f"Query '{query}': {len(results)} results")

        # 5. Show storage statistics
        print("\n📈 Step 5: Graph Storage Statistics")
        try:
            stats = memory.storage.get_graph_stats()
            print("Graph storage statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

            if stats.get('storage_type') == 'networkx':
                print("\n💡 Note: Using NetworkX fallback. Install Neo4j for full graph capabilities:")
                print("   pip install neo4j")
                print("   # Then start Neo4j server")

        except AttributeError:
            print("Storage statistics not available")

        # 6. Demonstrate batch operations
        print("\n📦 Step 6: Batch Operations")
        batch_experiences = [
            {
                "action": "implement caching layer",
                "result": "added Redis cache, reduced API response time by 60%",
                "tags": ["caching", "performance", "redis"]
            },
            {
                "action": "design microservices architecture",
                "result": "created event-driven services with message queues",
                "tags": ["architecture", "microservices", "scalability"]
            }
        ]

        batch_ids = memory.add_trajectory_batch(batch_experiences)
        print(f"✅ Added {len(batch_ids)} experiences in batch")

        print("\n🎉 Graph storage example completed!")
        print("\nKey takeaways:")
        print("- Graph storage enables relationship-aware memory retrieval")
        print("- Similar memories are automatically linked based on content/tags")
        print("- Graph traversal can find indirectly related experiences")
        print("- Fallback to NetworkX provides development/testing capabilities")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nThis example requires a running LLM API.")
        print("Make sure MEMEVOLVE_MEMORY_BASE_URL and MEMEVOLVE_MEMORY_API_KEY are set.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())