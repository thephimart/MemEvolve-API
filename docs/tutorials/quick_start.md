# MemEvolve Quick Start Tutorial

This tutorial will guide you through the basics of using MemEvolve, from installation to building your first memory-augmented application.

## Prerequisites

- Python 3.8+
- Access to an LLM API (OpenAI, vLLM, or compatible)
- Optional: Neo4j database for graph storage

## Installation

```bash
# Clone the repository
git clone https://github.com/thephimart/memevolve.git
cd memevole

# Install dependencies
pip install -r requirements.txt

# Optional: Install Neo4j for graph storage
pip install neo4j
```

## 1. Basic Memory System Setup

Let's start with the most basic MemEvolve setup using default configurations.

```python
from memevole import MemorySystem, MemorySystemConfig

# Configure with your LLM API
config = MemorySystemConfig(
    llm_base_url="http://localhost:8080/v1",  # Your LLM API endpoint
    llm_api_key="your-api-key-here",          # API authentication
    llm_model="your-model-name"               # Optional: specific model
)

# Create the memory system
memory = MemorySystem(config)

print("Memory system initialized successfully!")
```

### Environment Variables (Alternative)

You can also use environment variables for configuration:

```bash
export MEMEVOLVE_LLM_BASE_URL="http://localhost:8080/v1"
export MEMEVOLVE_LLM_API_KEY="your-api-key-here"
export MEMEVOLVE_LLM_MODEL="your-model-name"
```

```python
# Then simply create the system with defaults
memory = MemorySystem()
```

## 2. Adding Your First Memories

MemEvolve can learn from any experiences you provide. Let's add some programming-related experiences.

```python
# Add individual experiences
experiences = [
    {
        "action": "debug null pointer exception",
        "result": "found uninitialized variable in C++ code",
        "context": "working on embedded system firmware",
        "timestamp": "2024-01-15T10:30:00Z"
    },
    {
        "action": "optimize database query",
        "result": "added proper indexing, reduced query time from 5s to 0.2s",
        "context": "PostgreSQL database with 1M+ records",
        "timestamp": "2024-01-15T14:20:00Z"
    },
    {
        "action": "implement REST API authentication",
        "result": "used JWT tokens with refresh mechanism",
        "context": "Node.js Express application",
        "timestamp": "2024-01-16T09:15:00Z"
    }
]

# Add each experience to memory
memory_ids = []
for exp in experiences:
    memory_id = memory.add_experience(exp)
    memory_ids.append(memory_id)
    print(f"Added experience: {memory_id}")

print(f"Total memories stored: {len(memory_ids)}")
```

## 3. Querying Memories

Now that we have some memories stored, let's query them to retrieve relevant information.

```python
# Basic query - find debugging techniques
debug_results = memory.query_memory("debugging techniques", top_k=3)

print(f"Found {len(debug_results)} debugging-related memories:")
for i, result in enumerate(debug_results, 1):
    print(f"{i}. {result['type'].title()}: {result['content']}")
    print(f"   Tags: {', '.join(result.get('tags', []))}")
    print(f"   Relevance: {result.get('score', 'N/A'):.3f}")
    print()
```

### Advanced Queries with Filters

```python
# Query with type filtering
skill_results = memory.query_memory(
    "programming best practices",
    top_k=5,
    filters={"types": ["skill", "lesson"]}
)

print("Skills and lessons about programming:")
for result in skill_results:
    print(f"- {result['type']}: {result['content'][:60]}...")
```

## 4. Working with Trajectories

For more complex scenarios, you can add entire trajectories of related experiences.

```python
# A complete debugging session trajectory
debugging_trajectory = [
    {
        "id": "debug_step_1",
        "action": "run application",
        "result": "application crashes with segmentation fault",
        "context": "Linux environment, C++ application",
        "timestamp": "2024-01-17T10:00:00Z"
    },
    {
        "id": "debug_step_2",
        "action": "attach debugger",
        "result": "found crash at line 42 in memory allocation function",
        "context": "gdb debugger, stack trace shows heap corruption",
        "timestamp": "2024-01-17T10:15:00Z"
    },
    {
        "id": "debug_step_3",
        "action": "analyze memory allocation",
        "result": "discovered double-free in cleanup function",
        "context": "code review revealed race condition in multi-threaded code",
        "timestamp": "2024-01-17T10:45:00Z"
    },
    {
        "id": "debug_step_4",
        "action": "fix double-free bug",
        "result": "added proper mutex locking, application now stable",
        "context": "implemented thread-safe cleanup using std::mutex",
        "timestamp": "2024-01-17T11:30:00Z"
    }
]

# Add the entire trajectory
trajectory_ids = memory.add_trajectory(debugging_trajectory)
print(f"Added debugging trajectory: {len(trajectory_ids)} memories")

# Query about the debugging process
process_results = memory.query_memory("step-by-step debugging process")
print("
Debugging process memories:")
for result in process_results:
    print(f"- {result['content']}")
```

## 5. Using Different Storage Backends

MemEvolve supports multiple storage backends. Let's try the graph storage for relationship-aware queries.

```python
from components.store import GraphStorageBackend

# Create graph storage backend
graph_config = MemorySystemConfig(
    llm_base_url="http://localhost:8080/v1",
    llm_api_key="your-api-key",
    storage_backend=GraphStorageBackend(
        create_relationships=True  # Enable automatic relationship creation
    )
)

graph_memory = MemorySystem(graph_config)

# Add some related experiences
related_experiences = [
    {"action": "design database schema", "result": "normalized to 3NF", "tags": ["database", "design"]},
    {"action": "implement indexing", "result": "added B-tree indexes", "tags": ["database", "performance"]},
    {"action": "optimize queries", "result": "reduced execution time 10x", "tags": ["database", "optimization"]}
]

for exp in related_experiences:
    graph_memory.add_experience(exp)

# Query for related memories (graph traversal)
database_results = graph_memory.query_memory("database optimization")
print(f"\nFound {len(database_results)} database-related memories")

# If using graph storage, we can also query relationships
if hasattr(graph_memory.storage, 'query_related'):
    # Find memories related to database design
    db_memory_id = database_results[0]['id']
    related_memories = graph_memory.storage.query_related(db_memory_id, max_depth=2)

    print(f"\nMemories related to '{database_results[0]['content'][:30]}...':")
    for rel in related_memories[:3]:  # Show first 3
        print(f"- Related: {rel['unit']['content'][:50]}...")
        print(f"  Depth: {rel['depth']}, Type: {rel['relationship'].get('type', 'unknown')}")
```

## 6. Memory Management

MemEvolve automatically manages memory lifecycle, but you can also trigger manual management.

```python
# Check current memory statistics
memory_stats = memory.get_memory_stats()
print("Current memory statistics:")
for key, value in memory_stats.items():
    print(f"- {key}: {value}")

# Manually trigger memory management
print("\nTriggering memory management...")
memory.manage_memory()

# Check updated statistics
updated_stats = memory.get_memory_stats()
print("After management:")
for key, value in updated_stats.items():
    print(f"- {key}: {value}")
```

## 7. Exporting and Importing Memories

You can export memories for backup or sharing, and import them into other systems.

```python
# Export memories to JSON
exported_memories = memory.export_memories()
print(f"Exported {len(exported_memories)} memories")

# Save to file
import json
with open('my_memories.json', 'w') as f:
    json.dump(exported_memories, f, indent=2, default=str)

print("Memories exported to 'my_memories.json'")

# Later, you could import them into another system:
# new_memory = MemorySystem(config)
# imported_count = new_memory.import_memories(exported_memories)
# print(f"Imported {imported_count} memories")
```

## Next Steps

This tutorial covered the basics of MemEvolve. Here are some next steps:

1. **Advanced Retrieval**: Try different retrieval strategies (semantic, hybrid, LLM-guided)
2. **Custom Components**: Implement your own encoding, storage, or retrieval strategies
3. **Benchmarking**: Use the built-in evaluation framework to test your memory system
4. **Production Deployment**: Learn about configuration, monitoring, and scaling

For more advanced tutorials, see:
- [Advanced Memory Patterns](advanced_patterns.md)
- [Custom Component Development](custom_components.md)
- [Production Deployment](production_deployment.md)