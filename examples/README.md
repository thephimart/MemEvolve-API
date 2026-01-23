# MemEvolve Examples

These examples demonstrate **library usage** of MemEvolve components. They show how to use MemEvolve programmatically in your own applications.

## ⚠️ Not for API Proxy Usage

**If you want to add memory to existing OpenAI-compatible APIs**, use the **API proxy approach** instead:
- See the main [README.md](../README.md) for API proxy setup
- The API proxy is the recommended approach for most users
- No code changes needed in your existing applications

## 📚 Library Usage Examples

### basic_usage.py
Demonstrates fundamental MemorySystem operations:
- Creating and configuring a memory system
- Adding experiences
- Querying memories
- Basic memory management

### graph_store_example.py
Shows graph storage backend capabilities:
- Neo4j integration for relationship-aware storage
- Automatic relationship creation between memories
- Graph traversal queries

### embedding_config.py
Illustrates custom embedding configurations:
- Setting up embedding functions for different providers
- Configuring vector stores with custom embeddings
- Dimension handling and optimization

## 🚀 Running Examples

```bash
# Install dependencies
pip install -r requirements.txt

# Run an example
python examples/basic_usage.py
```

## 📖 When to Use Library Examples

Use these examples when you need:
- **Custom memory architectures** not available via API proxy
- **Direct integration** into existing Python applications
- **Advanced memory operations** requiring programmatic control
- **Research and experimentation** with memory system components

For production API memory injection, use the API proxy approach instead.