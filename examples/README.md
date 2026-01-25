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
- Real-time fitness scoring and optimization

### graph_store_example.py
Shows graph storage backend capabilities:
- Neo4j integration for relationship-aware storage
- Automatic relationship creation between memories
- Graph traversal queries
- Evolution-aware graph storage strategies

### embedding_config.py
Illustrates custom embedding configurations:
- Setting up embedding functions for different providers
- Configuring vector stores with custom embeddings
- Dimension handling and optimization
- Adaptive embedding strategies for evolution

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
- **Custom evolution strategies** beyond built-in auto-evolution
- **Fine-grained fitness scoring** and optimization control

## 🚀 Evolution Features Available

The MemEvolve system includes intelligent auto-evolution with:
- Adaptive fitness scoring based on historical performance
- Multi-trigger automatic evolution (request count, performance degradation, plateau, time-based)
- Business impact validation with ROI tracking
- Comprehensive analytics for executive decision-making

For production API memory injection with built-in evolution and business analytics, use the API proxy approach instead.