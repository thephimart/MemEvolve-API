# MemEvolve Troubleshooting Guide

This guide helps you diagnose and resolve common issues with MemEvolve. If you can't find a solution here, please check the [GitHub Issues](https://github.com/thephimart/MemEvolve-API/issues) or create a new issue.

## 🚨 Quick Diagnosis

Run this diagnostic script to check your MemEvolve setup:

```python
#!/usr/bin/env python3
"""MemEvolve Diagnostic Script"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_diagnostics():
    print("🔍 MemEvolve Diagnostic Report")
    print("=" * 40)

    issues = []

    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 8):
        issues.append("❌ Python 3.8+ required")

    # Check environment variables
    required_env_vars = ["MEMEVOLVE_MEMORY_BASE_URL", "MEMEVOLVE_MEMORY_API_KEY"]
    print("\n🔑 Environment Variables:")
    for var in required_env_vars:
        value = os.getenv(var)
        status = "✅ Set" if value else "❌ Missing"
        print(f"  {var}: {status}")
        if not value:
            issues.append(f"❌ {var} not set")

    # Check optional environment variables
    optional_env_vars = ["MEMEVOLVE_MEMORY_MODEL", "MEMEVOLVE_LOG_LEVEL"]
    for var in optional_env_vars:
        value = os.getenv(var)
        status = f"Set to '{value}'" if value else "Using default"
        print(f"  {var}: {status}")

    # Check imports
    print("\n📦 Import Tests:")
    try:
        from memevole import MemorySystem, MemorySystemConfig
        print("  ✅ Core imports successful")
    except ImportError as e:
        print(f"  ❌ Core import failed: {e}")
        issues.append("❌ Core import failed")

    # Check storage backends
    storage_backends = []
    try:
        from components.store import JSONFileStore
        storage_backends.append("JSON")
    except ImportError:
        pass

    try:
        from components.store import VectorStore
        storage_backends.append("Vector")
    except ImportError:
        pass

    try:
        from components.store import GraphStorageBackend
        storage_backends.append("Graph")
    except ImportError:
        pass

    print(f"  📦 Available storage backends: {', '.join(storage_backends) if storage_backends else 'None'}")

    # Summary
    print(f"\n📊 Summary: {len(issues)} issues found")

    if issues:
        print("\n🔧 Issues to resolve:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ All basic checks passed!")

    return len(issues) == 0

if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)
```

Save as `diagnostics.py` and run with `python diagnostics.py`.

## 🔧 Common Issues and Solutions

### 1. "LLM client not initialized" Error

**Symptoms:**
```
RuntimeError: Encoder not initialized
RuntimeError: LLM client not initialized
```

**Causes & Solutions:**

**Missing Environment Variables:**
```bash
# Check .env file
cat .env | grep MEMEVOLVE_MEMORY_BASE_URL
cat .env | grep MEMEVOLVE_MEMORY_API_KEY

# Add to .env if missing
echo "MEMEVOLVE_MEMORY_BASE_URL=http://localhost:11433/v1" >> .env
echo "MEMEVOLVE_MEMORY_API_KEY=your-api-key" >> .env
```

**Incorrect API URL Format:**
```python
# ❌ Wrong
config.memory.base_url = "localhost:11433"

# ✅ Correct
config.memory.base_url = "http://localhost:11433/v1"
```

**LLM Service Not Running:**
```bash
# Check if your LLM service is accessible
curl http://localhost:11433/v1/models

# For vLLM, check status
curl http://localhost:11433/health
```

### 2. API Wrapper Issues

#### API Server Won't Start

**Symptoms:**
```
RuntimeError: Failed to initialize MemEvolve API server
```

**Solutions:**

**Check Dependencies:**
```bash
# Ensure API dependencies are installed
pip install fastapi uvicorn httpx

# Check Python path
python -c "import fastapi, uvicorn, httpx; print('Dependencies OK')"
```

**Configuration Issues:**
```bash
# Check .env file configuration
cat .env | grep MEMEVOLVE_UPSTREAM_BASE_URL
cat .env | grep MEMEVOLVE_API_MEMORY_INTEGRATION

# Check if upstream API is accessible
curl $(grep MEMEVOLVE_UPSTREAM_BASE_URL .env | cut -d= -f2)/health
```

#### Memory Not Being Used

**Symptoms:**
- API responses don't seem enhanced with memory
- Memory stats show 0 experiences

**Solutions:**

**Check Memory Integration:**
```bash
# Ensure memory integration is enabled
echo $MEMEVOLVE_API_MEMORY_INTEGRATION  # Should be "true"

# Check memory system status
curl http://localhost:11436/health
```

**Verify Memory Configuration:**
```bash
# Check LLM configuration for memory encoding
echo $MEMEVOLVE_MEMORY_BASE_URL
echo $MEMEVOLVE_MEMORY_API_KEY
echo "MEMEVOLVE_MEMORY_BASE_URL=http://localhost:11433/v1" >> .env
echo "MEMEVOLVE_MEMORY_API_KEY=your-api-key" >> .env
curl $MEMEVOLVE_MEMORY_BASE_URL/v1/models
```

**Check Storage Permissions:**
```bash
# Ensure data directory exists and is writable
mkdir -p data
chmod 755 data

# Check storage path
echo $MEMEVOLVE_STORAGE_PATH
```

#### 503 Service Unavailable

**Symptoms:**
```
HTTP 503 from MemEvolve API
```

**Causes & Solutions:**

**Upstream API Down:**
```bash
# Check if upstream LLM is responding
curl $MEMEVOLVE_UPSTREAM_BASE_URL/health

# Check network connectivity
ping $(echo $MEMEVOLVE_UPSTREAM_BASE_URL | sed 's|http://||' | cut -d: -f1)
```

**Memory System Failed:**
```bash
# Check memory system health
curl http://localhost:11436/health

# View detailed logs
echo "MEMEVOLVE_LOG_LEVEL=DEBUG" >> .env
python scripts/start_api.py  # Restart with debug logging
```

### 3. Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'faiss'
ModuleNotFoundError: No module named 'neo4j'
```

**Solutions:**

**Install Optional Dependencies:**
```bash
# For vector storage
pip install faiss-cpu

# For graph storage
pip install neo4j

# For development
pip install networkx
```

**Check Installation:**
```bash
# Verify package installation
python -c "import faiss; print('FAISS OK')"
python -c "import neo4j; print('Neo4j OK')"
```

### 3. Storage Backend Issues

**JSON Storage Not Persisting:**
```python
# Check file permissions
ls -la memories.json

# Ensure directory is writable
mkdir -p data/
chmod 755 data/
```

**Vector Storage IVF Index Training Error:**
```python
# Symptoms: "Error in virtual void faiss::IndexIVFFlat::add_core... 'is_trained' failed"
# This occurs when IVF indexes are used without proper training

# Solution: IVF indexes require training before adding vectors
# This is automatically handled in VectorStore._train_ivf_if_needed()
# If issues persist, try switching to flat index:
MEMEVOLVE_STORAGE_INDEX_TYPE=flat

# Or ensure sufficient data for training (IVF needs at least nlist vectors)
```

**Vector Storage Embedding API Response Error:**
```python
# Symptoms: "'list' object has no attribute 'data'" or embedding failures
# This occurs when embedding API returns unexpected response format

# Solution: The embedding provider automatically handles multiple formats:
# - Standard OpenAI: response.data[0].embedding
# - Direct list: response[0]
# - Direct embedding: response.embedding
# - Dict formats: response['data'][0]['embedding']
# - llama.cpp: Automatic format detection with hybrid client approach

# If issues persist, check your embedding API compatibility:
curl $MEMEVOLVE_EMBEDDING_BASE_URL/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MEMEVOLVE_EMBEDDING_API_KEY" \
  -d '{"input": "test", "model": "your-model"}'
```

### 7. Memory Scoring Issues

#### Memory Scores Display as "N/A"

**Symptoms:**
```
Retrieved memories for API request:
  #1: memory_0 (score: N/A) - 'Memory content here...'
```

**Causes & Solutions:**

**Outdated Memory System Code:**
This issue was resolved in the latest commit. Ensure you're running the updated version:

```bash
# Pull latest changes
git pull origin master

# Restart the API server
python scripts/start_api.py --reload
```

**Verification:**
After updating, memory scores should display properly:
```
Retrieved memories for API request:
  #1: unit_123 (score: 0.743) - 'Memory content here...'
  #2: unit_456 (score: 0.658) - 'Another memory...'
```

#### Quality Scoring Issues

**All Quality Scores are the Same:**
```python
# Symptoms: Quality scores appear constant or don't vary

# Cause: Missing query context or response content
# Solution: Ensure proper context structure
context = {
    "original_query": "your actual query",
    "messages": [{"role": "user", "content": "your actual query"}]
}
```

**Reasoning Models Score Lower Than Expected:**
```python
# Symptoms: Thinking models getting consistently lower scores

# Cause: Bias correction needs time to learn from your data
# Solution: Enable debug logging and monitor bias tracking
export MEMEVOLVE_LOG_MIDDLEWARE_ENABLE=true
grep "bias correction" logs/api-server.log

# Bias correction automatically adjusts as more data is collected
# Give the system 20-50 responses before evaluating bias effectiveness
```

**Quality Scores Seem Too High/Low:**
```python
# Symptoms: Scores don't match perceived quality

# Solution: Adjust minimum threshold for your use case
export MEMEVOLVE_QUALITY_MIN_THRESHOLD=0.2  # Raise threshold

# Or disable bias correction if not needed
export MEMEVOLVE_QUALITY_BIAS_CORRECTION=false
```

#### Embedding Compatibility Issues

**llama.cpp Embedding Failures:**
```python
# Symptoms: Embedding errors with llama.cpp services
# "Connection failed" or "Invalid response format"

# Solution: The system now automatically detects llama.cpp endpoints
# Ensure proper URL format:
export MEMEVOLVE_EMBEDDING_BASE_URL=http://localhost:11435

# Test embedding endpoint directly:
curl http://localhost:11435/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test text"}'

# The system handles both:
# - llama.cpp format: Direct requests
# - OpenAI format: OpenAI client
```

**Vector Storage Dimension Mismatch:**
```python
# Match embedding dimensions
config.storage_backend = VectorStore(dimension=768)  # For 768-dim embeddings
config.storage_backend = VectorStore(dimension=384)  # For 384-dim embeddings
```

**Graph Storage Connection Failed:**
```python
# Check Neo4j is running
curl http://localhost:7474/browser/

# Verify connection parameters
config.storage_backend = GraphStorageBackend(
    uri="bolt://localhost:7687",  # Default Neo4j bolt port
    user="neo4j",
    password="your-password"
)
```

### 4. Performance Issues

**Slow Encoding:**
```python
# Use batch processing
memory.add_trajectory_batch(experiences, use_parallel=True)

# Increase LLM timeout
config.memory.timeout = 60  # seconds
```

**Slow Retrieval:**
```python
# Reduce retrieval count
results = memory.query_memory("query", top_k=3)  # Instead of 10

# Use faster storage backend
config.storage_backend = JSONFileStore()  # For development
config.storage_backend = VectorStore()    # For production
```

**Memory Growing Indefinitely:**
```python
# Enable automatic management
config.enable_auto_management = True
config.auto_prune_threshold = 1000

# Manual cleanup
memory.manage_memory()
```

### 5. Encoding Failures

**Empty or Invalid Encodings:**
```python
# Check experience format
experience = {
    "action": "debug code",           # Required
    "result": "found the bug",        # Required
    "context": "Python application",  # Optional but helpful
    "timestamp": "2024-01-01T10:00:00Z"  # Optional
}

# Validate before adding
assert "action" in experience
assert "result" in experience
```

**LLM Response Parsing Errors:**
```python
# Check LLM is returning valid JSON
# The encoder expects responses in this format:
{
  "type": "lesson|skill|tool|abstraction",
  "content": "The transformed content",
  "metadata": {},
  "tags": ["tag1", "tag2"]
}
```

### 6. Retrieval Issues

**No Results Returned:**
```python
# Check query is meaningful
results = memory.query_memory("debugging techniques", top_k=5)

# Try different retrieval strategies
from components.retrieve import KeywordRetrievalStrategy
config.retrieval_strategy = KeywordRetrievalStrategy()
```

**Poor Result Quality:**
```python
# Use hybrid retrieval for better results
from components.retrieve import HybridRetrievalStrategy
config.retrieval_strategy = HybridRetrievalStrategy()

# Increase retrieval count and filter
results = memory.query_memory("query", top_k=10, filters={"types": ["lesson"]})
```

**Memory Retrieval Scores Not Displaying:**
```python
# Symptoms: Score values show as "N/A" in logs

# This issue has been resolved in the latest version
# Update to the latest code:
git pull origin master
python scripts/start_api.py --reload

# Verify scores are now displayed:
grep "Retrieved memories" logs/api-server.log
# Should show: "unit_123 (score: 0.743)" instead of "(score: N/A)"
```

## ❓ Frequently Asked Questions

### General Questions

**Q: What's the difference between the storage backends?**
- **JSON**: Simple file-based, good for development and debugging
- **Vector**: High-performance semantic search using embeddings
- **Graph**: Relationship-aware storage with graph traversal capabilities

**Q: How do I choose the right retrieval strategy?**
- **Keyword**: For exact matches and structured queries
- **Semantic**: For meaning-based search and general questions
- **Hybrid**: Best balance of precision and recall (recommended)
- **LLM-Guided**: For complex reasoning and context-aware search

**Q: Can I use MemEvolve without an LLM?**
A: No, MemEvolve requires an LLM for experience encoding. The LLM transforms raw experiences into structured memories.

### Technical Questions

**Q: How much memory does MemEvolve use?**
A: Depends on storage backend:
- JSON: Minimal (file size)
- Vector: ~dimension × num_memories × 4 bytes (embeddings)
- Graph: Variable, depends on relationships

**Q: Can I run MemEvolve in production?**
A: Yes, with proper configuration:
- Use VectorStore or GraphStorageBackend
- Enable auto-management
- Set appropriate timeouts and retries
- Monitor performance and memory usage

**Q: How do I backup my memories?**
```python
# Export to JSON
memories = memory.export_memories()
with open('backup.json', 'w') as f:
    json.dump(memories, f)

# Import from backup
with open('backup.json', 'r') as f:
    memories = json.load(f)
memory.import_memories(memories)
```

### API Wrapper Questions

**Q: Does the API wrapper work with my LLM?**
A: Yes, if your LLM supports OpenAI-compatible APIs (OpenAI, vLLM, llama.cpp, Anthropic via proxy, etc.). The wrapper is protocol-compatible with any service that implements the OpenAI chat completions API.

**Q: How much latency does the memory integration add?**
A: Typically 50-200ms depending on:
- Memory retrieval complexity (top_k setting)
- Storage backend (JSON is fastest, vector search adds ~50ms)
- LLM API response time
- Memory size and indexing

**Q: Can I use different LLMs for memory encoding vs. chat?**
A: Yes! Set `MEMEVOLVE_MEMORY_BASE_URL` to a different LLM service than `MEMEVOLVE_UPSTREAM_BASE_URL`. This allows using a smaller, faster model for encoding while keeping a larger model for chat responses.

**Q: How do I scale the API wrapper?**
A: Multiple approaches:
- Run multiple instances behind a load balancer
- Use shared storage backend (Redis, PostgreSQL) for memory
- Configure auto-pruning to manage memory size
- Use connection pooling for upstream API calls

### Integration Questions

**Q: How do I integrate with LangChain?**
```python
from langchain.memory import ConversationBufferMemory
from memevole import MemorySystem

class MemEvolveLangChainMemory(ConversationBufferMemory):
    def __init__(self, memory_system, **kwargs):
        super().__init__(**kwargs)
        self.memevole = memory_system

    def save_context(self, inputs, outputs):
        experience = {
            "action": "conversation",
            "result": outputs.get("output", ""),
            "context": inputs.get("input", "")
        }
        self.memevole.add_experience(experience)
```

**Q: Can I use custom embeddings?**
```python
def custom_embedding_function(text: str) -> List[float]:
    # Your embedding logic here
    return embeddings

config.embedding_function = custom_embedding_function
config.storage_backend = VectorStore(dimension=your_embedding_dim)
```

### Performance Questions

**Q: How fast is MemEvolve?**
A: Performance varies by configuration:
- Encoding: 2-10 seconds per experience (LLM-dependent)
- Retrieval: <100ms for JSON, <10ms for Vector (with FAISS)
- Quality Scoring: 5-25ms additional overhead (direct vs reasoning responses)
- Batch encoding: 2-5x faster than sequential

**Q: How many memories can MemEvolve handle?**
A: Scales with storage backend:
- JSON: Limited by file system (good for <100K)
- Vector: Millions of memories (limited by RAM/FAISS)
- Graph: Thousands to millions (depends on Neo4j setup)

**Q: What's the overhead of quality scoring?**
A: Minimal overhead with intelligent optimization:
- Direct responses: ~5-10ms additional processing
- Reasoning responses: ~15-25ms (includes consistency analysis)
- Bias correction: <1ms with efficient tracking
- Total impact: Negligible for typical API workloads (<2% latency increase)

### Quality Scoring Questions

**Q: Why do thinking models get different score treatment?**
A: To ensure fair competition between model types:
- Direct models: Evaluated 100% on answer quality
- Thinking models: 70% answer + 30% reasoning quality
- Bias correction: Automatically adjusts for systematic differences
- Result: Fair evaluation regardless of reasoning capabilities

**Q: Can I customize quality scoring criteria?**
A: Yes, through multiple approaches:
```python
# Adjust weighting factors
scorer = ResponseQualityScorer(
    reasoning_weight=0.3,  # Default 30%
    answer_weight=0.7      # Default 70%
)

# Disable bias correction
export MEMEVOLVE_QUALITY_BIAS_CORRECTION=false

# Set minimum thresholds
export MEMEVOLVE_QUALITY_MIN_THRESHOLD=0.2
```

**Q: How do I monitor quality trends?**
A: Built-in monitoring and logging:
```bash
# Enable detailed quality logs
export MEMEVOLVE_LOG_MIDDLEWARE_ENABLE=true

# Monitor in logs
tail -f logs/api-server.log | grep "quality scoring"

# View quality trends via dashboard
open http://localhost:11436/dashboard
```

## 🆘 Getting Help

If you're still having issues:

1. **Check the logs**: Set `MEMEVOLVE_LOG_LEVEL=DEBUG` for detailed output
2. **Run diagnostics**: Use the diagnostic script above
3. **Check GitHub Issues**: Search for similar problems
4. **Create an issue**: Include your diagnostic output and error messages

### Debug Information to Include

When reporting issues, please include:

```python
import sys
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")

# MemEvolve version
try:
    import memevole
    print("MemEvolve import: OK")
except ImportError as e:
    print(f"MemEvolve import: FAILED - {e}")

# Environment variables (without sensitive data)
import os
env_vars = ['MEMEVOLVE_MEMORY_BASE_URL', 'MEMEVOLVE_MEMORY_MODEL', 'MEMEVOLVE_LOG_LEVEL']
for var in env_vars:
    print(f"{var}: {'SET' if os.getenv(var) else 'NOT SET'}")
```

This information helps diagnose issues quickly and provide targeted solutions.

## 📋 Known Limitations

This section documents current limitations and workarounds for MemEvolve. These are not bugs but areas where the system has constraints or known performance characteristics.

### Memory System Limitations

#### Large Memory Performance Degradation

**Issue**: Performance degrades significantly with large memory databases (>10,000 units)

**Symptoms**:
- Increased latency on memory retrieval (>500ms)
- Higher CPU usage during search operations
- Potential memory leaks in long-running processes

**Workaround**:
- Enable auto-pruning: `MEMEVOLVE_MANAGEMENT_ENABLE_AUTO_MANAGEMENT=true`
- Set reasonable limits: `MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD=5000`
- Use vector storage backend for better performance with large datasets

**Status**: Known limitation. Performance optimization needed for enterprise-scale deployments.

#### JSON Storage Concurrency Issues

**Issue**: JSON file storage can have concurrency issues with multiple simultaneous requests

**Symptoms**:
- Race conditions when multiple processes access memory.json simultaneously
- Potential data corruption in high-throughput scenarios
- File locking errors in multi-threaded environments

**Workaround**:
- Use single-threaded deployment for JSON storage
- Switch to vector storage backend for concurrent access: `MEMEVOLVE_STORAGE_BACKEND_TYPE=vector`
- Implement file locking mechanisms if JSON storage must be used

**Status**: Limitation of JSON storage backend. Consider database backends for production use.

### API Compatibility Issues

#### Provider-Specific Streaming Support

**Issue**: Some LLM providers may have limited streaming compatibility

**Symptoms**:
- Provider-specific streaming format variations
- Potential compatibility issues with certain models

**Workaround**: Test streaming with your specific provider; fall back to non-streaming if issues occur.

**Status**: Streaming support is implemented but may vary by provider.

### Monitoring and Observability Gaps

#### Limited Error Reporting

**Issue**: Error messages and logging could be more informative for troubleshooting

**Symptoms**:
- Generic error messages that don't pinpoint root causes
- Insufficient debug information for complex issues
- Difficulty diagnosing configuration problems

**Workaround**:
- Enable debug logging: `MEMEVOLVE_LOG_LEVEL=DEBUG`
- Check individual component logs: `MEMEVOLVE_LOG_API_SERVER_ENABLE=true`

**Status**: Error reporting improvements planned for Phase 2.

### Security Considerations

#### API Key Exposure Risk

**Issue**: API keys may be exposed in logs when debug logging is enabled

**Symptoms**:
- Sensitive API keys appear in log files
- Potential security risk in shared environments

**Workaround**:
- Disable detailed logging in production: `MEMEVOLVE_LOG_LEVEL=WARNING`
- Use environment variables instead of config files for sensitive data

**Status**: Security hardening planned for Phase 2 (IMPORTANT priority).

### Evolution System Resilience

#### Force-Kill Protection

**Issue**: Server force-kills previously caused evolution state corruption

**Symptoms** (resolved):
- Evolution state reset to empty after unexpected shutdowns
- JSON parsing errors on restart: "Expecting value: line X column Y"

**Solution**: Implemented atomic writes and automatic backup recovery:
- Evolution state survives force-kills without corruption
- Automatic recovery from `data/evolution_backups/` on startup
- Corrupted files moved to `.corrupted` extension for debugging

**Status**: ✅ Resolved - Evolution persistence is now robust against unexpected shutdowns.

---

*Last updated: January 24, 2026*