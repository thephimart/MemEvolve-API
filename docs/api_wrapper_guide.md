# MemEvolve API Wrapper Guide

This guide covers the MemEvolve API wrapper - a memory-enhanced proxy for OpenAI-compatible LLM APIs that provides seamless memory integration without changing your existing applications.

## 🌟 Overview

The MemEvolve API wrapper acts as a transparent proxy that:

- **Intercepts** requests to any OpenAI-compatible LLM API
- **Injects** relevant context from memory into prompts
- **Learns** from interactions by encoding new experiences
- **Maintains** memory health through automatic management
- **Provides** additional endpoints for memory inspection and control

## 🚀 Quick Start

### 1. Start the API Server

```bash
# Using the startup script
python scripts/start_api.py

# Or with Docker
docker-compose up -d

# Or manually
uvicorn src.api.server:app --host 0.0.0.0 --port 8001
```

### 2. Configure Your Upstream API

Configure your `.env` file with your AI service endpoints:

```bash
# Chat completions API (required)
MEMEVOLVE_UPSTREAM_BASE_URL=http://localhost:8000/v1
MEMEVOLVE_UPSTREAM_API_KEY=your-llm-api-key

# Embedding API (optional - defaults to same as upstream)
# Only needed if your embedding service is on a different endpoint
# MEMEVOLVE_EMBEDDING_BASE_URL=http://different-endpoint:8001/v1
# MEMEVOLVE_EMBEDDING_API_KEY=your-embedding-key

# Enable memory integration
MEMEVOLVE_API_MEMORY_INTEGRATION=true
```

**API Requirements:**
- **LLM Endpoint**: For chat completions and memory encoding
- **Embedding Endpoint**: For vectorizing memories (defaults to same as LLM endpoint)

**Simple Defaults:** MemEvolve uses your LLM endpoint for both chat and embeddings by default. Only specify separate embedding settings if your service requires it.

### Production Storage Configuration

For production deployments with >1,000 memories, switch from the default JSON storage to vector storage for much better performance:

```bash
# Switch to vector storage for better performance
MEMEVOLVE_STORAGE_BACKEND_TYPE=vector
MEMEVOLVE_STORAGE_VECTOR_DIM=768
```

This provides ~100x faster memory retrieval. See the [Configuration Guide](configuration_guide.md) for detailed storage backend comparisons.

### 3. Use Like Any OpenAI API

Your existing applications can now use the MemEvolve proxy:

```python
import openai

# Point to MemEvolve proxy instead of direct LLM API
client = openai.OpenAI(
    base_url="http://localhost:8001/v1",  # MemEvolve proxy
    api_key="dummy-key"  # Not used, but required by client
)

# Your existing code works unchanged!
response = client.chat.completions.create(
    model="your-model",
    messages=[
        {"role": "user", "content": "How do I debug a Python application?"}
    ]
)

# MemEvolve automatically:
# 1. Retrieves relevant debugging memories
# 2. Adds them to your prompt
# 3. Calls your upstream LLM
# 4. Learns from this interaction
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MEMEVOLVE_API_HOST` | Server bind address | `127.0.0.1` | No |
| `MEMEVOLVE_API_PORT` | Server port | `8001` | No |
| `MEMEVOLVE_UPSTREAM_BASE_URL` | Upstream LLM API URL | `http://localhost:8000/v1` | Yes |
| `MEMEVOLVE_UPSTREAM_API_KEY` | Upstream API key | None | No |
| `MEMEVOLVE_API_MEMORY_INTEGRATION` | Enable memory features | `true` | No |
| `MEMEVOLVE_LLM_BASE_URL` | Memory system LLM (for encoding) | Same as upstream | No |
| `MEMEVOLVE_LLM_API_KEY` | Memory system API key | Same as upstream | No |

### Advanced Configuration

```python
# Custom memory configuration
export MEMEVOLVE_STORAGE_PATH="./data/memory.json"      # Memory storage location
export MEMEVOLVE_DEFAULT_TOP_K="5"                      # Retrieval count
export MEMEVOLVE_MANAGEMENT_ENABLE_AUTO_MANAGEMENT="true"                     # Auto memory management
export MEMEVOLVE_AUTO_PRUNE_THRESHOLD="1000"            # Memory size limit
```

### Evolution Configuration (Advanced)

MemEvolve supports optional runtime evolution to automatically optimize memory architectures based on API usage patterns. Configure these settings in your `.env` file:

```bash
# Enable evolution (default: false)
MEMEVOLVE_ENABLE_EVOLUTION=true

# Evolution parameters (only used when evolution is enabled)
MEMEVOLVE_EVOLUTION_POPULATION_SIZE=10        # Number of architectures to test
MEMEVOLVE_EVOLUTION_GENERATIONS=20            # Evolution cycles to run
MEMEVOLVE_EVOLUTION_MUTATION_RATE=0.1         # Parameter mutation probability
MEMEVOLVE_EVOLUTION_CROSSOVER_RATE=0.5        # Architecture combination probability
MEMEVOLVE_EVOLUTION_SELECTION_METHOD=pareto   # Selection strategy
MEMEVOLVE_EVOLUTION_TOURNAMENT_SIZE=3         # Tournament selection size
```

**Note**: Evolution is an advanced feature that runs genetic optimization in the background. It may increase CPU usage but can discover better memory configurations over time. All settings are configured in your `.env` file.

## 📡 API Endpoints

### Proxy Endpoints

The API wrapper proxies all OpenAI-compatible endpoints transparently:

- `POST /v1/chat/completions` - Chat completions with memory
- `POST /v1/completions` - Text completions with memory
- `GET /v1/models` - Model information
- All other OpenAI endpoints pass through unchanged

### Memory Management Endpoints

Additional endpoints for memory inspection and control:

#### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "memory_enabled": true,
  "memory_integration_enabled": true,
  "upstream_url": "http://localhost:8000/v1"
}
```

### Evolution Endpoints (when evolution is enabled)

Control and monitor the evolution process:

#### Start Evolution
```http
POST /evolution/start
```

#### Stop Evolution
```http
POST /evolution/stop
```

#### Get Evolution Status
```http
GET /evolution/status
```

Response:
```json
{
  "is_running": true,
  "current_genotype": "abc123def",
  "population_size": 10,
  "evolution_cycles_completed": 5,
  "last_evolution_time": 1640995200.0,
  "api_requests_total": 1000,
  "average_response_time": 0.25,
  "memory_retrievals_total": 500,
  "average_retrieval_time": 0.05
}
```

#### Memory Statistics
```http
GET /memory/stats
```

Response:
```json
{
  "total_experiences": 150,
  "retrieval_count": 42,
  "last_updated": "2024-01-19T21:30:00Z",
  "architecture": "AgentKB"
}
```

#### Search Memory
```http
POST /memory/search
Content-Type: application/json

{
  "query": "debugging techniques",
  "limit": 10,
  "include_metadata": false
}
```

Response:
```json
[
  {
    "content": "Use print statements to trace variable values...",
    "score": 0.89,
    "type": "lesson"
  },
  {
    "content": "Check logs for error patterns...",
    "score": 0.76,
    "type": "skill"
  }
]
```

#### Clear Memory
```http
POST /memory/clear
```

Response:
```json
{
  "message": "Memory operation log cleared successfully"
}
```

#### Memory Configuration
```http
GET /memory/config
```

Returns the current memory system configuration.

## 🧠 How Memory Integration Works

### Request Processing

1. **Context Extraction**: Analyzes incoming messages for query intent
2. **Memory Retrieval**: Searches for relevant past experiences
3. **Context Injection**: Adds retrieved memories to system prompt
4. **API Forwarding**: Sends enhanced request to upstream LLM

### Example Enhancement

**Original Request:**
```json
{
  "messages": [
    {"role": "user", "content": "How do I fix a memory leak in Python?"}
  ]
}
```

**Enhanced Request (with memory):**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "Relevant past experiences:\n1. Memory leak debugging: Use tracemalloc to track allocations (relevance: 0.92)\n2. Python GC monitoring: Check reference cycles with gc module (relevance: 0.85)"
    },
    {"role": "user", "content": "How do I fix a memory leak in Python?"}
  ]
}
```

### Experience Encoding

After each response, MemEvolve learns by:

1. **Interaction Analysis**: Categorizes the conversation (lesson, skill, tool, abstraction)
2. **Content Extraction**: Identifies key learnings and patterns
3. **Metadata Generation**: Adds timestamps, tags, and context
4. **Memory Storage**: Persists the experience for future retrieval

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```yaml
# docker-compose.yml
version: '3.8'
services:
  memevolve-api:
    image: memevolve-api:latest
    ports:
      - "8001:8001"
    environment:
      - MEMEVOLVE_API_HOST=0.0.0.0
      - MEMEVOLVE_UPSTREAM_BASE_URL=http://llm-service:8000/v1
      - MEMEVOLVE_API_MEMORY_INTEGRATION=true
    volumes:
      - ./data:/app/data
    depends_on:
      - llm-service

  llm-service:
    # Your LLM service (vLLM, llama.cpp, etc.)
    image: your-llm-image
    ports:
      - "8000:8000"
```

```bash
# Deploy
docker-compose up -d

# View logs
docker-compose logs -f memevolve-api

# Scale
docker-compose up -d --scale memevolve-api=3
```

### Using Docker Run

```bash
# Build image
docker build -t memevolve-api .

# Run container
docker run -d \
  --name memevolve-api \
  -p 8001:8001 \
  -e MEMEVOLVE_API_HOST=0.0.0.0 \
  -e MEMEVOLVE_UPSTREAM_BASE_URL=http://host.docker.internal:8000/v1 \
  -v $(pwd)/data:/app/data \
  memevolve-api
```

## 🔍 Monitoring and Debugging

### Logs

Enable detailed logging:

```bash
export MEMEVOLVE_LOG_LEVEL=DEBUG
export MEMEVOLVE_LOGGING_ENABLE_OPERATION_LOG=true
```

### Memory Inspection

```python
import requests

# Check memory health
health = requests.get("http://localhost:8001/health").json()
print(f"Memory enabled: {health['memory_enabled']}")

# Inspect recent memories
stats = requests.get("http://localhost:8001/memory/stats").json()
print(f"Total memories: {stats['total_experiences']}")

# Search for specific knowledge
results = requests.post("http://localhost:8001/memory/search",
    json={"query": "error handling", "limit": 5}).json()
for memory in results:
    print(f"- {memory['content'][:50]}... (score: {memory['score']:.2f})")
```

### Performance Monitoring

The API includes built-in performance tracking:

```python
# Check response times and memory usage
import time
start = time.time()
response = requests.post("http://localhost:8001/v1/chat/completions", ...)
end = time.time()
print(f"Response time: {end - start:.2f}s")
```

## 🧪 Testing the API

### Basic Functionality Test

```python
import requests

# Test health
assert requests.get("http://localhost:8001/health").json()["status"] == "healthy"

# Test memory integration
response = requests.post("http://localhost:8001/v1/chat/completions",
    json={
        "model": "test",
        "messages": [{"role": "user", "content": "Hello"}]
    })
assert response.status_code == 200

# Check that memory was updated
stats = requests.get("http://localhost:8001/memory/stats").json()
assert stats["total_experiences"] > 0
```

### Integration Test with Real LLM

```python
import openai

# Configure client to use MemEvolve proxy
client = openai.OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy"
)

# Make a request that should benefit from memory
response = client.chat.completions.create(
    model="your-model",
    messages=[
        {"role": "user", "content": "Explain memory management in Python"}
    ]
)

# Verify response quality and check memory growth
stats_before = requests.get("http://localhost:8001/memory/stats").json()
# ... make more requests ...
stats_after = requests.get("http://localhost:8001/memory/stats").json()
assert stats_after["total_experiences"] > stats_before["total_experiences"]
```

## 🚨 Troubleshooting

### Common Issues

#### API Returns 503 Service Unavailable
```bash
# Check if server is running
curl http://localhost:8001/health

# Check logs
docker-compose logs memevolve-api
```

#### Memory Not Working
```bash
# Check configuration
echo $MEMEVOLVE_API_MEMORY_INTEGRATION

# Verify upstream API is accessible
curl $MEMEVOLVE_UPSTREAM_BASE_URL/health
```

#### Slow Responses
```bash
# Check memory retrieval settings
echo $MEMEVOLVE_DEFAULT_TOP_K  # Reduce if too high

# Monitor memory size
curl http://localhost:8001/memory/stats
```

#### Out of Memory
```bash
# Enable auto management
export MEMEVOLVE_MANAGEMENT_ENABLE_AUTO_MANAGEMENT=true
export MEMEVOLVE_AUTO_PRUNE_THRESHOLD=1000

# Manual cleanup
curl -X POST http://localhost:8001/memory/clear
```

### Debug Mode

Enable maximum debugging:

```bash
export MEMEVOLVE_LOG_LEVEL=DEBUG
export MEMEVOLVE_API_MEMORY_INTEGRATION=true
export MEMEVOLVE_LOGGING_ENABLE_OPERATION_LOG=true
export MEMEVOLVE_LOGGING_MAX_LOG_SIZE_MB=100
```

Check logs for detailed information about memory retrieval and encoding.

## 🔄 Migration Guide

### From Direct LLM API

**Before:**
```python
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-key"
)
```

**After:**
```python
client = openai.OpenAI(
    base_url="http://localhost:8001/v1",  # MemEvolve proxy
    api_key="dummy"  # Not used
)
```

### From LangChain

**Before:**
```python
from langchain.llms import OpenAI
llm = OpenAI(api_key="your-key", base_url="http://localhost:8000/v1")
```

**After:**
```python
from langchain.llms import OpenAI
llm = OpenAI(api_key="dummy", base_url="http://localhost:8001/v1")
```

## 📊 Performance Considerations

### Throughput
- Memory retrieval adds 50-200ms latency
- Batch requests for better performance
- Use connection pooling for high throughput

### Memory Usage
- Monitor memory growth with `/memory/stats`
- Configure auto-pruning thresholds
- Use appropriate storage backends for scale

### Scaling
- Horizontal scaling with load balancer
- Shared memory backend for multiple instances
- Consider Redis for distributed memory

## 🎯 Best Practices

1. **Start Simple**: Begin with default settings and basic memory integration
2. **Monitor Performance**: Use health checks and stats endpoints regularly
3. **Configure Pruning**: Set appropriate memory limits to prevent unbounded growth
4. **Test Thoroughly**: Validate that memory enhancement improves your use case
5. **Plan for Scale**: Choose storage backends that match your expected memory size

## 📚 Related Documentation

- [Quick Start Tutorial](../tutorials/quick_start.md) - Basic MemEvolve usage
- [Configuration Guide](../configuration_guide.md) - Detailed configuration options
- [Developer Onboarding](../developer_onboarding.md) - Development setup and workflow
- [Troubleshooting Guide](../troubleshooting.md) - Common issues and solutions

---

The MemEvolve API wrapper makes memory-augmented AI applications as simple as changing a URL. Your applications gain memory capabilities without any code changes!