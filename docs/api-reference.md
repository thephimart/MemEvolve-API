# MemEvolve API Reference

This reference covers all MemEvolve API endpoints, configuration options, and usage patterns for integrating memory capabilities into your applications.

## 🌐 API Architecture

MemEvolve provides two primary interfaces:

1. **Proxy API**: Transparent OpenAI-compatible API that adds memory automatically
2. **Management API**: Direct endpoints for memory inspection and control

## 🚀 Proxy API (OpenAI-Compatible)

The proxy API is a drop-in replacement for any OpenAI-compatible LLM service. All standard OpenAI endpoints are supported with automatic memory integration.

### Base URL
```
http://localhost:8001/v1
```

### Supported Endpoints

#### Chat Completions
```http
POST /v1/chat/completions
```

**Request Body**: Standard OpenAI chat completion format
```json
{
  "model": "your-model",
  "messages": [
    {"role": "user", "content": "How do I implement caching?"}
  ],
  "max_tokens": 1000
}
```

**MemEvolve Enhancement**: Automatically injects relevant memories into the conversation context.

#### Embeddings
```http
POST /v1/embeddings
```

**Request Body**: Standard OpenAI embedding format
```json
{
  "model": "your-embedding-model",
  "input": "Text to embed"
}
```

**Note**: Routes to `MEMEVOLVE_EMBEDDING_BASE_URL` if configured, otherwise uses `MEMEVOLVE_UPSTREAM_BASE_URL`.

## 🔧 Management API

Direct endpoints for memory system management and inspection.

### Base URL
```
http://localhost:8001
```

### Health & Status

#### Get Health Status
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-20T10:30:00Z",
  "version": "1.0.0"
}
```

#### Get System Metrics
```http
GET /metrics
```

**Response**:
```json
{
  "memory_units": 150,
  "total_requests": 1250,
  "avg_response_time": 0.234,
  "uptime_seconds": 3600
}
```

### Memory Operations

#### Get Memory Statistics
```http
GET /memory/stats
```

**Response**:
```json
{
  "total_units": 150,
  "units_by_type": {
    "lesson": 45,
    "skill": 32,
    "conversation": 73
  },
  "storage_backend": "json",
  "last_updated": "2024-01-20T10:30:00Z"
}
```

#### Search Memory
```http
POST /memory/search
```

**Request Body**:
```json
{
  "query": "database optimization",
  "top_k": 5,
  "filters": {
    "type": "lesson",
    "tags": ["performance"]
  }
}
```

**Response**:
```json
{
  "results": [
    {
      "id": "unit_123",
      "content": "Database indexing improves query performance...",
      "type": "lesson",
      "score": 0.89,
      "metadata": {
        "created_at": "2024-01-20T09:15:00Z",
        "tags": ["database", "performance"]
      }
    }
  ],
  "total_found": 1,
  "search_time": 0.023
}
```

#### Add Memory Unit
```http
POST /memory/add
```

**Request Body**:
```json
{
  "content": "JWT tokens should be validated on every request for security.",
  "type": "lesson",
  "tags": ["security", "authentication"],
  "metadata": {
    "source": "security_audit",
    "importance": "high"
  }
}
```

**Response**:
```json
{
  "unit_id": "unit_456",
  "status": "stored",
  "timestamp": "2024-01-20T10:30:00Z"
}
```

#### Get Memory Unit
```http
GET /memory/{unit_id}
```

**Response**:
```json
{
  "id": "unit_456",
  "content": "JWT tokens should be validated on every request for security.",
  "type": "lesson",
  "tags": ["security", "authentication"],
  "metadata": {
    "created_at": "2024-01-20T10:30:00Z",
    "source": "security_audit",
    "importance": "high"
  }
}
```

#### Update Memory Unit
```http
PUT /memory/{unit_id}
```

**Request Body**:
```json
{
  "content": "Updated content with additional security details...",
  "tags": ["security", "authentication", "jwt"],
  "metadata": {
    "last_reviewed": "2024-01-20T10:30:00Z"
  }
}
```

#### Delete Memory Unit
```http
DELETE /memory/{unit_id}
```

**Response**:
```json
{
  "status": "deleted",
  "unit_id": "unit_456"
}
```

#### Clear All Memory
```http
POST /memory/clear
```

**Request Body** (optional):
```json
{
  "confirm": true,
  "filters": {
    "type": "conversation",
    "older_than_days": 30
  }
}
```

### Maintenance Operations

#### Trigger Memory Optimization
```http
POST /maintenance/optimize
```

**Request Body** (optional):
```json
{
  "operations": ["deduplicate", "prune", "consolidate"],
  "dry_run": false
}
```

**Response**:
```json
{
  "status": "completed",
  "operations_performed": [
    "deduplicated 5 units",
    "pruned 12 old units",
    "consolidated 3 similar units"
  ],
  "units_before": 150,
  "units_after": 136
}
```

#### Export Memory
```http
POST /memory/export
```

**Request Body**:
```json
{
  "format": "json",
  "filters": {
    "type": "lesson",
    "tags": ["important"]
  },
  "include_metadata": true
}
```

**Response**: File download with exported memory data.

#### Import Memory
```http
POST /memory/import
```

**Request Body**:
```json
{
  "data": [...], // Array of memory units
  "merge_strategy": "skip_existing" // or "overwrite", "merge_metadata"
}
```

## ⚙️ Configuration Reference

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MEMEVOLVE_API_HOST` | Server bind address | `127.0.0.1` | No |
| `MEMEVOLVE_API_PORT` | Server port | `8001` | No |
| `MEMEVOLVE_UPSTREAM_BASE_URL` | Upstream LLM API URL | `http://localhost:8000/v1` | Yes |
| `MEMEVOLVE_UPSTREAM_API_KEY` | Upstream API key | None | No |
| `MEMEVOLVE_API_MEMORY_INTEGRATION` | Enable memory features | `true` | No |
| `MEMEVOLVE_LLM_BASE_URL` | Memory system LLM | Falls back to upstream | No |
| `MEMEVOLVE_LLM_API_KEY` | Memory system API key | Falls back to upstream | No |
| `MEMEVOLVE_EMBEDDING_BASE_URL` | Embedding API URL | Falls back to upstream | No |
| `MEMEVOLVE_EMBEDDING_API_KEY` | Embedding API key | Falls back to upstream | No |

### Memory Configuration

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `MEMEVOLVE_STORAGE_PATH` | Memory storage location | `./data/memory.json` | |
| `MEMEVOLVE_STORAGE_BACKEND_TYPE` | Storage type | `json` | `json`, `vector`, `graph` |
| `MEMEVOLVE_DEFAULT_TOP_K` | Default retrieval count | `5` | |
| `MEMEVOLVE_RETRIEVAL_STRATEGY_TYPE` | Retrieval method | `hybrid` | `semantic`, `keyword`, `hybrid` |
| `MEMEVOLVE_RETRIEVAL_SEMANTIC_WEIGHT` | Semantic vs keyword balance | `0.7` | |
| `MEMEVOLVE_MANAGEMENT_ENABLE_AUTO_MANAGEMENT` | Auto memory management | `true` | |
| `MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD` | Memory size limit | `1000` | |

## 🔒 Authentication

- **Upstream API**: Uses `MEMEVOLVE_UPSTREAM_API_KEY`
- **Management API**: No authentication required (configure firewall/host binding for security)
- **Proxy Requests**: Pass-through authentication to upstream API

## 📊 Response Codes

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (unit doesn't exist)
- `500`: Internal Server Error
- `503`: Service Unavailable (upstream API down)

## 🚀 Usage Examples

### Python Client
```python
import requests

base_url = "http://localhost:8001"

# Check health
health = requests.get(f"{base_url}/health").json()

# Search memory
results = requests.post(f"{base_url}/memory/search",
    json={"query": "machine learning", "top_k": 3}
).json()

# Add memory
requests.post(f"{base_url}/memory/add",
    json={
        "content": "Gradient descent minimizes loss functions in ML.",
        "type": "lesson",
        "tags": ["ml", "optimization"]
    }
)
```

### cURL Examples
```bash
# Health check
curl http://localhost:8001/health

# Memory stats
curl http://localhost:8001/memory/stats

# Search memory
curl -X POST http://localhost:8001/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "database optimization", "top_k": 3}'

# Add memory unit
curl -X POST http://localhost:8001/memory/add \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Use EXPLAIN ANALYZE to understand query execution plans.",
    "type": "lesson",
    "tags": ["database", "postgresql"]
  }'
```

## 🔧 Troubleshooting

- **Connection Refused**: Check if MemEvolve server is running on correct host/port
- **Empty Results**: Verify memory has been populated and search query is relevant
- **Slow Responses**: Check upstream API performance and network connectivity
- **Authentication Errors**: Verify API keys are correctly configured

For detailed troubleshooting, see the [troubleshooting guide](troubleshooting.md).