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
http://localhost:11436/v1
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
http://localhost:11436
```

### Health & Status

#### Get Health Status
```http
GET /health
```

**Response**:
```json
{
  "content": "Database indexing improves query performance...",
  "score": 0.89,
  "retrieval_metadata": {
    "semantic_score": 0.89,
    "keyword_score": 0.0,
    "semantic_rank": 0,
    "keyword_rank": null
  }
}
```

#### Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "memory_enabled": true,
  "memory_integration_enabled": true,
  "evolution_enabled": true,
  "upstream_url": "http://localhost:11434/v1"
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
  "limit": 5,
  "include_metadata": false
}
```

**Response**:
```json
[
  {
    "content": "Database indexing improves query performance...",
    "score": 0.89,
    "metadata": null
  }
]
```

#### Clear All Memory
```http
POST /memory/clear
```

**Request Body** (optional):
```json
{
  "confirm": true
}
```

**Response**:
```json
{
  "message": "Memory operation log cleared successfully"
}
```

#### Get Memory Configuration
```http
GET /memory/config
```

**Response**:
Returns the current memory system configuration as JSON.

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



### Quality Scoring Management

#### Get Quality Metrics
```http
GET /quality/metrics
```

**Response**:
```json
{
  "total_responses": 150,
  "average_quality_score": 0.342,
  "reasoning_model_count": 45,
  "direct_model_count": 105,
  "bias_adjustment_active": true,
  "score_distribution": {
    "excellent": 12,
    "good": 38,
    "average": 67,
    "poor": 28,
    "very_poor": 5
  }
}
```

#### Configure Quality Scoring
```http
POST /quality/configure
```

**Request Body**:
```json
{
  "bias_correction": true,
  "min_threshold": 0.15,
  "reasoning_weight": 0.3,
  "answer_weight": 0.7,
  "debug_logging": false
}
```

**Response**:
```json
{
  "message": "Quality scoring configuration updated"
}
```

### Evolution Management

#### Start Evolution
```http
POST /evolution/start
```

**Response**:
```json
{
  "message": "Evolution started successfully"
}
```

#### Stop Evolution
```http
POST /evolution/stop
```

**Response**:
```json
{
  "message": "Evolution stopped successfully"
}
```

#### Get Evolution Status
```http
GET /evolution/status
```

**Response**:
Returns current evolution status including metrics and generation information.

#### Record API Request
```http
POST /evolution/record-request
```

**Query Parameters**:
- `time_seconds`: Request duration in seconds
- `success`: Whether request was successful (default: true)

**Response**:
```json
{
  "message": "Request recorded"
}
```

#### Record Memory Retrieval
```http
POST /evolution/record-retrieval
```

**Query Parameters**:
- `time_seconds`: Retrieval duration in seconds
- `success`: Whether retrieval was successful (default: true)

**Response**:
```json
{
  "message": "Retrieval recorded"
}
```

## ⚙️ Configuration Reference

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MEMEVOLVE_API_HOST` | Server bind address | `127.0.0.1` | No |
| `MEMEVOLVE_API_PORT` | Server port | `11436` | No |
| `MEMEVOLVE_UPSTREAM_BASE_URL` | Upstream API URL | `http://localhost:11434/v1` | Yes |
| `MEMEVOLVE_UPSTREAM_API_KEY` | Upstream API key | None | No |
| `MEMEVOLVE_API_MEMORY_INTEGRATION` | Enable memory features | `true` | No |
| `MEMEVOLVE_MEMORY_BASE_URL` | Memory API endpoint | Falls back to upstream | No |
| `MEMEVOLVE_MEMORY_API_KEY` | Memory API key | Falls back to upstream | No |
| `MEMEVOLVE_EMBEDDING_BASE_URL` | Embedding API URL | Falls back to upstream | No |
| `MEMEVOLVE_EMBEDDING_API_KEY` | Embedding API key | Falls back to upstream | No |

### Memory Configuration

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `MEMEVOLVE_API_MAX_RETRIES` | Global max retries for all API calls | `3` | |
| `MEMEVOLVE_DEFAULT_TOP_K` | Global default retrieval count | `5` | |
| `MEMEVOLVE_STORAGE_PATH` | Memory storage location | `./data/memory.json` | |
| `MEMEVOLVE_STORAGE_BACKEND_TYPE` | Storage type | `json` | `json`, `vector`, `graph` |
| `MEMEVOLVE_DEFAULT_TOP_K` | Default retrieval count | `5` | |
| `MEMEVOLVE_RETRIEVAL_STRATEGY_TYPE` | Retrieval method | `hybrid` | `semantic`, `keyword`, `hybrid` |
| `MEMEVOLVE_RETRIEVAL_SEMANTIC_WEIGHT` | Semantic vs keyword balance | `0.7` | |
| `MEMEVOLVE_MANAGEMENT_ENABLE_AUTO_MANAGEMENT` | Auto memory management | `true` | |
| `MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD` | Memory size limit | `1000` | |
| `MEMEVOLVE_LOG_MIDDLEWARE_ENABLE` | Enable detailed quality scoring logs | `false` | Set to `true` for debugging |
| `MEMEVOLVE_QUALITY_BIAS_CORRECTION` | Enable bias correction for model types | `true` | Ensures fair evaluation |
| `MEMEVOLVE_QUALITY_MIN_THRESHOLD` | Minimum score for experience storage | `0.1` | Filter low-quality responses |

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

base_url = "http://localhost:11436"

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
curl http://localhost:11436/health

curl http://localhost:11436/memory/stats

curl -X POST http://localhost:11436/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "database optimization", "top_k": 3}'

# Add memory unit
curl -X POST http://localhost:11436/memory/add \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Use EXPLAIN ANALYZE to understand query execution plans.",
    "type": "lesson",
    "tags": ["database", "postgresql"]
  }'
```

## 📊 Monitoring & Dashboard

### Real-time Health Dashboard
```http
GET /dashboard
```

**Response**: HTML page with comprehensive system monitoring dashboard.

**Features**:
- Real-time system health metrics
- API performance monitoring (requests, success rates, response times)
- Memory system statistics and utilization
- Evolution system progress and fitness scores
- Resource usage tracking (logs, storage)
- Dark mode toggle with persistent preferences
- Auto-refresh every 30 seconds

### Dashboard Data API
```http
GET /dashboard-data
```

**Response**:
```json
{
  "timestamp": "2026-01-24T20:15:14.564445",
  "system_health": {
    "status": "healthy",
    "uptime_percentage": 100,
    "last_check": "2026-01-24T20:15:14.564455"
  },
  "api_performance": {
    "total_requests": 86,
    "success_rate": 98.8,
    "avg_response_time": 0.0,
    "upstream_response_time": 0,
    "memory_api_response_time": 0,
    "embedding_api_response_time": 0,
    "tokens_per_sec": 0
  },
  "memory_system": {
    "total_experiences": 0,
    "retrieval_count": 0,
    "avg_retrieval_time": 0,
    "utilization": 0.0,
    "file_size_kb": 0.0
  },
  "quality_scoring": {
    "total_responses_scored": 45,
    "average_quality_score": 0.324,
    "reasoning_responses": 9,
    "direct_responses": 36,
    "bias_correction_active": true,
    "quality_trend": "stable"
  },
  "evolution_system": {
    "status": "Inactive",
    "current_genotype": "None",
    "generations_completed": 0,
    "fitness_score": 0.0,
    "response_quality_score": 0.324
  }
}
```

Provides JSON data for dashboard widgets and external monitoring integration, including quality scoring metrics.

## 🔧 Troubleshooting

- **Connection Refused**: Check if MemEvolve server is running on correct host/port
- **Empty Results**: Verify memory has been populated and search query is relevant
- **Memory Scores Show N/A**: Update to latest code - this issue has been resolved
- **Quality Scoring Issues**: Enable debug logging with `MEMEVOLVE_LOG_MIDDLEWARE_ENABLE=true`
- **Slow Responses**: Check upstream API performance and network connectivity
- **Authentication Errors**: Verify API keys are correctly configured

For detailed troubleshooting, see the [troubleshooting guide](troubleshooting.md) and [quality scoring documentation](quality-scoring.md).