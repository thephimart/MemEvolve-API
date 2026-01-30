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
> **⚠️ Important**: All management endpoints are in development. May not function as expected or at all.

Direct endpoints for memory system management and inspection.

### Base URL
```
http://localhost:11436
```

### Endpoints Overview

#### Health & Status
- `GET /health` - System health check
- `GET /memory/stats` - Memory system statistics
- `GET /memory/config` - Memory system configuration
- `GET /evolution/status` - Evolution system status

#### Memory Operations
- `POST /memory/search` - Search memory units
- `POST /memory/clear` - Clear memory data

#### Evolution Control
- `POST /evolution/start` - Start evolution process
- `POST /evolution/stop` - Stop evolution process
- `POST /evolution/record-request` - Record API request metrics
- `POST /evolution/record-retrieval` - Record memory retrieval metrics

#### Dashboard & Monitoring
- `GET /dashboard` - HTML dashboard interface
- `GET /dashboard-data` - JSON dashboard data
- `GET /web/dashboard/dashboard.css` - Dashboard styles
- `GET /web/dashboard/dashboard.js` - Dashboard scripts
- `GET /docs` - API documentation (Swagger)
- `GET /redoc` - API documentation (ReDoc)
- `GET /openapi.json` - OpenAPI schema

### Health & Status

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
  "evolution_status": {
    "is_running": false,
    "current_genotype": null,
    "population_size": 0,
    "evolution_cycles_completed": 1,
    "last_evolution_time": 1769364174.658245,
    "api_requests_total": 870,
    "average_response_time": 43.30,
    "memory_retrievals_total": 435,
    "average_retrieval_time": 17.61,
    "response_quality_score": 0.454,
    "memory_utilization": 0.0,
    "fitness_score": 0.495
  },
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
  "total_experiences": 458,
  "retrieval_count": 2,
  "last_updated": "2026-01-25T19:15:34.328531+00:00Z",
  "architecture": "unknown"
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
    "content": "The round shape of pizzas is a result of their baking process...",
    "score": 0.5,
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
```json
{
  "is_running": false,
  "current_genotype": null,
  "population_size": 0,
  "evolution_cycles_completed": 1,
  "last_evolution_time": 1769364174.658245,
  "api_requests_total": 870,
  "average_response_time": 43.30,
  "memory_retrievals_total": 435,
  "average_retrieval_time": 17.61,
  "response_quality_score": 0.454,
  "memory_utilization": 0.0,
  "fitness_score": 0.495,
  "metrics_persistence": {
    "metrics_directory": "data/metrics",
    "metrics_files_count": 41
  }
}
```

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
| `MEMEVOLVE_EVOLUTION_CYCLE_SECONDS` | Evolution cycle rate | `60` | `MEMEVOLVE_ENABLE_EVOLUTION=true` |
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
- **Memory API**: Uses `MEMEVOLVE_MEMORY_API_KEY` (falls back to upstream if not set)

## 📊 Response Codes

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (endpoint doesn't exist)
- `422`: Unprocessable Entity (validation failed)
- `500`: Internal Server Error
- `503`: Service Unavailable (upstream API down or system not initialized)

## 🚨 Current Issues & Limitations

### **🔴 v2.0.0 Development Issues**

**Critical Issues Affecting Core Functionality:**

- **Memory Encoding Verbosity**: All encoded memories contain verbose prefixes instead of direct insights
  - **Impact**: 100% of new memory creation affected
  - **Detection**: Memories start with "The experience provided a partial overview..."
  - **Status**: Fix identified in dev_tasks.md, awaiting implementation

- **Negative Token Efficiency**: Consistent -1000+ token losses per request
  - **Impact**: Business analytics and ROI calculations are incorrect
  - **Detection**: Token efficiency scores consistently below -0.5
  - **Status**: Baseline calculation fixes needed

- **Static Business Scoring Values**: All responses show identical scores
  - **Impact**: Business impact analytics provide no meaningful insights
  - **Detection**: All responses show business_value_score: 0.3 and roi_score: 0.1
  - **Status**: Dynamic scoring integration required

- **Top-K Configuration Sync Failures**: Evolution settings don't propagate to runtime
  - **Impact**: Evolution parameter changes don't take effect in memory retrieval
  - **Detection**: Evolution logs show top_k=11 but runtime uses top_k=3
  - **Status**: Configuration sync improvements needed

### **Known Limitations**

- **Dashboard Data API**: Business analytics integration currently failing - may return error for comprehensive metrics
- **Memory Search**: May return empty content results with valid scores
- **Evolution System**: Auto-evolution requires sufficient activity to trigger
- **Quality Scoring**: Some responses show N/A scores in older versions (resolved in current code)

### **Development vs Production**

**Main API Ready for Use**: The OpenAI-compatible chat completions endpoint is fully functional and ready for production use.

**Management API in Testing**: Management endpoints and evolution/scoring features are in active development and may not function as expected. Use this branch for:
- Development and testing of new features
- Understanding system architecture and capabilities
- Contributing to issue resolution
- Testing management endpoints and advanced features

For detailed issue status and implementation plans, see [troubleshooting guide](troubleshooting.md#known-issues-in-v20) and [dev_tasks.md](../../dev_tasks.md).

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

# Memory statistics
curl http://localhost:11436/memory/stats

# Search memory
curl -X POST http://localhost:11436/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "database optimization", "limit": 3}'

# Evolution status
curl http://localhost:11436/evolution/status

# Dashboard data
curl http://localhost:11436/dashboard-data

# Start evolution
curl -X POST http://localhost:11436/evolution/start

# Stop evolution
curl -X POST http://localhost:11436/evolution/stop
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
Returns comprehensive system metrics including:
- System health status
- API performance metrics (requests, response times)
- Memory system statistics
- Quality scoring analytics
- Evolution system progress

**Note**: Currently experiencing integration issues with business analytics. May return error for comprehensive metrics.

### Static Resources
```http
GET /web/dashboard/dashboard.css
GET /web/dashboard/dashboard.js
```

**Response**: CSS and JavaScript files for dashboard functionality.

## 🔧 Troubleshooting

- **Connection Refused**: Check if MemEvolve server is running on correct host/port
- **Empty Results**: Verify memory has been populated and search query is relevant
- **Dashboard Data Error**: Business analytics integration issue - check comprehensive metrics availability
- **Memory Search Issues**: May return empty content with valid scores - check memory population
- **Quality Scoring Issues**: Enable debug logging with `MEMEVOLVE_LOG_MIDDLEWARE_ENABLE=true`
- **Slow Responses**: Check upstream API performance and network connectivity
- **Authentication Errors**: Verify API keys are correctly configured
- **Evolution Not Starting**: Requires sufficient API activity - check auto-evolution trigger conditions

For detailed troubleshooting, see the [troubleshooting guide](troubleshooting.md) and [quality scoring documentation](quality-scoring.md).

---

## 📋 Documentation Updates Made

This API reference has been updated to reflect the current implementation status:

### Fixed Issues:
1. **Duplicate Endpoints**: Removed duplicate `/memory/clear` definition
2. **Outdated Response Formats**: Updated all examples with actual current responses
3. **Missing Endpoints**: Added comprehensive endpoint list (18 total)
4. **Incorrect Parameters**: Fixed `top_k` vs `limit` parameter naming
5. **Authentication Details**: Clarified API key fallback behavior

### Current System Status (as of last update):
- **Total Available Endpoints**: 18
- **Memory Units**: 458 experiences stored
- **Evolution Cycles**: 1 completed, auto-evolution ready
- **Health Status**: ✅ All systems operational
- **Known Issues**: Dashboard data API integration problems

### API Coverage:
- ✅ All endpoints tested and documented
- ✅ Real response examples provided
- ✅ Current limitations identified
- ✅ Troubleshooting guidance updated