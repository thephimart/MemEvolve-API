# MemEvolve Deployment Guide

This guide covers deploying MemEvolve in production environments, including Docker deployment, scaling strategies, and monitoring best practices.

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
# 1. Clone and navigate
git clone https://github.com/thephimart/memevolve.git
cd memevolve

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings

# 3. Deploy with Docker Compose
docker-compose up -d

# 4. Check status
curl http://localhost:11436/health
```

### Recommended Port Scheme

For local development and testing, the following port assignments work well:

- **11433**: Memory LLM service (LFM-2.5-1.2B-Instruct recommended)
- **11434**: Upstream LLM service (GPT-OSS-20B or LFM-2.5-1.2B-Instruct)
- **11435**: Embedding service (nomic-embed-text-v2-moe recommended)
- **11436**: MemEvolve API server
- **11437**: Future web UI (reserved)

This scheme keeps related services grouped together and avoids conflicts with common development ports.

### Docker Compose Configuration

```yaml
version: '3.8'

services:
  memevolve-api:
    build: .
    ports:
      - "11436:11436"
    environment:
      # API Configuration
      - MEMEVOLVE_API_HOST=0.0.0.0
      - MEMEVOLVE_API_PORT=11436
      - MEMEVOLVE_UPSTREAM_BASE_URL=http://llm-service:8000/v1
      - MEMEVOLVE_UPSTREAM_API_KEY=${LLM_API_KEY}

      # Memory Configuration
      - MEMEVOLVE_API_MEMORY_INTEGRATION=true
      - MEMEVOLVE_LLM_BASE_URL=http://llm-service:8000/v1
      - MEMEVOLVE_LLM_API_KEY=${LLM_API_KEY}
      - MEMEVOLVE_STORAGE_PATH=/app/data/memory.json
      - MEMEVOLVE_RETRIEVAL_TOP_K=5
      - MEMEVOLVE_MANAGEMENT_ENABLE_AUTO_MANAGEMENT=true
      - MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD=5000

      # System Configuration
      - MEMEVOLVE_LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - llm-service
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11436/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  llm-service:
    # Example: vLLM deployment
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    command: >
      --model microsoft/DialoGPT-medium
      --host 0.0.0.0
      --port 8000
    volumes:
      - ./models:/root/.cache/huggingface
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Optional: Monitoring stack
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-data:
```

## 🚀 Production Configuration

### Environment Variables

```bash
# API Server
MEMEVOLVE_API_HOST=0.0.0.0
MEMEVOLVE_API_PORT=11436

# Upstream LLM
MEMEVOLVE_UPSTREAM_BASE_URL=https://your-llm-service.com/v1
MEMEVOLVE_UPSTREAM_API_KEY=your-production-key

# Memory System
MEMEVOLVE_LLM_BASE_URL=https://your-llm-service.com/v1
MEMEVOLVE_LLM_API_KEY=your-production-key
MEMEVOLVE_STORAGE_PATH=/data/memory.db
MEMEVOLVE_RETRIEVAL_TOP_K=10
MEMEVOLVE_MANAGEMENT_ENABLE_AUTO_MANAGEMENT=true
MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD=10000

# Performance
MEMEVOLVE_LOG_LEVEL=WARNING
MEMEVOLVE_STORAGE_BACKEND_TYPE=vector
MEMEVOLVE_RETRIEVAL_STRATEGY_TYPE=hybrid

# Security
MEMEVOLVE_API_ENABLE_CORS=false
MEMEVOLVE_API_TRUSTED_PROXIES=10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
```

### Memory Storage for Production

#### PostgreSQL Backend (Recommended for Production)

```python
from components.store import PostgreSQLStore

config.storage_backend = PostgreSQLStore(
    host="prod-db-host",
    port=5432,
    database="memevolve",
    user="memevolve_user",
    password="secure-password",
    table_name="memories"
)
```

#### Redis for High Performance

```python
from components.store import RedisStore

config.storage_backend = RedisStore(
    host="redis-cluster",
    port=6379,
    db=0,
    password="redis-password",
    use_vector_extensions=True  # Requires Redis Stack
)
```

## 📊 Monitoring and Observability

### Health Checks

```bash
# Basic health check
curl http://localhost:11436/health

# Detailed memory stats
curl http://localhost:8001/memory/stats

# Performance metrics
curl http://localhost:8001/metrics
```

### Prometheus Metrics

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'memevolve'
    static_configs:
      - targets: ['memevolve-api:11436']
    metrics_path: '/metrics'
```

### Grafana Dashboard

Import the MemEvolve dashboard from `monitoring/grafana-dashboard.json`:

```json
{
  "dashboard": {
    "title": "MemEvolve API Monitoring",
    "tags": ["memevolve", "ai", "memory"],
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(memevolve_api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "memevolve_memory_total_experiences",
            "legendFormat": "Total Experiences"
          }
        ]
      }
    ]
  }
}
```

### Log Aggregation

```bash
# Structured logging to JSON
MEMEVOLVE_LOG_FORMAT="json"
MEMEVOLVE_LOG_FILE="/var/log/memevolve/api.log"

# Log rotation
MEMEVOLVE_LOGGING_MAX_LOG_SIZE_MB=100
MEMEVOLVE_LOGGING_BACKUP_COUNT=5
```

## 🔄 Scaling Strategies

### Horizontal Scaling

```yaml
# Scale API instances
services:
  memevolve-api:
    deploy:
      replicas: 3
    environment:
      - MEMEVOLVE_STORAGE_BACKEND_TYPE=redis  # Shared storage
```

### Load Balancing

```yaml
# NGINX load balancer
upstream memevolve_backend {
    server memevolve-api-1:11436;
    server memevolve-api-2:11436;
    server memevolve-api-3:11436;
}

server {
    listen 80;
    location / {
        proxy_pass http://memevolve_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Database Scaling

#### Connection Pooling

```python
# For high-throughput scenarios
config.db_connection_pool_size = 20
config.db_max_overflow = 30
config.db_pool_recycle = 3600
```

#### Read Replicas

```python
# Use read replicas for memory queries
config.storage_backend = PostgreSQLStore(
    master_host="db-master",
    replica_hosts=["db-replica-1", "db-replica-2"],
    use_read_replicas=True
)
```

## 🔒 Security Considerations

### API Security

```bash
# Disable external access to management endpoints
MEMEVOLVE_API_EXPOSE_MANAGEMENT=false

# Enable CORS only for trusted domains
MEMEVOLVE_API_CORS_ORIGINS=https://yourapp.com,https://admin.yourapp.com

# API key authentication for management endpoints
MEMEVOLVE_API_MANAGEMENT_KEY=your-secure-key
```

### Network Security

```yaml
# Internal network for API communication
services:
  memevolve-api:
    networks:
      - internal
    # No external ports exposed

  nginx:
    ports:
      - "443:443"
    networks:
      - internal
      - public
    # SSL termination and rate limiting
```

### Data Security

```python
# Encrypt sensitive memory content
config.enable_encryption = True
config.encryption_key = os.getenv("MEMEVOLVE_ENCRYPTION_KEY")

# PII detection and masking
config.enable_pii_masking = True
config.pii_patterns = ["email", "phone", "ssn"]
```

## 🚨 Backup and Recovery

### Automated Backups

```bash
#!/bin/bash
# Daily backup script
BACKUP_DIR="/backups/memevolve"
DATE=$(date +%Y%m%d_%H%M%S)

# Export memories
curl -X POST http://localhost:8001/memory/export \
  -H "Content-Type: application/json" \
  -d '{"format": "json"}' \
  -o "$BACKUP_DIR/memories_$DATE.json"

# Backup configuration
cp /app/.env "$BACKUP_DIR/config_$DATE.env"

# Compress and rotate
tar -czf "$BACKUP_DIR/backup_$DATE.tar.gz" -C "$BACKUP_DIR" .
find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +30 -delete
```

### Disaster Recovery

```bash
#!/bin/bash
# Recovery script
BACKUP_FILE="/backups/memevolve/backup_latest.tar.gz"

# Stop services
docker-compose down

# Restore data
tar -xzf "$BACKUP_FILE" -C /tmp/restore
cp /tmp/restore/memories.json /app/data/
cp /tmp/restore/config.env /app/.env

# Restart services
docker-compose up -d

# Verify restoration
curl http://localhost:11436/health
curl http://localhost:8001/memory/stats
```

## 🧪 Testing in Production

### Staging Environment

```yaml
# staging-docker-compose.yml
version: '3.8'

services:
  memevolve-api-staging:
    image: memevolve-api:latest
    environment:
      - MEMEVOLVE_ENV=staging
      - MEMEVOLVE_STORAGE_PATH=/app/data/staging-memory.json
    ports:
      - "8002:11436"  # Different port for staging
```

### Canary Deployment

```yaml
# Route 10% of traffic to new version
services:
  memevolve-api-v2:
    image: memevolve-api:v2.0.0
    environment:
      - MEMEVOLVE_STORAGE_PATH=/app/data/canary-memory.json
    deploy:
      replicas: 1  # Single canary instance
```

## 📈 Performance Tuning

### Memory Optimization

```python
# Production memory settings
config = MemorySystemConfig(
    # Limit memory growth
    auto_prune_threshold=50000,
    max_memory_age_days=90,

    # Optimize retrieval
    default_retrieval_top_k=5,  # Balance quality vs speed
    retrieval_cache_size=10000,

    # Batch processing
    batch_size=100,
    num_workers=8
)
```

### LLM Optimization

```python
# Connection pooling for upstream API
config.upstream_connection_pool_size = 20
config.upstream_timeout = 30
config.upstream_retry_attempts = 3

# Separate encoding LLM (smaller/faster model)
config.encoding_llm_base_url = "http://fast-llm:8000/v1"
config.encoding_llm_model = "distilbert-llm"
```

## 🔧 Maintenance Tasks

### Regular Maintenance

```bash
# Weekly: Check memory health
curl http://localhost:11436/health
curl http://localhost:8001/memory/stats

# Monthly: Optimize storage
curl -X POST http://localhost:8001/maintenance/optimize

# Quarterly: Full backup and integrity check
# Note: Implement custom backup and integrity scripts as needed
```

### Automated Maintenance

For automated maintenance, consider using cron jobs or scheduled tasks to run the provided cleanup scripts periodically.
```

## 📞 Support and Troubleshooting

### Production Monitoring Checklist

- [ ] API response times < 500ms (95th percentile)
- [ ] Memory retrieval adds < 200ms latency
- [ ] Error rate < 1%
- [ ] Memory size stays within configured limits
- [ ] Backup integrity verified weekly

### Getting Help

1. Check application logs: `docker-compose logs memevolve-api`
2. Verify upstream LLM connectivity: `curl $MEMEVOLVE_UPSTREAM_BASE_URL/health`
3. Test memory system: `curl http://localhost:8001/memory/stats`
4. Review configuration: `docker exec memevolve-api env | grep MEMEVOLVE`
5. Contact support with diagnostic information

---

This deployment guide ensures MemEvolve runs reliably and efficiently in production environments. Monitor performance, implement proper security measures, and maintain regular backups for optimal operation.