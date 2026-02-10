# MemEvolve Deployment Guide

This guide covers deploying MemEvolve in development and production environments.

## 🚨 v2.1.0 Deployment Status - NOT PRODUCTION READY

**IMPORTANT**: Core memory system is functional (75%+ success rate) with robust error handling. Evolution system requires analysis and implementation. This is development branch code, not ready for production deployment.

### **✅ Functional Core Systems**
- **OpenAI-Compatible API**: Chat completions endpoint operational for development
- **Memory Retrieval & Injection**: Context enhancement with growing database
- **Experience Encoding**: Memory creation with schema transformation (75%+ success)
- **Schema & JSON Handling**: Robust transformation and repair systems implemented
- **Centralized Configuration**: Unified logging and token management

### **⚠️ Systems Pending Analysis**
- **Evolution System**: Current state unknown, next priority for investigation
- **Management & Analytics**: Framework in place, development pending

## 🚀 Development Deployment

### Quick Start

```bash
# 1. Clone and navigate
git clone https://github.com/thephimart/MemEvolve-API.git
cd MemEvolve-API

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Start the API server
python scripts/start_api.py
```

## ⚙️ Environment Configuration

Key production settings in `.env`:

```bash
# API Server
MEMEVOLVE_API_HOST=0.0.0.0
MEMEVOLVE_API_PORT=11436

# Upstream LLM (REQUIRED)
MEMEVOLVE_UPSTREAM_BASE_URL=your-llm-endpoint
MEMEVOLVE_UPSTREAM_API_KEY=your-api-key

# Memory LLM (OPTIONAL)
MEMEVOLVE_MEMORY_BASE_URL=your-memory-endpoint

# Data directories
MEMEVOLVE_DATA_DIR=/path/to/data
MEMEVOLVE_LOGS_DIR=/path/to/logs

# Evolution settings
MEMEVOLVE_ENABLE_EVOLUTION=true
```

## 🔧 Production Considerations

### Performance
- Use appropriate `default_top_k` for your workload
- Configure timeouts based on model response times
- Enable auto-management for memory pruning

### Security
- Use API keys instead of environment variables for secrets
- Configure proper firewall rules
- Enable SSL/TLS termination

### Monitoring
- Monitor logs in `logs/` directory
- Check API health endpoint regularly
- Track memory system performance

### Scaling
- Use load balancer for multiple instances
- Consider separate storage for high volume
- Monitor resource usage

## 🛠️ Service Management

### Starting the Service
```bash
# Development mode
source .venv/bin/activate
python scripts/start_api.py

# Background mode
nohup python scripts/start_api.py > logs/api.log 2>&1 &
```

### Checking Status
```bash
# API health
curl http://localhost:11436/health

# Memory system status
curl http://localhost:11436/memory/stats
```

### Stopping the Service
```bash
# Find and kill the process
pkill -f "python scripts/start_api.py"
```

## 🔍 Troubleshooting

### Common Issues

**Service won't start:**
- Check environment variables in `.env`
- Verify LLM endpoints are accessible
- Check port availability

**Memory system errors:**
- Verify storage directory permissions
- Check disk space
- Review memory configuration

**Performance issues:**
- Monitor memory usage
- Check LLM response times
- Review retrieval settings

### Getting Help

1. Check application logs: `tail -f logs/api.log`
2. Verify upstream connectivity: `curl $MEMEVOLVE_UPSTREAM_BASE_URL/health`
3. Test memory system: `curl http://localhost:11436/memory/stats`
4. Review configuration: `cat .env`

For development deployment support, see the [troubleshooting guide](../api/troubleshooting.md). Note: Core memory system is functional for development testing. Evolution system requires analysis and implementation. NOT PRODUCTION READY.