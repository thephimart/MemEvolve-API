# Centralized Logging System

Comprehensive logging architecture with component-specific event routing for enhanced observability, debugging, and system monitoring in MemEvolve deployments.

## Overview

The centralized logging system provides fine-grained control over what gets logged and where, enabling efficient debugging without performance overhead. Each major system component writes to dedicated log files with structured formatting.

## 🏗️ Architecture

### Component-Based Routing

```
logs/
├── api-server/api_server.log          # HTTP requests and API operations
├── middleware/enhanced_middleware.log   # Request processing and metrics
├── memory/memory.log                  # Memory operations and retrievals
├── evolution/evolution.log             # Evolution cycles and mutations
└── memevolve.log                       # System-wide events and startup
```

### Data Flow

1. **Component Events** → Component Logger (api_server, middleware, memory, evolution, system)
2. **Component Logger** → Structured Entry with Timestamp & Level
3. **Log Router** → Component-Specific File (based on enable flags)
4. **File System** → Rotated Log Files (size-based management)

## 🔧 Configuration

### Environment Variables

All logging configuration is handled via environment variables:

```bash
# Component Logging Controls
MEMEVOLVE_LOG_API_SERVER_ENABLE=true     # HTTP requests and API events
MEMEVOLVE_LOG_MIDDLEWARE_ENABLE=true      # Request processing and metrics  
MEMEVOLVE_LOG_MEMORY_ENABLE=true          # Memory operations and retrievals
MEMEVOLVE_LOG_EVOLUTION_ENABLE=true       # Evolution cycles and mutations
MEMEVOLVE_LOG_MEMEVOLVE_ENABLE=true       # System-wide events and startup

# Additional Logging Controls
MEMEVOLVE_LOG_OPERATION_ENABLE=true       # In-memory operation tracking
MEMEVOLVE_LOG_LEVEL=DEBUG                # Global log level (DEBUG, INFO, WARNING, ERROR)
MEMEVOLVE_LOGGING_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
MEMEVOLVE_LOGGING_MAX_LOG_SIZE_MB=1024   # Maximum log file size before rotation
```

### Default Settings

- **All Components**: Enabled by default for comprehensive monitoring
- **Log Level**: DEBUG for development, INFO for production
- **Max File Size**: 1024MB before rotation
- **Format**: Structured with timestamp, component name, and level

## 📊 Log Contents

### API Server Logs (`api-server/api_server.log`)

HTTP request/response logging with performance metrics:

```bash
2026-02-03 20:15:32,123 - api_server - INFO - POST /v1/chat/completions - 200 - 1.234s - 45 tokens
2026-02-03 20:15:33,456 - api_server - INFO - GET /v1/models - 200 - 0.045s
```

### Middleware Logs (`middleware/enhanced_middleware.log`)

Memory integration and request processing:

```bash
2026-02-03 20:15:32,124 - middleware - INFO - Memory integration started for request
2026-02-03 20:15:32,234 - middleware - INFO - Retrieved 3 memories (relevance: 0.78)
2026-02-03 20:15:32,345 - middleware - INFO - Memory injection completed successfully
```

### Memory Logs (`memory/memory.log`)

Core memory system operations:

```bash
2026-02-03 20:15:32,234 - memory - INFO - Storing experience: type=lesson, relevance=0.89
2026-02-03 20:15:32,345 - memory - INFO - Retrieval: query="debug memory", top_k=3, results=2
2026-02-03 20:15:32,456 - memory - INFO - Management: pruned 5 old memories (LRU strategy)
```

### Evolution Logs (`evolution/evolution.log`)

Evolution cycles with detailed parameter tracking:

```bash
2026-02-03 20:10:00,123 - evolution - INFO - Evolution cycle 23 started
2026-02-03 20:10:05,456 - evolution - INFO - 🧬 EVOLUTION PARAMETER CHANGES - Genotype: agentkb_v3_optimized_23
2026-02-03 20:10:05,457 - evolution - INFO - 📊 Retrieval Component (7 parameters):
2026-02-03 20:10:05,458 - evolution - INFO -   🔧 strategy_type: hybrid → semantic
2026-02-03 20:10:05,459 - evolution - INFO -   📈 default_top_k: 5 → 7 (+40.0%)
2026-02-03 20:10:05,460 - evolution - INFO -   ⚖️  similarity_threshold: 0.700 → 0.850 (+21.4%)
2026-02-03 20:10:05,461 - evolution - INFO - ⚡ PERFORMANCE IMPLICATIONS:
2026-02-03 20:10:05,462 - evolution - INFO -    🧠 Semantic retrieval may improve relevance but increase latency
2026-02-03 20:10:05,463 - evolution - INFO - 📋 PARAMETER SUMMARY: 3/18 parameters changed
2026-02-03 20:10:15,123 - evolution - INFO - Evolution cycle 23 completed - fitness: 1.087 (+5.9%)
```

### System Logs (`memevolve.log`)

Application-level events and startup:

```bash
2026-02-03 20:00:00,123 - memevolve - INFO - MemEvolve-API v2.1.0 starting up
2026-02-03 20:00:00,234 - memevolve - INFO - Configuration loaded: 137 variables
2026-02-03 20:00:00,345 - memevolve - INFO - Centralized logging system initialized
2026-02-03 20:00:05,456 - memevolve - INFO - API server ready on http://127.0.0.1:11436
```

## 🎯 Enhanced Parameter Tracking

When `MEMEVOLVE_LOG_EVOLUTION_ENABLE=true`, the system provides detailed evolution parameter analysis:

### Before/After Analysis
- **Parameter Changes**: Exact old → new values for all mutations
- **Percentage Impact**: Quantified change magnitude for each parameter
- **Component Grouping**: Organized by Retrieval, Encoder, Management components
- **Performance Implications**: Automatic analysis of potential impact

### Example Output

```bash
🧬 EVOLUTION PARAMETER CHANGES - Genotype: agentkb_v3_optimized_22
📊 Retrieval Component (7 parameters):
  🔧 strategy_type: hybrid → semantic
  📈 default_top_k: 5 → 7 (+40.0%)
  ⚖️  similarity_threshold: 0.700 → 0.850 (+21.4%)
  🔄 semantic_cache_enabled: true → true
⚡ PERFORMANCE IMPLICATIONS:
   🧠 Semantic retrieval may improve relevance but increase latency
📋 PARAMETER SUMMARY: 3/18 parameters changed

✅ Successfully reconfigured retrieval → SemanticRetrievalStrategy
🔧 retrieval configuration: strategy=semantic, threshold=0.850, top_k=7
```

## 🚀 Performance Optimization

### Production Configuration

```bash
# Production-ready logging setup
MEMEVOLVE_LOG_LEVEL=INFO                    # Reduce verbosity
MEMEVOLVE_LOG_API_SERVER_ENABLE=true         # Track API requests
MEMEVOLVE_LOG_MEMORY_ENABLE=true             # Monitor memory operations
MEMEVOLVE_LOG_MIDDLEWARE_ENABLE=false        # Reduce overhead
MEMEVOLVE_LOG_EVOLUTION_ENABLE=false         # Disable detailed evolution logs
MEMEVOLVE_LOG_OPERATION_ENABLE=false         # Disable in-memory tracking
MEMEVOLVE_LOGGING_MAX_LOG_SIZE_MB=512       # Smaller log files
```

### Development Configuration

```bash
# Development debugging setup
MEMEVOLVE_LOG_LEVEL=DEBUG                   # Maximum verbosity
MEMEVOLVE_LOG_API_SERVER_ENABLE=true         # HTTP requests debugging
MEMEVOLVE_LOG_MIDDLEWARE_ENABLE=true          # Request processing details
MEMEVOLVE_LOG_MEMORY_ENABLE=true              # Memory operation tracking
MEMEVOLVE_LOG_EVOLUTION_ENABLE=true           # Evolution parameter tracking
MEMEVOLVE_LOG_OPERATION_ENABLE=true           # In-memory operation analysis
MEMEVOLVE_LOGGING_MAX_LOG_SIZE_MB=2048      # Larger files for debugging
```

### High-Throughput Configuration

```bash
# Minimal logging for performance-critical deployments
MEMEVOLVE_LOG_LEVEL=WARNING                  # Only warnings and errors
MEMEVOLVE_LOG_API_SERVER_ENABLE=false        # Disable API request logging
MEMEVOLVE_LOG_MIDDLEWARE_ENABLE=false         # Disable middleware logging
MEMEVOLVE_LOG_MEMORY_ENABLE=false            # Disable memory operation logging
MEMEVOLVE_LOG_EVOLUTION_ENABLE=false         # Disable evolution logging
MEMEVOLVE_LOG_OPERATION_ENABLE=false         # Disable in-memory tracking
MEMEVOLVE_LOGGING_MAX_LOG_SIZE_MB=256       # Small log files
```

## 🔍 Log Analysis

### Monitoring with Tools

```bash
# Real-time log monitoring
tail -f logs/api-server/api_server.log
tail -f logs/evolution/evolution.log

# Multi-component monitoring
tail -f logs/*/*.log

# Error monitoring across all components
grep -r "ERROR" logs/

# Evolution parameter change tracking
grep -r "EVOLUTION PARAMETER CHANGES" logs/evolution/

# Performance analysis
grep -r "Performance" logs/
```

### Log Rotation and Management

The system automatically manages log file sizes:

- **Size-Based Rotation**: Files rotate when reaching `MEMEVOLVE_LOGGING_MAX_LOG_SIZE_MB`
- **Backup Retention**: Old files are timestamped and retained
- **Cleanup Automation**: Use `scrubber.sh` for manual cleanup

```bash
# Manual log cleanup
./scripts/scrubber.sh

# Selective log cleanup (development)
rm -rf logs/*/

# Production log rotation (automated)
find logs/ -name "*.log" -size +1G -exec mv {} {}.old \;
```

## 🛠️ Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure write permissions for logs directory
2. **Disk Space**: Monitor log sizes with size limits
3. **Performance Impact**: Disable DEBUG level in production
4. **Missing Logs**: Check component enable flags

### Debug Steps

```bash
# Check logging configuration
env | grep MEMEVOLVE_LOG

# Verify directory structure
ls -la logs/

# Test component logging
curl -X POST http://localhost:11436/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "test", "messages": [{"role": "user", "content": "test"}]}' && tail -n 5 logs/api-server/api_server.log

# Monitor evolution logging
tail -f logs/evolution/evolution.log
```

## 📚 Integration

### Custom Logging

Components can integrate with the centralized system:

```python
import logging
from memevolve.utils.logging import setup_component_logging

# Get configured logger
logger = setup_component_logging("custom_component", config)

# Use structured logging
logger.info("Custom operation completed", extra={
    "operation": "process_data",
    "count": 42,
    "duration": 1.234
})
```

### Log Analysis Tools

```python
# Log file parser for analysis
from memevolve.utils.log_analyzer import LogAnalyzer

analyzer = LogAnalyzer("logs/evolution/evolution.log")
parameter_changes = analyzer.get_evolution_changes()
performance_trends = analyzer.get_performance_trends()
```

---

This centralized logging system provides comprehensive observability while maintaining performance through configurable component isolation and efficient event routing.