# Auto-Evolution Configuration Guide

This guide covers MemEvolve's intelligent auto-evolution system with multi-trigger automatic optimization.

## Overview

MemEvolve's auto-evolution system continuously optimizes memory architecture based on multiple intelligent triggers, eliminating the need for manual intervention or scheduled API calls. The system monitors performance in real-time and automatically initiates evolution when optimization opportunities are detected.

## 🎯 Auto-Evolution Triggers

The system supports four independent triggers that can initiate automatic evolution:

### 1. Request Count Trigger
**Environment Variable**: `MEMEVOLVE_AUTO_EVOLUTION_REQUESTS`  
**Default**: `500`  
**Purpose**: Evolution begins after processing N requests

When enabled, the system automatically initiates evolution after processing the configured number of API requests. This ensures sufficient data has been collected for meaningful optimization.

```bash
# Start evolution after every 1000 requests
MEMEVOLVE_AUTO_EVOLUTION_REQUESTS=1000
```

### 2. Performance Degradation Trigger
**Environment Variable**: `MEMEVOLVE_AUTO_EVOLUTION_DEGRADATION`  
**Default**: `0.2` (20%)  
**Purpose**: Evolution begins when performance degrades by X%

The system continuously monitors key performance indicators and automatically triggers evolution when performance drops below the threshold. This provides automatic recovery from performance issues.

```bash
# Trigger evolution when performance drops by 15%
MEMEVOLVE_AUTO_EVOLUTION_DEGRADATION=0.15
```

### 3. Fitness Plateau Trigger
**Environment Variable**: `MEMEVOLVE_AUTO_EVOLUTION_PLATEAU`  
**Default**: `5`  
**Purpose**: Evolution begins when fitness is stable for N generations

Detects when the optimization process has reached a local maximum and continues searching for better configurations.

```bash
# Trigger evolution after 3 generations of stable fitness
MEMEVOLVE_AUTO_EVOLUTION_PLATEAU=3
```

### 4. Time-Based Trigger
**Environment Variable**: `MEMEVOLVE_AUTO_EVOLUTION_HOURS`  
**Default**: `24`  
**Purpose**: Periodic evolution every N hours

Ensures regular optimization regardless of other triggers, providing consistent improvement over time.

```bash
# Periodic evolution every 12 hours
MEMEVOLVE_AUTO_EVOLUTION_HOURS=12
```

## 🔧 Configuration

### Enabling Auto-Evolution

**Environment Variable**: `MEMEVOLVE_AUTO_EVOLUTION_ENABLED`  
**Default**: `true`  
**Purpose**: Master switch for auto-evolution system

```bash
# Disable auto-evolution (manual evolution only)
MEMEVOLVE_AUTO_EVOLUTION_ENABLED=false

# Enable auto-evolution (recommended)
MEMEVOLVE_AUTO_EVOLUTION_ENABLED=true
```

### Complete Auto-Evolution Configuration

```bash
# Enable intelligent auto-evolution
MEMEVOLVE_AUTO_EVOLUTION_ENABLED=true

# Multi-trigger configuration
MEMEVOLVE_AUTO_EVOLUTION_REQUESTS=500      # After 500 requests
MEMEVOLVE_AUTO_EVOLUTION_DEGRADATION=0.2     # 20% performance drop
MEMEVOLVE_AUTO_EVOLUTION_PLATEAU=5           # 5 generations stable
MEMEVOLVE_AUTO_EVOLUTION_HOURS=24            # Every 24 hours
```

## 📊 Performance Monitoring

### Key Metrics Tracked

The auto-evolution system monitors these performance indicators:

1. **Fitness Score**: Overall system performance (adaptive scoring with historical context)
2. **Response Time**: API response latency tracking
3. **Quality Score**: Response quality assessment
4. **Token Efficiency**: Token reduction and optimization
5. **Memory Utilization**: Storage and memory usage efficiency

### Business Impact Integration

Auto-evolution is integrated with comprehensive business analytics:

- **ROI Tracking**: Measures business value of evolution improvements
- **Statistical Significance**: Validates improvements with confidence intervals
- **Executive Reporting**: Provides business-centric impact analysis
- **Investment Recommendations**: Suggests optimization opportunities

## 🚀 Production Best Practices

### Recommended Settings

```bash
# Production configuration with balanced triggers
MEMEVOLVE_AUTO_EVOLUTION_ENABLED=true
MEMEVOLVE_AUTO_EVOLUTION_REQUESTS=500
MEMEVOLVE_AUTO_EVOLUTION_DEGRADATION=0.15
MEMEVOLVE_AUTO_EVOLUTION_PLATEAU=5
MEMEVOLVE_AUTO_EVOLUTION_HOURS=24
```

### High-Traffic Applications

```bash
# For high-traffic applications with frequent optimization
MEMEVOLVE_AUTO_EVOLUTION_ENABLED=true
MEMEVOLVE_AUTO_EVOLUTION_REQUESTS=1000
MEMEVOLVE_AUTO_EVOLUTION_DEGRADATION=0.1
MEMEVOLVE_AUTO_EVOLUTION_PLATEAU=3
MEMEVOLVE_AUTO_EVOLUTION_HOURS=12
```

### Resource-Constrained Environments

```bash
# For resource-constrained environments with conservative evolution
MEMEVOLVE_AUTO_EVOLUTION_ENABLED=true
MEMEVOLVE_AUTO_EVOLUTION_REQUESTS=2000
MEMEVOLVE_AUTO_EVOLUTION_DEGRADATION=0.25
MEMEVOLVE_AUTO_EVOLUTION_PLATEAU=10
MEMEVOLVE_AUTO_EVOLUTION_HOURS=48
```

## 🔄 Evolution Process

### Pre-Evolution Checks

Before triggering evolution, the system performs safety checks:

1. **Minimum Data Requirements**: Ensures sufficient experience data
2. **System Health Check**: Verifies all components are operational
3. **Resource Availability**: Confirms adequate system resources
4. **Circuit Breaker Status**: Prevents evolution during issues

### Evolution Execution

1. **Genotype Generation**: Creates new memory architecture variants
2. **Performance Evaluation**: Tests each variant with business impact analysis
3. **Selection**: Chooses best-performing architectures
4. **Deployment**: Safely applies improvements with rollback capability
5. **Validation**: Confirms improvements with statistical significance

### Post-Evolution Validation

After evolution completes:

1. **Performance Validation**: Confirms improvements meet expectations
2. **Business Impact Assessment**: Measures ROI and business value
3. **Rollback Preparation**: Maintains fallback configuration
4. **Monitoring**: Enhanced monitoring of new configuration

## 📈 Monitoring Auto-Evolution

### Dashboard Integration

Monitor auto-evolution through the executive dashboard:

- **Evolution History**: Track all evolution cycles and triggers
- **Performance Trends**: View performance improvements over time
- **Business Impact**: Monitor ROI and business value generation
- **Trigger Analysis**: See which triggers are most active

### Logging

Auto-evolution provides comprehensive logging:

```bash
# Monitor evolution events
tail -f logs/evolution.log

# Check trigger activations
grep "Auto-evolution triggered" logs/evolution.log

# View business impact
grep "Business impact" logs/evolution.log
```

## 🛠️ Troubleshooting

### Common Issues

#### Auto-Evolution Not Triggering

1. **Check Master Switch**: Ensure `MEMEVOLVE_AUTO_EVOLUTION_ENABLED=true`
2. **Verify Configuration**: Check trigger thresholds are reasonable
3. **Monitor Load**: Ensure sufficient API traffic for request-based triggers
4. **Check Logs**: Review evolution logs for error messages

#### Excessive Evolution

1. **Adjust Thresholds**: Increase trigger values to reduce frequency
2. **Monitor Performance**: Check for underlying performance issues
3. **Review Triggers**: Disable specific triggers if overactive
4. **Resource Check**: Ensure adequate system resources

#### Performance Degradation After Evolution

1. **Rollback Available**: System maintains previous configuration
2. **Monitor Logs**: Check evolution validation results
3. **Adjust Parameters**: Fine-tune evolution parameters
4. **Contact Support**: Use issue tracker for assistance

### Debug Commands

```bash
# Check auto-evolution status
curl http://localhost:11436/health | jq '.auto_evolution'

# View trigger history
curl http://localhost:11436/evolution-history

# Monitor current performance
curl http://localhost:11436/dashboard-data | jq '.performance'
```

## 🔗 Related Resources

- [Business Impact Analyzer](../tools/business-impact-analyzer.md) - Executive-level analytics
- [Performance Analyzer](../tools/performance_analyzer.md) - System monitoring
- [Configuration Guide](configuration.md) - Complete configuration options
- [Evolution System](../development/evolution.md) - Technical evolution details

---

*Last updated: January 25, 2026*