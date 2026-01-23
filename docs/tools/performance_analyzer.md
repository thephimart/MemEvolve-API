# Performance Analyzer

The MemEvolve Performance Analyzer is a comprehensive monitoring and analysis tool that generates detailed performance reports from system logs and data files without requiring external LLM services.

## Overview

The Performance Analyzer processes system telemetry to provide actionable insights into:
- **API Performance** - Request patterns, success rates, response times
- **Memory Operations** - Encoding performance, retrieval efficiency, storage utilization
- **Quality Metrics** - Response quality trends and evaluation performance
- **Evolution System** - Meta-optimization progress and genotype effectiveness
- **System Resources** - Log growth, storage usage, and resource efficiency

## Quick Start

```bash
# Analyze the last 24 hours
python scripts/performance_analyzer.py --days 1

# Analyze a specific date range
python scripts/performance_analyzer.py --start-date 2026-01-20 --end-date 2026-01-25

# Save report to custom location
python scripts/performance_analyzer.py --days 7 --output weekly_report.md
```

## Report Structure

Reports are automatically saved to `data/reports/` with timestamped filenames and include:

### Executive Summary
- Analysis time period and duration
- Key performance indicators (success rate, evolution status, memory size)
- Overall system health assessment

### Quality Analysis
- Number of quality evaluations performed
- Average quality score and performance rating
- Score distribution and range analysis

### API Performance
- Total API requests and success rates
- Response time analysis (when available)
- Memory encoding performance metrics

### Evolution System
- Evolution generations completed
- Current genotype configuration
- Fitness scores and optimization progress

### Memory System
- Total stored experiences and memory types
- Storage utilization and efficiency metrics
- Memory injection patterns and retrieval performance

### System Resources
- Log file sizes and growth rates
- Storage consumption and efficiency
- Resource utilization trends

### Key Insights
- Performance bottleneck identification
- Optimization recommendations
- System health assessment and actionable improvements

## Use Cases

### Daily Monitoring
```bash
# Daily performance check
python scripts/performance_analyzer.py --days 1
```

### Weekly Reporting
```bash
# Weekly performance analysis
python scripts/performance_analyzer.py --days 7 --output weekly_performance.md
```

### Incident Analysis
```bash
# Analyze specific problematic period
python scripts/performance_analyzer.py --start-date 2026-01-20 --end-date 2026-01-21
```

### Automated Reporting (Cron)
```bash
# Add to crontab for daily reports
0 6 * * * cd /path/to/MemEvolve-API && python scripts/performance_analyzer.py --days 1
```

## Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--days N` | Analyze last N days | `--days 7` |
| `--start-date YYYY-MM-DD` | Start date for analysis | `--start-date 2026-01-20` |
| `--end-date YYYY-MM-DD` | End date for analysis | `--end-date 2026-01-25` |
| `--output FILE` | Save report to specific file | `--output custom_report.md` |
| `--data-dir PATH` | Custom data directory | `--data-dir ./custom_data` |
| `--logs-dir PATH` | Custom logs directory | `--logs-dir ./custom_logs` |

## Data Sources

The analyzer processes data from:

- **API Server Logs** (`logs/api-server.log`) - HTTP requests, responses, timing
- **Middleware Logs** (`logs/middleware.log`) - Memory operations, quality scores, encoding times
- **Memory System** (`data/memory/memory_system.json`) - Experience storage and retrieval patterns
- **Evolution State** (`data/evolution_state.json`) - Meta-optimization progress and genotypes

## Report Output

Reports are saved as Markdown files with:
- **Clean formatting** for easy reading and sharing
- **Structured sections** for quick navigation
- **Actionable insights** rather than raw data dumps
- **Executive summaries** for management reporting
- **Technical details** for engineering analysis

## Dependencies

- **Python Standard Library only** - No external dependencies
- **System logs and data files** - Reads from local filesystem
- **No network access required** - Completely offline analysis

## Integration

### CI/CD Pipelines
```yaml
# Add to GitHub Actions or Jenkins
- name: Generate Performance Report
  run: python scripts/performance_analyzer.py --days 1 --output performance_report.md
```

### Monitoring Dashboards
```bash
# Generate reports for dashboard ingestion
python scripts/performance_analyzer.py --days 1 --output /var/reports/$(date +%Y%m%d).md
```

### Alerting Systems
```bash
# Check for performance regressions
SUCCESS_RATE=$(python scripts/performance_analyzer.py --days 1 | grep "Success Rate" | cut -d: -f2 | tr -d '%')
if [ "$SUCCESS_RATE" -lt 95 ]; then
    echo "Performance alert: Success rate dropped to ${SUCCESS_RATE}%"
fi
```

## Troubleshooting

### Common Issues

**"No data found in analysis period"**
- Check that the system has been running during the specified time range
- Verify log files exist in the expected locations
- Ensure date format is correct (YYYY-MM-DD)

**"Permission denied" on log files**
- Check file permissions on `logs/` directory
- Ensure the user running the script has read access

**"Memory system file not found"**
- Verify the system has processed requests and created memory data
- Check that `data/memory/memory_system.json` exists

### Log Format Requirements

The analyzer expects standard Python logging format:
```
2026-01-23 08:38:42,123 - logger_name - INFO - Log message content
```

## Examples

### Daily Operations Report
```bash
python scripts/performance_analyzer.py --days 1
# Generates: data/reports/performance_report_20260123_083842.md
```

### Weekly Performance Review
```bash
python scripts/performance_analyzer.py --days 7 --output weekly_performance.md
# Creates comprehensive weekly analysis
```

### Custom Analysis Window
```bash
python scripts/performance_analyzer.py \
  --start-date 2026-01-15 \
  --end-date 2026-01-22 \
  --output deployment_analysis.md
```

## Related Documentation

- [System Architecture](../development/architecture.md) - Understanding system components
- [Configuration Guide](../user-guide/configuration.md) - Performance tuning options
- [Evolution System](../development/evolution.md) - Meta-optimization details
- [Troubleshooting Guide](../api/troubleshooting.md) - Common issues and solutions

## Performance Impact

The analyzer has minimal performance impact:
- **CPU Usage**: Lightweight parsing and statistical analysis
- **Memory Usage**: Loads log files incrementally, processes in memory
- **Storage**: Generates small report files (~1-5KB)
- **Runtime**: Typically completes in seconds for daily analysis

## Future Enhancements

Planned improvements include:
- **Real-time monitoring** mode for continuous analysis
- **Performance regression detection** with historical comparison
- **Automated alerting** based on configurable thresholds
- **Dashboard integration** with visualization exports
- **Performance prediction** using trend analysis</content>
<parameter name="filePath">docs/tools/performance_analyzer.md