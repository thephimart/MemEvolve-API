# MemEvolve-API Performance Analyzer

A comprehensive performance analysis tool for MemEvolve-API that generates detailed reports from system logs and data files without requiring LLM endpoints.

## Overview

The Performance Analyzer processes:
- **API server logs** - Request patterns, success rates, response times
- **Middleware logs** - Memory operations, quality scores, encoding times
- **Evolution state** - System metrics, genotype performance
- **Memory system data** - Experience counts, content analysis

## Usage

### Basic Usage

```bash
# Analyze last 7 days
python scripts/performance_analyzer.py --days 7

# Analyze specific date range
python scripts/performance_analyzer.py --start-date 2026-01-23 --end-date 2026-01-23

# Save report to file
python scripts/performance_analyzer.py --days 1 --output performance_report.md
```

### Command Line Options

```
--start-date      Start date for analysis (YYYY-MM-DD format)
--end-date        End date for analysis (YYYY-MM-DD format)
--days            Analyze last N days
--output, -o      Output file path (default: stdout)
--data-dir        Data directory path (default: ./data)
--logs-dir        Logs directory path (default: ./logs)
```

## Report Sections

### ⚡ API Performance Metrics
- **Request Statistics**: Total requests, success rates, status codes
- **Response Time Analysis**: Average, min, max response times
- **Memory Operations**: Experience creation counts, encoding performance
- **Quality Analysis**: Quality score distributions and trends

### 🔄 Evolution System Analysis
- **Evolution Cycles**: Completed generations, population size
- **Genotype Performance**: Current genotype, fitness scores
- **Timing Metrics**: Average response and retrieval times

### 🧠 Memory System Analysis
- **Experience Metrics**: Total experiences, memory types, file sizes
- **Content Analysis**: Content length statistics
- **System Health**: Memory utilization and growth patterns

### 🎯 Recommendations
- **Performance Insights**: Actionable recommendations based on analysis
- **Optimization Opportunities**: Identified bottlenecks and improvement areas

## Dependencies

- **Python Standard Library**: `json`, `re`, `statistics`, `datetime`, `pathlib`
- **No external dependencies** required

## Output Format

Reports are generated in Markdown format with:
- **Structured sections** for easy reading
- **Statistical summaries** with averages, ranges, and counts
- **Performance insights** and actionable recommendations
- **Timestamp information** for analysis period

## Integration

### Automated Reporting

```bash
# Daily performance report
0 2 * * * cd /path/to/MemEvolve-API && python scripts/performance_analyzer.py --days 1 --output reports/daily_$(date +\%Y\%m\%d).md

# Weekly summary
0 3 * * 1 cd /path/to/MemEvolve-API && python scripts/performance_analyzer.py --days 7 --output reports/weekly_$(date +\%Y\%m\%d).md
```

### CI/CD Integration

```bash
# Performance regression testing
python scripts/performance_analyzer.py --days 1 | grep "Average Response Time" | awk '{if ($4 > 150) exit 1}'
```

## Analysis Capabilities

### ✅ What It Analyzes

- **API Request Patterns**: Success rates, status codes, timing
- **Memory Operations**: Encoding performance, injection counts
- **Quality Metrics**: Score distributions, evaluation trends
- **Evolution Progress**: Genotype performance, fitness scores
- **Memory Growth**: Experience accumulation, content statistics
- **System Health**: Error rates, resource utilization

### ❌ Limitations

- **No Real-time Monitoring**: Analyzes historical log data only
- **Log-dependent**: Requires properly formatted log files
- **No External APIs**: Cannot test live system endpoints
- **File-based**: Depends on local log and data file access

## Example Output

```markdown
# MemEvolve-API Performance Analysis

**Analysis Period**: 2026-01-23 to 2026-01-23
**Generated**: 2026-01-23 08:38:42

## ⚡ API Performance Metrics

### Request Statistics:
- Total Requests: 200
- Successful Requests: 200
- Success Rate: 100.00%

### Response Time Analysis:
- Average Response Time: 141.11 seconds
- Min Response Time: 85.23 seconds
- Max Response Time: 203.45 seconds

### Memory Encoding Performance:
- Average Encoding Time: 10.39 seconds
- Min Encoding Time: 7.33 seconds
- Max Encoding Time: 29.15 seconds
- Encoding Operations: 200

## 🎯 Quality Analysis
- Quality Evaluations: 200
- Average Quality Score: 0.546
- Quality Score Range: 0.405 - 0.707

## 🔄 Evolution System Analysis
- Evolution Cycles Completed: 20
- Fitness Score: 0.707207
- Average Response Time: 141.11s
- Average Retrieval Time: 0.0413s

## 🧠 Memory System Analysis
- Total Experiences: 200
- File Size: 144.9 KB
- Average Content Length: 725.3 characters
```

## Troubleshooting

### Common Issues

**"API server log not found"**
- Ensure logs are being written to the expected directory
- Check log file permissions and paths

**"No quality scores found"**
- Quality evaluation may not be enabled in middleware
- Check middleware configuration

**"Evolution state file not found"**
- Evolution may not be enabled or initialized
- Check evolution system configuration

### Log Format Requirements

The analyzer expects logs in standard Python logging format:
```
2026-01-23 08:38:42,123 - logger_name - INFO - Log message content
```

## Contributing

To extend the analyzer:

1. **Add new analysis methods** in the `PerformanceAnalyzer` class
2. **Update report generation** to include new metrics
3. **Add command-line options** for new analysis parameters
4. **Update documentation** with new capabilities

## Related Files

- `JAN_23_200_RUNS_RESULTS.md` - Example comprehensive analysis report
- `logs/` - Log files analyzed by the script
- `data/` - Data files analyzed by the script
- `scripts/` - Other management scripts</content>
<parameter name="filePath">scripts/README_performance_analyzer.md