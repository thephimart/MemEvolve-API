# Quality Scoring User Guide

This guide helps users understand and work with MemEvolve's quality scoring system to get the most out of their memory-enhanced LLM applications.

## 🎯 What is Quality Scoring?

Quality scoring is MemEvolve's built-in system that evaluates how good your LLM responses are. It helps ensure that:

- **All models are evaluated fairly** regardless of whether they use reasoning or not
- **High-quality responses are stored** in memory for future reference
- **Poor responses are filtered out** to maintain memory quality
- **System performance improves** over time through learning

## 🔄 How It Works

### For Regular Models
If your model gives direct answers (no reasoning):
- **100% score based on answer quality**
- Factors: relevance, clarity, accuracy, completeness

### For Thinking Models  
If your model uses reasoning (like "thinking" processes):
- **70% based on answer quality** + **30% based on reasoning quality**
- Ensures fair comparison with regular models
- Rewards good step-by-step thinking

### Score Ranges
| Score | Quality | What it means |
|--------|---------|---------------|
| 0.8-1.0 | Excellent | Outstanding response, very helpful |
| 0.6-0.8 | Good | Solid answer, worth remembering |
| 0.4-0.6 | Average | Basic response, may be useful |
| 0.2-0.4 | Poor | Limited value, likely ignored |
| 0.0-0.2 | Very Poor | Not worth storing |

## 🚀 Quick Start

### Basic Setup
Quality scoring works automatically - no setup required! Just use MemEvolve normally:

```bash
# Start MemEvolve with quality scoring enabled
python scripts/start_api.py

# Make requests as usual
curl -X POST http://localhost:11436/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "your-model", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Enable Detailed Logging
To see quality scores in action:

```bash
# Enable quality scoring logs
export MEMEVOLVE_LOG_MIDDLEWARE_ENABLE=true

# Start server
python scripts/start_api.py

# Watch the logs
tail -f logs/api-server.log | grep "quality scoring"
```

You'll see output like:
```
Quality scoring: base=0.342, final=0.375, normalized=0.375, has_reasoning=False
Independent quality scoring: score=0.375, has_reasoning=False
Recorded response quality score: 0.375
```

## ⚙️ Configuration Options

### Basic Settings

```bash
# Set minimum quality threshold (only scores >= 0.2 get stored)
export MEMEVOLVE_QUALITY_MIN_THRESHOLD=0.2

# Enable/disable bias correction (usually leave enabled)
export MEMEVOLVE_QUALITY_BIAS_CORRECTION=true

# Enable detailed logging for debugging
export MEMEVOLVE_LOG_MIDDLEWARE_ENABLE=true
```

### Advanced Configuration
For custom quality scoring needs, you can adjust how the system works:

```python
# Create custom scorer (advanced usage)
from utils.quality_scorer import ResponseQualityScorer

scorer = ResponseQualityScorer(
    debug=True,              # Show detailed scoring analysis
    min_threshold=0.15,     # Custom minimum threshold
    bias_correction=False     # Disable if you want raw scores
)
```

## 📊 Monitoring Quality

### Via Dashboard
Visit http://localhost:11436/dashboard to see:
- Average quality scores over time
- Distribution of score ranges (excellent, good, etc.)
- Performance of reasoning vs direct models

### Via API
```bash
# Get quality metrics
curl http://localhost:11436/quality/metrics
```

Response:
```json
{
  "total_responses": 150,
  "average_quality_score": 0.342,
  "reasoning_model_count": 45,
  "direct_model_count": 105,
  "score_distribution": {
    "excellent": 12,
    "good": 38,
    "average": 67,
    "poor": 28,
    "very_poor": 5
  }
}
```

### From Logs
```bash
# Extract quality scores from logs
grep "Recorded response quality score" logs/api-server.log | \
  awk '{print $NF}' | \
  sort -n | \
  uniq -c
```

## 🎛️ Common Use Cases

### Use Case 1: Ensuring High-Quality Memory
```bash
# Set a higher threshold to only store excellent responses
export MEMEVOLVE_QUALITY_MIN_THRESHOLD=0.6

# This ensures only scores >= 0.6 (Good or Excellent) get stored
```

### Use Case 2: Debugging Low Scores
```bash
# Enable debug mode to see why scores are low
export MEMEVOLVE_LOG_MIDDLEWARE_ENABLE=true

# Look for these patterns in logs:
# - "Semantic density" - How information-dense the response is
# - "Query alignment" - How well it answers the question
# - "Insight score" - Whether it provides unique value
```

### Use Case 3: Comparing Model Performance
```bash
# Use the same prompts with different models
# Quality scoring ensures fair comparison regardless of reasoning

# Model A (direct response)
curl -X POST http://localhost:11436/v1/chat/completions \
  -d '{"model": "model-a", "messages": [...]}'

# Model B (thinking model)  
curl -X POST http://localhost:11436/v1/chat/completions \
  -d '{"model": "model-b", "messages": [...]}'

# Compare scores in logs - they're fairly weighted!
```

### Use Case 4: Filtering Content for Production
```bash
# Only store responses above production quality threshold
export MEMEVOLVE_QUALITY_MIN_THRESHOLD=0.7

# This acts as a quality gate for your memory system
```

## 🔧 Troubleshooting Quality Issues

### "All my scores are low!"
**Common causes:**
- Responses are too generic or repetitive
- Missing key information the user asked for
- Poor structure or unclear explanations

**Solutions:**
```bash
# Check what's being scored
export MEMEVOLVE_LOG_MIDDLEWARE_ENABLE=true

# Look for these factors in logs:
grep "semantic_density\|query_alignment\|insight" logs/api-server.log
```

### "Thinking models score worse than direct models"
**This shouldn't happen with bias correction enabled!**

**Solution:**
```bash
# Ensure bias correction is on
export MEMEVOLVE_QUALITY_BIAS_CORRECTION=true

# Give it time - bias correction learns from 20-50 responses
# Monitor bias tracking:
grep "bias correction" logs/api-server.log
```

### "Scores seem inconsistent"
**Check your prompts and model:**
```bash
# Test with consistent prompts
# Quality scoring rewards specific, well-structured answers

# Example good prompt:
"How do I implement Python caching with examples?"

# Example poor prompt:
"tell me about caching"
```

## 📈 Improving Your Quality Scores

### For Better Model Responses
1. **Be specific in prompts**: More detail → better responses
2. **Request structure**: Ask for numbered lists, examples, steps
3. **Define quality**: "Provide a comprehensive answer with examples"

### For Better System Performance
1. **Monitor trends**: Watch average scores over time
2. **Adjust thresholds**: Set appropriate minimums for your use case
3. **Use debug mode**: Understand why responses score certain ways

### Example Prompt Improvements
```bash
# ❌ Basic prompt
curl -X POST http://localhost:11436/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Explain databases"}]}'
# Score: ~0.3 (generic, basic)

# ✅ Structured prompt  
curl -X POST http://localhost:11436/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Explain databases with 3 key points, examples, and best practices"}]}'
# Score: ~0.6 (structured, comprehensive)
```

## 🎛️ Advanced Features

### Custom Quality Factors (Developers)
If you have specific quality needs:

```python
class CustomQualityScorer(ResponseQualityScorer):
    def _calculate_custom_factors(self, content, query):
        # Add your own scoring criteria
        technical_accuracy = self._check_technical_terms(content)
        code_quality = self._analyze_code_snippets(content)
        
        return {
            "technical": technical_accuracy,
            "code": code_quality
        }
```

### Integration with Monitoring
```python
# Export quality data for analysis
import json
from datetime import datetime

quality_data = {
    "timestamp": datetime.now().isoformat(),
    "avg_score": 0.342,
    "response_count": 45,
    "threshold": 0.2
}

with open("quality_report.json", "w") as f:
    json.dump(quality_data, f, indent=2)
```

## 📚 Quick Reference

### Environment Variables Cheat Sheet
| Variable | Default | When to Change |
|----------|---------|----------------|
| `MEMEVOLVE_QUALITY_MIN_THRESHOLD` | 0.1 | Want higher quality memory |
| `MEMEVOLVE_QUALITY_BIAS_CORRECTION` | true | Only disable for testing |
| `MEMEVOLVE_LOG_MIDDLEWARE_ENABLE` | false | Enable for debugging |

### Score Interpretation Quick Guide
- **0.7+**: Store immediately, excellent quality
- **0.5-0.7**: Good quality, worth storing  
- **0.3-0.5**: Average, store if relevant
- **0.1-0.3**: Poor, usually filtered out
- **<0.1**: Very poor, never stored

### Common Log Patterns
```bash
# Quality scoring working correctly
"Independent quality scoring: score=0.375, has_reasoning=False"

# Bias correction active  
"Bias correction applied: +0.05 for direct model"

# Threshold filtering
"Response quality 0.15 below threshold 0.2 - not stored"
```

## 🤝 Getting Help

### Check Your Setup
```bash
# Verify quality scoring is working
curl http://localhost:11436/quality/metrics

# Check current configuration
env | grep MEMEVOLVE_QUALITY
```

### Common Issues & Solutions
- **Scores all the same**: Enable debug logging to check context
- **Memory not growing**: Check if threshold is too high
- **Performance slow**: Quality scoring adds minimal overhead (<2%)

### More Resources
- [Technical Documentation](../api/quality-scoring.md) - Deep technical details
- [API Reference](../api/api-reference.md) - Complete endpoint documentation  
- [Troubleshooting Guide](../api/troubleshooting.md) - Common issues and solutions
- [Community Support](https://github.com/thephimart/MemEvolve-API/issues) - Get help from the community

---

*Last updated: January 24, 2026*