# Quality Scoring System

MemEvolve includes an independent, parity-based quality scoring system that provides unbiased evaluation of LLM responses across different model types and architectures.

## 🎯 Overview

The quality scoring system addresses a critical challenge in LLM evaluation: ensuring fair assessment regardless of whether a model uses direct responses or reasoning/thinking processes.

### Key Principles

1. **Parity-Based Evaluation**: Treats all models fairly regardless of reasoning capabilities
2. **Independent Assessment**: Separate from memory system to avoid bias
3. **Multi-Factor Analysis**: Considers content quality, structure, and insight
4. **Model-Agnostic**: Works with any OpenAI-compatible model
5. **Adaptive Learning**: Tracks performance patterns and adjusts automatically

## 🏗️ Architecture

### Core Components

```
ResponseQualityScorer
├── _calculate_response_quality()     # Main scoring logic
├── _evaluate_reasoning_response()    # For thinking models
├── _evaluate_direct_response()       # For standard models
├── _calculate_content_factors()      # Semantic density, structure, insight
├── _evaluate_reasoning_consistency() # Reasoning-answer alignment
└── _apply_bias_correction()         # Model-type bias adjustment
```

### Scoring Process

1. **Content Analysis**: Evaluate semantic density, logical structure, and insight
2. **Query Alignment**: Assess how well response addresses the user's question
3. **Model Type Detection**: Identify if response includes reasoning content
4. **Parity Application**: Apply appropriate scoring methodology:
   - **Direct responses**: 100% answer quality evaluation
   - **Reasoning responses**: 70% answer + 30% reasoning quality
5. **Bias Correction**: Adjust for systematic model-type biases
6. **Final Score**: Normalized score between 0.0 and 1.0

## 📊 Scoring Factors

### Content Quality Factors

#### Semantic Density
- **Definition**: Meaningful concepts per word
- **Purpose**: Encourages concise, information-rich responses
- **Calculation**: Concept-to-word ratio with uniqueness weighting

#### Logical Structure  
- **Definition**: Presence of step-by-step reasoning and examples
- **Purpose**: Rewards well-organized, educational responses
- **Indicators**: Numbered steps, bullet points, illustrative examples

#### Insight & Novelty
- **Definition**: Beyond generic responses, provides unique perspectives
- **Purpose**: Encourages creative, valuable contributions
- **Detection**: Comparison against common response patterns

#### Query Alignment
- **Definition**: Direct addressing of question aspects
- **Purpose**: Ensures relevance and completeness
- **Scoring**: Coverage of question components, accuracy

### Reasoning-Specific Factors

#### Reasoning Quality
- **Step-by-step logic**: Coherent thought process
- **Error handling**: Identification and correction of mistakes
- **Self-correction**: Acknowledgment and revision of initial thoughts

#### Reasoning-Answer Consistency
- **Alignment**: How well reasoning leads to final answer
- **Contradiction detection**: Identifying logical inconsistencies
- **Support**: Whether reasoning supports the conclusion

## 🔄 Bias Correction System

### Problem Statement
Without bias correction, models without reasoning capabilities often receive lower scores simply because they're evaluated on different criteria than thinking models.

### Solution: Adaptive Bias Tracking

```python
# Track performance by model type
bias_tracker = {
    "reasoning_models": {
        "scores": [],
        "avg_score": 0.0,
        "count": 0
    },
    "direct_models": {
        "scores": [], 
        "avg_score": 0.0,
        "count": 0
    }
}

# Apply bias correction
if model_type == "reasoning":
    bias_adjustment = -bias_adjustment_factor
elif model_type == "direct":
    bias_adjustment = +bias_adjustment_factor
```

### Bias Detection Mechanisms

1. **Performance Tracking**: Monitor average scores by model type
2. **Statistical Analysis**: Identify significant performance gaps
3. **Automatic Adjustment**: Apply corrections when bias is detected
4. **Continuous Learning**: System adapts as more data is collected

## 📈 Score Interpretation

### Score Ranges

| Score Range | Quality Level | Characteristics |
|--------------|---------------|------------------|
| **0.8 - 1.0** | Excellent | Insightful, well-structured, novel, perfectly aligned |
| **0.6 - 0.8** | Good | Clear, relevant, some insight, decent structure |
| **0.4 - 0.6** | Average | Basic relevance, minimal structure, generic content |
| **0.2 - 0.4** | Poor | Partial relevance, poor structure, generic |
| **0.0 - 0.2** | Very Poor | Irrelevant, incoherent, incorrect |

### Thinking Model Scores

For models with reasoning content, scores are distributed:
- **70%** based on final answer quality
- **30%** based on reasoning process quality

This ensures:
- **Fair Competition**: Direct models aren't penalized for lacking reasoning
- **Reasoning Value**: High-quality reasoning is properly rewarded
- **Answer Focus**: Final response quality remains primary

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `MEMEVOLVE_LOG_MIDDLEWARE_ENABLE` | Enable detailed scoring logs | `false` | Set to `true` for debugging |
| `MEMEVOLVE_QUALITY_BIAS_CORRECTION` | Enable bias correction | `true` | Disable for raw scores |
| `MEMEVOLVE_QUALITY_MIN_THRESHOLD` | Minimum score for experience storage | `0.1` | Filter low-quality responses |

### Score Weighting

```python
# Reasoning model weights
reasoning_weights = {
    "answer_quality": 0.7,
    "reasoning_quality": 0.3
}

# Direct model weights  
direct_weights = {
    "answer_quality": 1.0
}
```

## 📋 Usage Examples

### Basic Quality Scoring

```python
from utils.quality_scorer import ResponseQualityScorer

scorer = ResponseQualityScorer(debug=True)

# Score a direct response
response = {
    "role": "assistant", 
    "content": "Water appears wet due to surface tension..."
}
context = {
    "original_query": "Why does water feel wet?",
    "messages": [{"role": "user", "content": "Why does water feel wet?"}]
}

score = scorer.calculate_response_quality(response, context, "Why does water feel wet?")
print(f"Quality score: {score:.3f}")
```

### Scoring Reasoning Content

```python
# Score a thinking model response
reasoning_response = {
    "role": "assistant",
    "content": "Water feels wet due to surface tension...",
    "reasoning_content": "First, consider what 'wet' means..."
}

score = scorer.calculate_response_quality(
    reasoning_response, context, "Why does water feel wet?"
)
print(f"Reasoning model score: {score:.3f}")
```

### Monitoring Quality Trends

```python
# Track quality over time
import json
from datetime import datetime

quality_log = []
for response_batch in responses:
    scores = []
    for response in response_batch:
        score = scorer.calculate_response_quality(response, context, query)
        scores.append(score)
    
    quality_log.append({
        "timestamp": datetime.now().isoformat(),
        "avg_score": sum(scores) / len(scores),
        "count": len(scores),
        "has_reasoning": any(r.get("reasoning_content") for r in response_batch)
    })

# Save for analysis
with open("quality_trends.json", "w") as f:
    json.dump(quality_log, f, indent=2)
```

## 🐛 Troubleshooting

### Common Issues

#### "All scores are the same"
**Cause**: Likely missing query context or response content
**Solution**: Ensure both `content` and original query are provided

```python
# ❌ Missing context
score = scorer.calculate_response_quality(response, {}, query)

# ✅ Proper context
context = {"original_query": query, "messages": [{"role": "user", "content": query}]}
score = scorer.calculate_response_quality(response, context, query)
```

#### "Reasoning models score lower than expected"
**Cause**: Bias correction may need adjustment or insufficient data
**Solution**: Monitor bias tracking and allow system to learn

```bash
# Enable bias correction logging
export MEMEVOLVE_LOG_MIDDLEWARE_ENABLE=true
export MEMEVOLVE_QUALITY_BIAS_CORRECTION=true

# Check bias tracking in logs
grep "bias correction" logs/api-server.log
```

#### "Scores seem too high/low"
**Cause**: May need threshold adjustment for your specific use case
**Solution**: Adjust minimum threshold or weighting factors

```python
# Custom scorer with adjusted thresholds
custom_scorer = ResponseQualityScorer(
    debug=True,
    min_threshold=0.2,  # Higher threshold
    bias_correction=False  # Disable if bias not needed
)
```

### Debug Mode

Enable detailed scoring analysis:

```bash
export MEMEVOLVE_LOG_MIDDLEWARE_ENABLE=true
python scripts/start_api.py
```

Debug output includes:
- Content factor breakdown
- Reasoning evaluation details  
- Bias correction calculations
- Final score composition

## 📊 Performance Impact

### Computational Overhead
- **Direct responses**: ~5-10ms additional processing
- **Reasoning responses**: ~15-25ms due to consistency analysis
- **Bias correction**: Minimal overhead (<1ms)

### Memory Usage
- **Bias tracking**: ~100KB for performance history
- **Scoring cache**: Optional, ~1MB for recent scores
- **Total impact**: Negligible for typical deployments

## 🔮 Advanced Features

### Custom Quality Factors

Extend scoring with custom evaluation criteria:

```python
class CustomQualityScorer(ResponseQualityScorer):
    def _calculate_custom_factors(self, content, query):
        # Add your custom scoring logic
        technical_accuracy = self._evaluate_technical_accuracy(content)
        code_quality = self._evaluate_code_quality(content)
        
        return {
            "technical_accuracy": technical_accuracy,
            "code_quality": code_quality
        }

    def _evaluate_technical_accuracy(self, content):
        # Implementation-specific logic
        return score
```

### Integration with Evolution System

Quality scores feed directly into the evolution framework:

```python
# Quality scores influence fitness evaluation
fitness_calculator = FitnessCalculator(
    quality_weight=0.6,
    performance_weight=0.4
)

# Evolution prioritizes high-quality responses
best_genotypes = evolution_manager.select_top_performers(
    quality_scores=quality_history,
    performance_metrics=timing_data
)
```

## 📚 Related Documentation

- [API Reference](api-reference.md) - Complete API endpoints
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [Architecture Overview](../development/architecture.md) - System design
- [Getting Started](../user-guide/getting-started.md) - Quick setup guide

---

*Last updated: January 24, 2026*