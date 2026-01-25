# MemEvolve: Evolution System

## Overview

The MemEvolve evolution system implements intelligent auto-evolution of memory architectures, enabling automatic discovery and optimization of memory system configurations through genetic algorithms, multi-trigger evolution, and comprehensive business impact validation.

## Core Concept: Meta-Evolution

Unlike traditional memory systems that learn from experiences, MemEvolve's evolution system learns **how to learn** by optimizing the memory architecture itself.

### Intelligent Auto-Evolution Loop
1. **Population**: Multiple memory architectures (genotypes) compete
2. **Multi-Trigger Evaluation**: Evolution initiated by requests, performance, plateau, or time triggers
3. **Business Impact Assessment**: Each genotype evaluated for ROI and business value
4. **Adaptive Selection**: Historical context-based performance evaluation
5. **Mutation**: Selected genotypes are modified to create new variants
6. **Continuous Optimization**: Process repeats, discovering progressively better architectures with measurable business value

### Auto-Evolution Triggers
- **Request Count Trigger**: Evolution after N requests (default: 500)
- **Performance Degradation Trigger**: Evolution when performance drops by X% (default: 20%)
- **Fitness Plateau Trigger**: Evolution when fitness stable for N generations (default: 5)
- **Time-Based Trigger**: Periodic evolution every N hours (default: 24)

## Architecture Components

### Genotype Representation
Each memory system is encoded as a genotype with four components:

```python
@dataclass
class MemoryGenotype:
    encode: EncodeConfig      # Encoding strategies and parameters
    store: StoreConfig        # Storage backend and settings
    retrieve: RetrieveConfig  # Retrieval strategies and parameters
    manage: ManageConfig      # Management policies and settings
```

### Reference Architectures

| Architecture | Key Features | Use Case |
|-------------|---------------|----------|
| **AgentKB** | Static baseline, lesson-based, minimal overhead | Simple applications, low resource |
| **Lightweight** | Trajectory-based, JSON storage, auto-pruning | Fast, memory-constrained environments |
| **Riva** | Agent-centric, vector storage, hybrid retrieval | General-purpose, balanced performance |
| **Cerebra** | Tool distillation, semantic graphs, advanced caching | Complex reasoning, tool-heavy tasks |

## Evolution Algorithm

### Core Parameters
Evolution behavior is controlled by configurable parameters:

- **Population Size** (default: 10): Number of genotypes evaluated per generation
  - Larger populations = Better optimization, higher computational cost
  - Smaller populations = Faster evolution, may miss optimal solutions

- **Generations** (default: 20): Evolution cycles before convergence assessment
  - More generations = More thorough search, longer optimization time
  - Fewer generations = Faster convergence, potentially suboptimal results

- **Evolution Cycle Rate** (default: 60 seconds): Time between evolution cycles
  - Lower values: More frequent optimization, faster adaptation to changing patterns
  - Higher values: Less frequent optimization, more stable performance
  - Configurable via `MEMEVOLVE_EVOLUTION_CYCLE_SECONDS` environment variable

- **Mutation Rate** (default: 0.1): Probability of parameter changes (0.0-1.0)
  - Higher rates = More exploration, more variability
  - Lower rates = More exploitation, slower adaptation

- **Crossover Rate** (default: 0.5): Genetic recombination probability (0.0-1.0)
  - Higher rates = More genetic mixing, faster convergence
  - Lower rates = More independent evolution, diverse solutions

- **Selection Method** (default: pareto): Multi-objective optimization strategy
  - Pareto: Balances multiple competing objectives simultaneously

- **Tournament Size** (default: 3): Selection pressure for tournament selection
  - Larger tournaments = Stronger selection pressure, faster convergence

### Selection Strategy
**Pareto-based multi-objective optimization** balancing:
- **Performance**: Response quality, task completion rate
- **Cost**: Memory usage, computational overhead
- **Accuracy**: Retrieval precision and recall
- **Speed**: Response latency and throughput

### Mutation Strategy
Constrained mutations respecting model capabilities:
- **Encoding**: Strategy combinations, embedding dimensions, temperature, max_tokens
- **Storage**: Index types, backend parameters (when applicable)
- **Retrieval**: Weight combinations, threshold adjustments
- **Management**: Pruning policies, decay rates, capacity limits
- **Storage**: Backend type, indexing parameters
- **Retrieval**: Strategy weights, similarity thresholds
- **Management**: Pruning policies, consolidation settings

### Fitness Evaluation
Rolling window evaluation with multiple metrics:
- **Response Quality Score**: Semantic coherence (0-1)
- **Retrieval Precision**: Accuracy of memory retrieval
- **Retrieval Recall**: Completeness of memory retrieval
- **Response Time**: Latency impact
- **Memory Utilization**: Storage efficiency

## Embedding Configuration Evolution

### Dimension Evolution
**Embedding dimensions ARE evolved** within model capabilities for optimal performance:
- **Dynamic optimization**: Dimensions adjust based on task requirements and performance
- **Capability constraints**: Respects embedding model's supported dimension ranges
- **Performance balancing**: Trades off semantic quality vs computational cost/speed

### What IS Evolved
- `embedding_dim`: Vector dimensionality for semantic representations
- `max_tokens`: Context window size for encoding operations
- **Constraints**: Both must be ≤ model's actual capabilities
- **Optimization**: Balances encoding quality vs computational cost

### Dimension Change Process
When evolution selects a genotype with different dimensions:
1. **New dimension saved** to `evolution_state.json`
2. **Index rebuild triggered** on next API request
3. **All memories re-embedded** with new dimensions (expensive operation)
4. **Service continues** with optimized dimensionality

### Change Frequency
- **Early optimization** (first 1000-5000 requests): **Frequent changes** (5-20 per hour)
  - Aggressive exploration of dimension space
  - Rapid iteration to find promising ranges
- **Mid optimization** (5000-50000 requests): **Moderate changes** (2-5 per day)
  - Fine-tuning within effective dimension bands
  - Balancing quality vs performance
- **Late optimization** (50000+ requests): **Rare changes** (0-2 per week)
  - Convergence on optimal dimensions
  - Only changes for significant performance gains
- **Stable production**: **Minimal changes** after convergence

### Priority Hierarchy
1. **Evolution State** (highest) - Optimized values from evolution cycles
2. **Environment Variables** - Manual configuration override
3. **Auto-detection** - From `/models` API endpoint metadata
4. **Fallback Defaults** - Safe defaults (768 dimensions, 512 tokens)

## Implementation Details

### EvolutionManager Class
Central orchestrator handling:
- **Population Management**: Creating and tracking genotypes
- **Fitness Evaluation**: Performance measurement and scoring
- **Evolution Cycles**: Selection, mutation, and replacement
- **State Persistence**: Saving/loading evolution progress
- **Safety Controls**: Rollback mechanisms and constraints

### Component Hot-Swapping
Dynamic reconfiguration without restart:
- **Safe Transitions**: Validate compatibility before switching
- **State Preservation**: Maintain memory continuity during changes
- **Rollback Capability**: Revert to previous genotype on failure

### Evolution State Persistence

Evolution state is automatically persisted to `data/evolution_state.json` with robust protection against corruption:

#### Atomic Writes
- Saves to temporary file first, then atomically replaces main file
- Prevents corruption from interrupted writes (force kills, crashes)

#### Automatic Backups
- Creates timestamped backups in `data/evolution_backups/` before each save
- Keeps last 3 backups automatically
- Enables recovery from corrupted main files

#### Smart Recovery
- On startup, tries main file first
- Automatically falls back to most recent backup if main file is corrupted
- Logs detailed recovery information
- Moves corrupted files to `.corrupted` extension for debugging

#### File Structure
```json
{
  "best_genotype": { /* optimized memory architecture */ },
  "evolution_embedding_max_tokens": 1024,
  "evolution_history": [
    {
      "generation": 1,
      "best_genotype": {/* genotype details */},
      "fitness_score": 0.85,
      "improvement": 0.12,
      "timestamp": 1705960000.0
    }
  ],
  "metrics": {
    "api_requests_total": 1250,
    "average_response_time": 0.234,
    "evolution_cycles_completed": 15
  }
}
```

## Evolution Constraints

### Model Capability Constraints
- **Embedding Dimension**: Fixed by model (cannot be evolved)
- **Context Window**: Cannot exceed model's max_tokens
- **Memory Limits**: Respect available RAM/disk space

### Safety Constraints
- **Minimum Performance**: Prevent catastrophic degradation
- **Gradual Rollout**: Blend old/new configurations during transition
- **Circuit Breakers**: Auto-rollback on performance drops

## Current Status

### ✅ Completed
- **Genotype Representation**: Complete memory architecture encoding
- **Evolution Algorithms**: Pareto selection, constrained mutation
- **Fitness Evaluation**: Rolling window metrics collection
- **Component Hot-Swapping**: Dynamic reconfiguration framework
- **Embedding Evolution**: max_tokens optimization with constraints
- **State Persistence**: Robust persistence with atomic writes, backups, and corruption recovery

### 🔄 In Testing Phase
- **Evolution Cycles**: Running with fitness evaluation
- **Performance Monitoring**: Real-time metrics collection
- **Safety Mechanisms**: Circuit breakers and rollback testing

### 🚧 Pending
- **Safe Evolution Cycles**: Production-grade staged rollout
- **Advanced Evolution Features**: Multi-objective optimization, transfer learning
- **Empirical Validation**: Benchmark performance testing

## Usage

### Enabling Evolution
```bash
# Enable evolution in environment
MEMEVOLVE_ENABLE_EVOLUTION=true

# Configure evolution parameters
MEMEVOLVE_EVOLUTION_POPULATION_SIZE=10
MEMEVOLVE_EVOLUTION_MUTATION_RATE=0.1
MEMEVOLVE_EVOLUTION_GENERATIONS=20
MEMEVOLVE_EVOLUTION_CYCLE_SECONDS=60  # Seconds between evolution cycles (default: 60)
```

### Monitoring Evolution
```bash
# Check current evolution status
curl http://localhost:11436/evolution/status

# View evolution metrics
curl http://localhost:11436/evolution/metrics

# Force evolution cycle (development only)
curl -X POST http://localhost:11436/evolution/cycle
```

## Evolution State File Format

The evolution system maintains persistent state in `data/evolution_state.json` and creates automatic backups in `data/evolution_backups/`:

```json
{
  "current_genotype": {
    "encode": {
      "encoding_strategies": ["lesson", "skill"],
      "max_tokens": 1024,
      "temperature": 0.7,
      "batch_size": 10,
      "enable_abstractions": true,
      "min_abstraction_units": 3
    },
    "store": {
      "backend_type": "vector",
      "storage_path": "./data/memory",
      "index_type": "flat"
    },
    "retrieve": {
      "strategy_type": "hybrid",
      "default_top_k": 5,
      "semantic_weight": 0.7,
      "keyword_weight": 0.3
    },
    "manage": {
      "strategy_type": "simple",
      "enable_auto_management": true,
      "auto_prune_threshold": 1000
    }
  },
  "evolution_embedding_max_tokens": 1024,
  "evolution_history": [
    {
      "generation": 1,
      "best_genotype_id": "abc123",
      "fitness_score": 0.85,
      "improvement": 0.12
    }
  ],
  "metrics": {
    "total_requests": 1250,
    "avg_response_time": 0.234,
    "memory_utilization": 0.67
  }
}
```

## Safety and Production Readiness

### Safeguards Implemented
- **Shadow Mode Testing**: New genotypes tested before production use
- **Performance Monitoring**: Real-time degradation detection
- **Gradual Adoption**: Weighted blending during transitions
- **Automatic Rollback**: Performance threshold triggers

### Monitoring and Alerts
- **Evolution Cycle Success/Failure**: Logged and tracked
- **Performance Degradation Alerts**: Configurable thresholds
- **Resource Usage Monitoring**: Memory and CPU tracking

## Future Enhancements

### Advanced Evolution Features
- **Multi-Objective Optimization**: Beyond Pareto front
- **Adaptive Evolution**: Dynamic parameters based on load
- **Transfer Learning**: Apply successful genotypes across domains
- **Ensemble Methods**: Combine multiple high-performing genotypes

### Production Features
- **Evolution Scheduling**: Time-based or load-triggered cycles
- **A/B Testing Framework**: Statistical significance testing
- **Evolution Analytics**: Performance trend analysis
- **Custom Fitness Functions**: Domain-specific optimization

## Troubleshooting Evolution

### Common Issues

**Evolution not improving performance:**
- Check fitness evaluation metrics are meaningful
- Verify genotype diversity in population
- Ensure sufficient evaluation time per genotype

**Evolution causing instability:**
- Reduce mutation rate or population size
- Enable shadow mode testing
- Check for resource constraints

**Evolution state corruption:**
- System automatically recovers from backups on startup
- Corrupted files are moved to `.corrupted` extension for debugging
- Use `cleanup_evolution.sh` to reset if needed (removes `data/evolution_state.json`)
- Force-killing the server is now safe (atomic writes prevent corruption)

### Debug Commands
```bash
# View detailed evolution logs
MEMEVOLVE_LOG_EVOLUTION_ENABLE=true
tail -f logs/evolution.log

# Check evolution state and backups
ls -la data/evolution_state.json
ls -la data/evolution_backups/

# Reset evolution state (removes main file and backups)
./scripts/cleanup_evolution.sh

# Force specific genotype (development)
curl -X POST http://localhost:11436/evolution/force-genotype \
  -H "Content-Type: application/json" \
  -d '{"genotype_id": "abc123"}'
```

## Research Context

This implementation is based on the MemEvolve paper: **"MemEvolve: Meta-Evolution of Agent Memory Systems"** ([arXiv:2512.18746](https://arxiv.org/abs/2512.18746)).

Key insights implemented:
- **Bilevel optimization**: Inner experience loop, outer architecture evolution
- **Modular design space**: Four orthogonal components (Encode, Store, Retrieve, Manage)
- **Pareto-based selection**: Multi-objective optimization balancing performance and cost
- **Constrained mutation**: Respecting model capabilities and safety bounds

---

*Last updated: January 23, 2026*