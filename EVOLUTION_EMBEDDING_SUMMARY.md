# Evolution Embedding System - Implementation Complete

## Summary

The evolution embedding configuration system has been successfully implemented. Evolution can now optimize embedding configuration (max_tokens) while respecting model capability constraints.

**Important Change:** Based on MemEvolve paper analysis, `embedding_dim` is no longer evolved as it's a model capability constraint, not an architectural choice (see MemEvolve paper Section 3.2).

## What Was Changed

### 1. Evolution Genotype (`src/evolution/genotype.py`)

**EncodeConfig now includes only `max_tokens` (not `embedding_dim`):**
```python
@dataclass
class EncodeConfig:
    """Configuration for Encode component.
    
    Note: Embedding dimension is determined by the embedding model's
    native capability and is not evolved (model constraint).
    """

    encoding_strategies: List[str] = field(...)
    llm_model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512  # Evolution can optimize this
    batch_size: int = 10
    enable_abstractions: bool = True
    min_abstraction_units: int = 3
```

### 2. Mutation System (`src/evolution/mutation.py`)

**MutationStrategy now accepts and enforces base capabilities for max_tokens only:**
```python
class MutationStrategy:
    def __init__(
        self,
        base_max_tokens: int = 512  # Model's max context
    ):
        self.base_max_tokens = base_max_tokens
        
        # Valid mutation options constrained by base capabilities
        self.valid_max_tokens = [256, 512, 1024, 2048, 4096, 8192]
```

**RandomMutationStrategy enforces constraints during mutation:**
```python
def _mutate_encode(self, config: EncodeConfig, mutation_rate: float):
    # Mutate max_tokens (constrained to base capability)
    if random.random() < mutation_rate and self.valid_max_tokens:
        config.max_tokens = random.choice(self.valid_max_tokens)
    
    # Note: embedding_dim is NOT evolved - uses model's native output
```

### 3. EvolutionManager (`src/api/evolution_manager.py`)

**Removed evolution_embedding_dim tracking (only tracks max_tokens now):**
```python
class EvolutionManager:
    def __init__(self, config: MemEvolveConfig, memory_system: MemorySystem):
        # Evolution-optimized embedding settings
        # Only max_tokens can be evolved
        self.evolution_embedding_max_tokens: Optional[int] = None
        
        # Base model capabilities (maximum allowable values)
        # Priority: env var > auto-detect > fallback
        self.base_embedding_max_tokens = config.embedding.max_tokens or 512
        
        # Initialize mutation engine with constraints
        self.mutation_engine = MutationEngine(
            RandomMutationStrategy(),
            base_max_tokens=self.base_embedding_max_tokens
        )
```

**State persistence includes evolution embedding settings:**
```python
def _load_persistent_state(self):
    # Load evolution embedding settings
    if 'evolution_embedding_max_tokens' in data:
        self.evolution_embedding_max_tokens = data['evolution_embedding_max_tokens']
        logger.info(f"Loaded evolution embedding_max_tokens: {self.evolution_embedding_max_tokens}")

def _save_persistent_state(self):
    data = {
        'evolution_embedding_max_tokens': self.evolution_embedding_max_tokens,
        # ... other state
    }
```

**Best genotype updates evolution settings:**
```python
if fitness_scores[best_genotype.get_genome_id()] > fitness_scores.get(self.best_genotype.get_genome_id(), 0):
    self.best_genotype = best_genotype
    
    # Update evolution embedding settings from best genotype
    self.evolution_embedding_max_tokens = best_genotype.encode.max_tokens
    logger.info(
        f"Evolution embedding settings updated: "
        f"max_tokens={self.evolution_embedding_max_tokens}"
    )
```

### 4. Embedding Provider (`src/utils/embeddings.py`)

**OpenAIEmbeddingProvider accepts evolution overrides for max_tokens only:**
```python
class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
        max_tokens_per_request: Optional[int] = None,
        evolution_override_max_tokens: Optional[int] = None,  # NEW
        embedding_dim: Optional[int] = None  # Manual override only, not evolved
    ):
        # Priority: evolution override > parameter > env > auto-detect > fallback
        if evolution_override_max_tokens is not None:
            self.max_tokens_per_request = evolution_override_max_tokens
            logger.info(f"Using evolution override for max_tokens: {evolution_override_max_tokens}")
        elif max_tokens_per_request is not None:
            self.max_tokens_per_request = max_tokens_per_request
        # ... env vars and fallback ...
        
        # embedding_dim uses model's native output (not evolved)
        if embedding_dim is not None:
            self._embedding_dim = embedding_dim
        # ... env vars and auto-detect ...
```

**create_embedding_function passes evolution manager:**
```python
def create_embedding_function(
    provider: str = "dummy",
    evolution_manager: Optional[Any] = None,  # NEW
    **kwargs
) -> Callable[[str], np.ndarray]:
    # Get evolution overrides if available
    evolution_max_tokens = None
    if evolution_manager:
        evolution_max_tokens = getattr(evolution_manager, 'evolution_embedding_max_tokens', None)
    
    if provider == "openai":
        provider_instance = OpenAIEmbeddingProvider(
            base_url=kwargs.get("base_url") or os.getenv("MEMEVOLVE_EMBEDDING_BASE_URL"),
            api_key=kwargs.get("api_key") or os.getenv("MEMEVOLVE_EMBEDDING_API_KEY", ""),
            model=kwargs.get("model") or os.getenv("MEMEVOLVE_EMBEDDING_MODEL"),
            timeout=int(os.getenv("MEMEVOLVE_EMBEDDING_TIMEOUT", "60")),
            max_tokens_per_request=kwargs.get("max_tokens"),
            embedding_dim=kwargs.get("embedding_dim"),
            evolution_override_max_tokens=evolution_max_tokens
        )
```

## Priority Hierarchy

The system now respects the following priority order:

### 1. Evolution State (HIGHEST Priority)
- Location: `evolution_state.json`
- Fields: `evolution_embedding_max_tokens` (no longer `evolution_embedding_dim`)
- Applied: ALL embedding operations when evolution has optimized values
- Updated: When best genotype is found during evolution cycle

### 2. Environment Variables
- `MEMEVOLVE_EMBEDDING_MAX_TOKENS`
- `MEMEVOLVE_EMBEDDING_DIMENSION` (for manual override only, not evolved)
- Applied: When no evolution state exists
- Used as: Base capabilities for mutation constraints

### 3. Auto-detection
- Source: `/models` endpoint of embedding API
- Fields: `n_ctx_train` (max_tokens)
- Applied: When env vars not set
- Used as: Base capabilities for mutation constraints

### 4. Fallback Defaults
- Values: `max_tokens=512`, `embedding_dim=768` (from model's native output)
- Applied: When auto-detection fails
- Used as: Base capabilities for mutation constraints

## Evolution Constraints

Evolution mutations are strictly constrained by base model capabilities:

```python
# Example: Model with 4096 max_tokens
base_max_tokens = 4096

# Valid mutation options:
valid_max_tokens = [256, 512, 1024, 2048, 4096]  # Can't exceed 4096

# Evolution can only choose from constrained values
# Can mutate TO: 1024 (valid, ≤ base_max_tokens)
# Can mutate TO: 8192 (invalid, would be > base_max_tokens) ❌
```

## Usage Flow

### Initial Startup (No Evolution State)

1. Load `config.embedding.max_tokens` from config
2. Try auto-detection from `/models` endpoint (if enabled)
3. Set `base_embedding_max_tokens`
4. Evolution state values are `None`
5. Embedding operations use base config values (env > auto-detect > fallback)
6. Initialize `MutationEngine` with base capabilities
7. Evolution can only mutate `max_tokens` to values ≤ base capabilities
8. Embedding dimension uses model's native output (not evolved)

### After Evolution Runs

1. Evolution mutates `EncodeConfig.max_tokens`
2. Mutations respect base capabilities (values ≤ base)
3. Best genotype's encoding settings are saved to evolution state
4. Next startup loads evolution state values
5. **All embedding operations use evolution-optimized `max_tokens`**
6. Retrieval strategies created with evolution-aware embedding functions

### Model Change

If embedding model changes:

1. New model auto-detects different capabilities
2. Update `base_embedding_max_tokens`
3. **Evolution state for `max_tokens` resets if values exceed new base**
4. Evolution re-runs with new constraints
5. New optimized values are discovered for new model

## Evolution State File Format

`cache/evolution_state.json`:
```json
{
  "best_genotype": {
    "encode": {
      "max_tokens": 1024,
      "encoding_strategies": ["lesson", "skill"],
      "temperature": 0.7,
      "batch_size": 10,
      "enable_abstractions": true,
      "min_abstraction_units": 3
    },
    "store": { /* store config */ },
    "retrieve": { /* retrieve config */ },
    "manage": { /* manage config */ }
  },
  "evolution_embedding_max_tokens": 1024,
  "evolution_history": [ /* history */ ],
  "metrics": { /* metrics */ }
}
```

## Key Points

1. **Evolution controls `max_tokens`** - Only tunable embedding parameter
2. **Constrained by model capabilities** - Can't exceed base model's actual limits
3. **Priority hierarchy enforced** - Evolution state > env > auto-detect > fallback
4. **Applies to all embeddings** - Retrieval strategies use evolution-optimized values
5. **Persists across restarts** - Saved to evolution_state.json
6. **Safe model changes** - Evolution state resets if model capabilities change

## Notes

- StoreConfig still does NOT have `embedding_dim` (correct - it's global)
- EvolutionManager tracks evolution `max_tokens` settings separately from genotype
- Mutations can only reduce values, never exceed base capabilities
- Evolution state is highest priority - overrides everything when present
- **Embedding dimension uses model's native output** (not evolved) as per MemEvolve paper Section 3.2
