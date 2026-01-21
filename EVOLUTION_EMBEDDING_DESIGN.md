# Evolution Embedding Configuration Design

## Problem Statement

Evolution system needs to optimize embedding configuration (max_tokens), but these must be constrained by the embedding model's actual capabilities.

**Important Note:** Embedding dimension is a model capability constraint, not an evolvable parameter (as per MemEvolve paper Section 3.2).

## Requirements

### Priority Hierarchy for Embedding Settings

1. **Evolution State** (HIGHEST priority - if exists)
    - Stored in `evolution_state.json` as optimized values
    - Applied to ALL embedding operations
    - Only `max_tokens` can be evolved (not `embedding_dim`)
    
2. **Environment Variables**
    - `MEMEVOLVE_EMBEDDING_MAX_TOKENS`
    - `MEMEVOLVE_EMBEDDING_DIMENSION` (for manual override, not evolved)
    
3. **Auto-detection**
    - From `/models` endpoint of embedding API
    - Fields: `n_ctx_train` (max_tokens)
    
4. **Fallback Defaults**
    - `max_tokens: 512`
    - `embedding_dim: 768` (from model's native output)

### Evolution Constraints

- Evolution can **only mutate `max_tokens` to values ≤ base model capabilities**
- Embedding dimension is **NOT evolved** - uses model's native output
- Base capabilities = Environment OR Auto-detect OR Fallback
- Evolution's optimized values override base settings if they exist
- When embedding model changes, evolution must reset to new base capabilities

## Implementation Plan

### 1. Add Evolution Embedding Settings to EvolutionManager

```python
class EvolutionManager:
    def __init__(self, config: MemEvolveConfig, memory_system: MemorySystem):
        # ...
        
        # Evolution-optimized embedding settings
        self.evolution_embedding_dim: Optional[int] = None
        self.evolution_embedding_max_tokens: Optional[int] = None
        
        # Base model capabilities (maximum allowable values)
        self.base_embedding_dim: int = config.embedding.dimension or 768
        self.base_embedding_max_tokens: int = config.embedding.max_tokens or 512
```

### 2. Load/Save Evolution Embedding Settings

```python
def _load_persistent_state(self):
    # Load existing logic
    # ...
    
    # Load evolution embedding settings
    if 'evolution_embedding_dim' in data:
        self.evolution_embedding_dim = data['evolution_embedding_dim']
    if 'evolution_embedding_max_tokens' in data:
        self.evolution_embedding_max_tokens = data['evolution_embedding_max_tokens']

def _save_persistent_state(self):
    # Save existing logic
    # ...
    
    data['evolution_embedding_dim'] = self.evolution_embedding_dim
    data['evolution_embedding_max_tokens'] = self.evolution_embedding_max_tokens
```

### 3. Add Embedding Fields to EncodeConfig

Since `EncodeConfig` already has `max_tokens`, add `embedding_dim`:

```python
@dataclass
class EncodeConfig:
    encoding_strategies: List[str] = field(
        default_factory=lambda: ["lesson", "skill", "tool", "abstraction"]
    )
    llm_model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    embedding_dim: int = 768  # NEW: Evolution can mutate this
    batch_size: int = 10
    enable_abstractions: bool = True
    min_abstraction_units: int = 3
```

### 4. Constraint Validation in Mutation

```python
def _mutate_encode(self, config: EncodeConfig, mutation_rate: float) -> EncodeConfig:
    """Mutate encode configuration with model capability constraints."""
    
    # Get base model capabilities from EvolutionManager
    base_dim = self.base_embedding_dim
    base_max_tokens = self.base_embedding_max_tokens
    
    # Mutate max_tokens (cannot exceed base capability)
    if random.random() < mutation_rate:
        # Choose from values less than or equal to base_max_tokens
        valid_tokens = [256, 512, 1024, 2048, 4096, 8192]
        valid_tokens = [t for t in valid_tokens if t <= base_max_tokens]
        config.max_tokens = random.choice(valid_tokens)
    
    # Mutate embedding_dim (cannot exceed base capability)
    if random.random() < mutation_rate:
        # Choose from values less than or equal to base_dim
        valid_dims = [256, 384, 512, 768, 1024, 1536, 3072]
        valid_dims = [d for d in valid_dims if d <= base_dim]
        config.embedding_dim = random.choice(valid_dims)
    
    return EncodeConfig(...)
```

### 5. Create Evolution-Aware Embedding Provider

Modify `OpenAIEmbeddingProvider` to accept evolution overrides:

```python
class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
        max_tokens_per_request: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        evolution_override_dim: Optional[int] = None,      # NEW
        evolution_override_max_tokens: Optional[int] = None  # NEW
    ):
        # ... existing logic ...
        
        # Priority: evolution override > parameter > env > auto-detect > fallback
        if evolution_override_dim is not None:
            self._embedding_dim = evolution_override_dim
        elif embedding_dim is not None:
            self._embedding_dim = embedding_dim
        elif env_dim and env_dim.strip():
            self._embedding_dim = int(env_dim)
        elif auto_detect_dim:
            self._embedding_dim = auto_detect_dim
        else:
            self._embedding_dim = 768
            
        if evolution_override_max_tokens is not None:
            self.max_tokens_per_request = evolution_override_max_tokens
        # ... similar logic for max_tokens ...
```

### 6. Update create_embedding_function

```python
def create_embedding_function(
    provider: str = "dummy",
    evolution_manager: Optional[Any] = None,  # NEW
    **kwargs
) -> Callable[[str], np.ndarray]:
    """Create an embedding function with evolution support."""
    
    # If evolution manager provided and has optimized values, use them
    evolution_dim = None
    evolution_max_tokens = None
    if evolution_manager:
        evolution_dim = getattr(evolution_manager, 'evolution_embedding_dim', None)
        evolution_max_tokens = getattr(evolution_manager, 'evolution_embedding_max_tokens', None)
    
    if provider == "openai":
        provider_instance = OpenAIEmbeddingProvider(
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model"),
            timeout=kwargs.get("timeout", 60),
            max_tokens_per_request=kwargs.get("max_tokens"),
            embedding_dim=kwargs.get("embedding_dim"),
            evolution_override_dim=evolution_dim,
            evolution_override_max_tokens=evolution_max_tokens
        )
    # ... other providers ...
```

### 7. Update EvolutionManager to Pass Evolution Manager

```python
def _create_retrieval_strategy(self, config):
    """Create retrieval strategy from genotype config."""
    try:
        if config.strategy_type == "semantic":
            from ..utils.embeddings import create_embedding_function
            embedding_function = create_embedding_function(
                provider="openai",
                base_url=self.config.embedding.base_url or self.config.llm.base_url,
                api_key=self.config.embedding.api_key or self.config.llm.api_key,
                evolution_manager=self  # Pass evolution manager for embedding overrides
            )
            return SemanticRetrievalStrategy(embedding_function=embedding_function)
        # ... other strategies ...
```

## Usage Flow

### Startup (No Evolution State)

1. Load `config.embedding.dimension` and `max_tokens`
2. Try auto-detection from `/models` endpoint
3. Set `base_embedding_dim` and `base_embedding_max_tokens`
4. Evolution state values are None
5. Embedding operations use base config values

### After Evolution Runs

1. Evolution mutates `EncodeConfig.max_tokens` and `EncodeConfig.embedding_dim`
2. Mutation respects base capabilities (values ≤ base)
3. Best genotype's encoding settings are saved to evolution state
4. Next startup loads evolution state values
5. Embedding operations use evolution-optimized values

### Model Change

1. If embedding model changes, auto-detect new capabilities
2. Update `base_embedding_dim` and `base_embedding_max_tokens`
3. Reset/clear evolution embedding state if they exceed new base
4. Re-run evolution with new constraints

## Testing

```python
# Test 1: Evolution respects base capabilities
base_dim = 1024
evolution_mutates_to_dim = 512  # Valid (≤ base)
evolution_mutation_to_dim = 2048  # Invalid (> base)

# Test 2: Evolution state overrides base config
evolution_state_dim = 512
env_var_dim = 768
result_dim = 512  # Evolution wins

# Test 3: No evolution state uses base config
evolution_state_dim = None
env_var_dim = 768
result_dim = 768  # Env var wins

# Test 4: Priority chain (no env, no auto-detect)
evolution_state_dim = None
env_var_dim = None
auto_detect_dim = None
result_dim = 768  # Fallback wins
```
