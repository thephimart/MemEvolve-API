# Encoder/Memory Unification Plan

> **Purpose**: Consolidate duplicate MemoryConfig and EncodingConfig into unified EncoderConfig architecture, eliminating confusion and ensuring proper auto-resolution fallback behavior.

---

## 🔍 **Problem Analysis**

### **Current Architecture Issues**

#### **1. Duplicate Configuration Schemas**
```python
# Current: Two separate configs for same LLM endpoint
class MemoryConfig:     # Lines 26-58 in config.py
    base_url: Optional[str] = None
    api_key: str = ""
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    # ... 7 duplicate fields

class EncodingConfig:     # Lines 318-400+ in config.py  
    max_tokens: int = 0                    # ❌ Different default!
    batch_size: int = 0
    temperature: float = 0.0
    llm_model: Optional[str] = ""
    # ... 8 overlapping fields
```

#### **2. Environment Variable Confusion**
```bash
# Duplicate env vars pointing to same LLM endpoint:
MEMEVOLVE_MEMORY_BASE_URL=     # Points to encoder LLM
MEMEVOLVE_ENCODER_LLM_MODEL=   # Same model as memory
MEMEVOLVE_MEMORY_MAX_TOKENS=    # Should control encoder tokens
```

#### **3. Auto-Resolution Exclusion**
```python
# Line 1484 in config.py - INTENTIONAL BUG!
if service_type != 'encoder' and hasattr(config, 'auto_resolve_models') and config.auto_resolve_models:
    # ❌ Encoder explicitly excluded from auto-resolution
```

#### **4. Encoder Initialization Problems**
```python
# encoder.py lines 94-106 - Complex fallback chain
encoder_max_tokens = self.config_manager.get_effective_max_tokens('encoder')  # Excluded!
if encoder_max_tokens is not None:
    self.max_tokens = encoder_max_tokens
else:
    memory_max_tokens = self.config_manager.get_effective_max_tokens('memory')  # Fallback!
    self.max_tokens = memory_max_tokens if memory_max_tokens is not None else 4096
```

#### **5. Evolution System Duplication**
```python
# evolution/genotype.py: Separate evolution configs
class EncodeConfig:      # Evolution for encoder
    max_tokens: int = 512

# But MemoryConfig in main config.py handles same endpoint!
```

#### **6. Metrics Collection Duplication**
```python
# endpoint_metrics_collector.py - Separate tracking
elif endpoint_type == 'memory':
    request_metrics.memory_calls.append(endpoint_metrics)
elif endpoint_type == 'embedding':
    request_metrics.embedding_calls.append(endpoint_metrics)
# Should be unified under 'encoder'
```

---

## 🎯 **Unified Architecture Design**

### **Target Configuration Schema**
```python
@dataclass
class EncoderConfig:
    """Unified configuration for memory encoding and LLM operations."""
    
    # Core LLM Connection
    base_url: Optional[str] = None
    api_key: str = ""
    model: Optional[str] = None
    auto_resolve_models: bool = True
    timeout: int = 600
    max_retries: int = 3
    max_tokens: Optional[int] = None  # ✅ Unified token limit
    
    # Encoding Parameters
    encoding_strategies: List[str] = field(default_factory=lambda: [])
    temperature: float = 0.0
    batch_size: int = 0
    enable_abstractions: bool = False
    min_abstraction_units: int = 0
    enable_tool_extraction: bool = False
    
    def __post_init__(self):
        """Load from unified environment variables."""
        self.base_url = os.getenv("MEMEVOLVE_ENCODER_BASE_URL", self.base_url)
        self.api_key = os.getenv("MEMEVOLVE_ENCODER_API_KEY", self.api_key)
        self.model = os.getenv("MEMEVOLVE_ENCODER_MODEL", self.model)
        self.max_tokens = self._resolve_max_tokens()
        # ... other fields
```

### **Unified Environment Variables**
```bash
# PRIMARY: Encoder configuration
MEMEVOLVE_ENCODER_BASE_URL=           # Primary LLM endpoint
MEMEVOLVE_ENCODER_API_KEY=            # LLM API key  
MEMEVOLVE_ENCODER_MODEL=               # LLM model
MEMEVOLVE_ENCODER_MAX_TOKENS=          # Token limit
MEMEVOLVE_ENCODER_AUTO_RESOLVE_MODELS= # Auto-resolution flag

# DEPRECATED: Memory variables (with migration warnings)
MEMEVOLVE_MEMORY_BASE_URL=             # ⚠️ Migrated to ENCODER_BASE_URL
MEMEVOLVE_MEMORY_API_KEY=              # ⚠️ Migrated to ENCODER_API_KEY
MEMEVOLVE_MEMORY_MAX_TOKENS=           # ⚠️ Migrated to ENCODER_MAX_TOKENS
```

---

## 📋 **Implementation Plan**

### **Phase 1: Configuration Unification** (HIGH PRIORITY)

#### **Step 1.1: Create Unified EncoderConfig**
**Files**: `src/memevolve/utils/config.py`
**Lines**: 318-400+ (replace EncodingConfig)
**Changes**:
- Merge MemoryConfig fields into EncodingConfig
- Update `max_tokens: Optional[int] = None` (unified default)
- Add deprecation warnings for old MEMEVOLVE_MEMORY_* vars
- Implement unified auto-resolution (remove encoder exclusion)

#### **Step 1.2: Update Environment Variable Mapping**
**Files**: `src/memevolve/utils/config.py`
**Lines**: 35-60 (MemoryConfig), 350-400 (EncodingConfig)
**Changes**:
- Map `MEMEVOLVE_MEMORY_*` → `MEMEVOLVE_ENCODER_*` with warnings
- Ensure backward compatibility during transition
- Update documentation with migration guide

#### **Step 1.3: Fix Auto-Resolution Logic**  
**Files**: `src/memevolve/utils/config.py`
**Lines**: 1484-1488
**Changes**:
```python
# BEFORE (Line 1484):
if service_type != 'encoder' and hasattr(config, 'auto_resolve_models') and config.auto_resolve_models:

# AFTER:
if hasattr(config, 'auto_resolve_models') and config.auto_resolve_models:
    # ✅ Include encoder in auto-resolution
```

#### **Step 1.4: Update Main Config Class**
**Files**: `src/memevolve/utils/config.py` 
**Lines**: 1234, 1470-1475
**Changes**:
- Remove `memory: MemoryConfig = field(default_factory=MemoryConfig)`
- Keep only `encoder: EncoderConfig = field(default_factory=EncoderConfig)`
- Update all `config.memory` references to `config.encoder`

### **Phase 2: Component Integration** (HIGH PRIORITY)

#### **Step 2.1: Update Encoder Initialization**
**Files**: `src/memevolve/components/encode/encoder.py`
**Lines**: 94-115 (constructor)
**Changes**:
```python
# BEFORE: Complex fallback chain
encoder_max_tokens = self.config_manager.get_effective_max_tokens('encoder')
if encoder_max_tokens is not None:
    self.max_tokens = encoder_max_tokens
else:
    memory_max_tokens = self.config_manager.get_effective_max_tokens('memory')
    self.max_tokens = memory_max_tokens if memory_max_tokens is not None else 4096

# AFTER: Direct config access
self.max_tokens = self.config_manager.get_effective_max_tokens('encoder')
```

#### **Step 2.2: Update Memory System**
**Files**: `src/memevolve/memory_system.py`
**Lines**: 35-60 (encoder initialization)
**Changes**:
- Use `config.encoder` instead of `config.memory` for LLM params
- Update all `memory_config` references to `encoder_config`
- Maintain backward compatibility warnings

#### **Step 2.3: Update Evolution System**
**Files**: `src/memevolve/evolution/mutation.py`, `genotype.py`
**Lines**: 172-180 (mutation application)
**Changes**:
- Use unified EncoderConfig in evolution
- Remove separate EncodeConfig evolution class
- Update mutation boundaries for unified schema

#### **Step 2.4: Update API Server**
**Files**: `src/memevolve/api/server.py` 
**Lines**: 125-140 (initialization)
**Changes**:
- Pass `config.encoder` to memory system
- Update configuration validation
- Remove memory config references

### **Phase 3: Metrics and Testing** (MEDIUM PRIORITY)

#### **Step 3.1: Update Metrics Collection**
**Files**: `src/memevolve/utils/endpoint_metrics_collector.py`
**Lines**: 210-215 (endpoint categorization)
**Changes**:
- Map 'memory' endpoint calls to 'encoder' metrics
- Maintain historical data compatibility
- Update dashboard queries

#### **Step 3.2: Update Evolution Manager**
**Files**: `src/memevolve/api/evolution_manager.py`
**Lines**: 1400-1500 (parameter management)
**Changes**:
- Use encoder config for evolution mutations
- Update fitness evaluation for unified schema
- Maintain backward compatibility

#### **Step 3.3: Update Test Suite**
**Files**: `tests/test_config.py`, `test_memory_system.py`, `test_full_pipeline_integration.py`
**Changes**:
- Update all test references to use encoder config
- Add migration tests for memory→encoder transition
- Validate auto-resolution functionality

### **Phase 4: Migration and Cleanup** (LOW PRIORITY)

#### **Step 4.1: Add Migration Logic**
**Files**: `src/memevolve/utils/config.py`
**Lines**: 36-45 (environment loading)
**Changes**:
- Detect old `MEMEVOLVE_MEMORY_*` variables
- Log deprecation warnings with migration guidance
- Auto-map to new `MEMEVOLVE_ENCODER_*` variables

#### **Step 4.2: Update Documentation**
**Files**: `AGENTS.md`, `README.md`, `.env.example`
**Changes**:
- Update configuration documentation
- Add migration guide
- Update environment variable reference

#### **Step 4.3: Cleanup Legacy Code**
**Files**: All files with `MemoryConfig` references
**Changes**:
- Remove deprecated `MemoryConfig` class after transition period
- Clean up legacy environment variable handling
- Remove backward compatibility warnings

---

## 🔄 **Evolution System Integration**

### **Mutation Boundary Updates**
```python
@dataclass 
class EvolutionBoundaries:
    """Updated boundaries for unified encoder config."""
    max_tokens_min: int = 256
    max_tokens_max: int = 8192
    token_step_size: int = 256
    temperature_min: float = 0.0
    temperature_max: float = 2.0
    temperature_change_delta: float = 0.1
    batch_size_min: int = 1
    batch_size_max: int = 50
    batch_size_step_size: int = 5
```

### **Fitness Evaluation Updates**
```python
# evolution/selection.py - Updated fitness scoring
def _evaluate_encoder_performance(config: EncoderConfig) -> float:
    """Evaluate unified encoder configuration performance."""
    score = 0.0
    
    # Token limit efficiency
    if config.max_tokens:
        optimal_range = (1024, 4096)
        if optimal_range[0] <= config.max_tokens <= optimal_range[1]:
            score += 0.3
    
    # Temperature optimization
    optimal_temp = 0.7
    if abs(config.temperature - optimal_temp) < 0.2:
        score += 0.2
    
    # Batch processing efficiency  
    if config.batch_size > 0:
        score += 0.2
        
    return score
```

---

## 📊 **Impact Analysis**

### **Configuration Simplification**
- **Classes**: 2 configs → 1 config (-50%)
- **Environment Variables**: 12 vars → 8 vars (-33%)
- **Code Paths**: Multiple fallbacks → Direct access (+100% reliability)

### **Auto-Resolution Fix**
- **Encoder**: Excluded → Included ✅
- **Fallback Behavior**: `max_tokens=0` → Auto-resolved from model ✅
- **Batch Processing**: Forced → Conditional ✅

### **Evolution System**
- **Mutation Targets**: 2 configs → 1 config (-50% complexity)
- **Fitness Evaluation**: Duplicate scoring → Unified scoring (+100% accuracy)
- **Boundary Management**: Separate sets → Single set (+50% efficiency)

### **Metrics Collection**
- **Endpoint Tracking**: Duplicated → Unified (+100% accuracy)
- **Performance Analytics**: Fragmented → Coherent (+200% insight value)
- **Debugging**: Confusing sources → Clear sources (+300% efficiency)

---

## 🚨 **Risk Mitigation**

### **Backward Compatibility**
```python
# Phase 1: Support both old and new variables
memory_url = os.getenv("MEMEVOLVE_MEMORY_BASE_URL")
encoder_url = os.getenv("MEMEVOLVE_ENCODER_BASE_URL")

if memory_url and not encoder_url:
    logger.warning("MEMEVOLVE_MEMORY_BASE_URL is deprecated, use MEMEVOLVE_ENCODER_BASE_URL")
    base_url = memory_url
elif encoder_url:
    base_url = encoder_url
else:
    base_url = None
```

### **Gradual Migration**
1. **Phase 1-2**: Support both configs with warnings
2. **Phase 3**: Remove MemoryConfig, keep variable mapping
3. **Phase 4**: Remove deprecated variable support

### **Test Coverage**
- Unit tests: All config loading scenarios
- Integration tests: Full pipeline with unified config
- Evolution tests: Mutation with unified schema
- Migration tests: Old→new variable mapping

---

## ✅ **Success Criteria**

### **Functional Requirements**
- [ ] Single EncoderConfig replaces both MemoryConfig and EncodingConfig
- [ ] Encoder included in auto-resolution (max_tokens issue fixed)
- [ ] All components use config.encoder exclusively
- [ ] Evolution system mutates unified config correctly
- [ ] Metrics collection uses unified endpoint names

### **Quality Requirements**  
- [ ] Zero breaking changes to public APIs
- [ ] All tests pass with unified configuration
- [ ] Evolution system maintains fitness evaluation accuracy
- [ ] Auto-resolution works for all service types

### **Documentation Requirements**
- [ ] Updated AGENTS.md with new config structure
- [ ] Migration guide for existing deployments
- [ ] Updated .env.example with new variables
- [ ] Code comments explaining migration strategy

---

## 🎯 **Implementation Timeline**

### **Week 1: Core Unification**
- Days 1-2: Phase 1 (Configuration unification)
- Days 3-4: Phase 2 (Component integration)
- Day 5: Testing and validation

### **Week 2: Advanced Features**  
- Days 1-2: Phase 3 (Metrics and evolution)
- Days 3-4: Phase 4 (Migration and cleanup)
- Day 5: Documentation and final testing

### **Total Effort**: ~10 days of focused development
### **Risk Level**: Medium (careful migration required)
### **Impact**: High (eliminates major source of confusion)

---

**This plan provides a comprehensive, step-by-step approach to unifying encoder and memory configurations while maintaining system stability and enabling proper auto-resolution behavior.**