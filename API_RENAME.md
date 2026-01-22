# API_RENAME.md

## API Endpoint Naming Plan

### Current Configuration Names

**✅ Keep As-Is:**
- `MEMEVOLVE_UPSTREAM_BASE_URL` - The main LLM API that users proxy through
- `MEMEVOLVE_EMBEDDING_BASE_URL` - Dedicated embedding API endpoint

**🔄 Rename Required:**
- `MEMEVOLVE_LLM_BASE_URL` → `MEMEVOLVE_MEMORY_BASE_URL`
- `MEMEVOLVE_LLM_API_KEY` → `MEMEVOLVE_MEMORY_API_KEY`
- `MEMEVOLVE_LLM_MODEL` → `MEMEVOLVE_MEMORY_MODEL`
- `MEMEVOLVE_LLM_TIMEOUT` → `MEMEVOLVE_MEMORY_TIMEOUT`
- `MEMEVOLVE_LLM_MAX_RETRIES` → `MEMEVOLVE_MEMORY_MAX_RETRIES`

### Rationale for Rename

**Current Issue:**
- `LLM_*` naming suggests this is for general LLM usage
- Actually used specifically for **memory management tasks** (experience encoding)
- Creates confusion with `UPSTREAM_*` which is the main LLM API
- `EMBEDDING_*` is clear, `LLM_*` is ambiguous

**Proposed Solution:**
- Rename to `MEMORY_*` to clearly indicate it's for memory management operations
- Aligns with the concept that this is a "memory LLM" for encoding tasks
- Reduces confusion in configuration and documentation

### Implementation Plan

#### Phase 1: Backward Compatibility
1. **Add new `MEMORY_*` variables** alongside existing `LLM_*` variables
2. **Update code** to prefer `MEMORY_*` variables but fall back to `LLM_*`
3. **Update documentation** to recommend `MEMORY_*` naming
4. **Deprecation warnings** for `LLM_*` usage

#### Phase 2: Full Migration
1. **Update all configuration files** (`.env.example`, `.docker.env.example`)
2. **Update documentation** to only reference `MEMORY_*` variables
3. **Update code comments** and variable names in source code
4. **Remove backward compatibility** fallbacks

#### Phase 3: Cleanup
1. **Update any hardcoded references** in scripts or examples
2. **Verify all tests pass** with new naming
3. **Update external documentation** and README examples

### Files to Update

#### Configuration Files
- `.env.example`
- `.docker.env.example`
- `.env` (user files)

#### Code Files
- `src/utils/config.py` - Add MEMORY_* variables, maintain LLM_* fallbacks
- `src/api/middleware.py` - Update variable usage
- `src/memory_system.py` - Update variable usage
- `src/components/encode/encoder.py` - Update variable usage

#### Documentation Files
- `README.md` - Update configuration examples
- `docs/user-guide/configuration.md` - Update variable references
- `AGENTS.md` - Update agent guidelines

### Migration Strategy

1. **Immediate**: Add `MEMORY_*` variables to config with priority over `LLM_*`
2. **Short-term**: Update documentation and examples to use `MEMORY_*`
3. **Medium-term**: Deprecate `LLM_*` variables with warnings
4. **Long-term**: Remove `LLM_*` variables entirely

### Benefits

- **Clarity**: `MEMORY_BASE_URL` clearly indicates purpose vs ambiguous `LLM_BASE_URL`
- **Consistency**: `UPSTREAM_*`, `EMBEDDING_*`, `MEMORY_*` create clear API hierarchy
- **Documentation**: Reduces confusion in setup and configuration
- **Maintenance**: Easier to understand which API is used for what purpose

### Timeline
- **Phase 1**: Implement backward-compatible changes (1-2 days)
- **Phase 2**: Full migration (1 week after testing)
- **Phase 3**: Cleanup and verification (1 week after migration)

---

## 📋 **Comprehensive File Impact Analysis**

### **Code Files (15 files)**

#### **Core System Files**
- `src/memory_system.py` - MemorySystemConfig class default values
- `src/components/encode/encoder.py` - ExperienceEncoder initialization
- `src/utils/config.py` - LLMConfig class and environment variable mappings
- `src/api/server.py` - Server startup warnings (MEMEVOLVE_LLM_API_KEY check)

#### **Test Files (6 files)**
- `src/tests/test_config.py` - LLM config testing
- `src/tests/test_api_server.py` - Server LLM configuration testing
- `src/tests/test_encode.py` - Encoder LLM configuration testing
- `src/tests/test_memory_system.py` - Memory system LLM configuration testing

#### **Example/Script Files (3 files)**
- `examples/basic_usage.py` - LLM configuration examples and checks
- `examples/graph_store_example.py` - LLM configuration references
- `scripts/start_api.py` - LLM API key validation

#### **Docker/Deployment Files (1 file)**
- `docker-compose.yml` - LLM service configuration

### **Configuration Files (2 files)**
- `.env.example` - Environment variable templates
- `.docker.env.example` - Docker environment variable templates

### **Documentation Files (6 files)**
- `docs/user-guide/getting-started.md` - Setup examples
- `docs/user-guide/configuration.md` - Configuration explanations
- `docs/user-guide/deployment_guide.md` - Deployment examples
- `docs/api/api-reference.md` - API configuration table
- `docs/api/troubleshooting.md` - Troubleshooting examples and checks
- `README.md` - Any configuration examples

### **Total Impact: 24 files**
- **Code**: 15 files (9 core + 6 tests)
- **Config**: 2 files
- **Docs**: 6 files
- **Examples**: 1 file (basic_usage.py)

### **Variables to Rename (6 variables)**
- `MEMEVOLVE_LLM_BASE_URL`
- `MEMEVOLVE_LLM_API_KEY`
- `MEMEVOLVE_LLM_MODEL`
- `MEMEVOLVE_LLM_AUTO_RESOLVE_MODELS`
- `MEMEVOLVE_LLM_TIMEOUT`
- `MEMEVOLVE_LLM_MAX_RETRIES`

### **Implementation Priority**
1. **Configuration system** (`src/utils/config.py`) - Core changes
2. **Core components** (`encoder.py`, `memory_system.py`) - Usage updates
3. **Configuration files** (`.env.example`, `.docker.env.example`) - Templates
4. **Documentation** - All doc files
5. **Tests** - Update test expectations
6. **Examples/Scripts** - Update examples and validation

---

*Created: January 22, 2026*
*Status: Planning Phase - Implementation after current testing round*</content>
<parameter name="filePath">API_RENAME.md