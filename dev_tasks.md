# MemEvolve-API Development Tasks

> **Purpose**: Debugging & development roadmap for MemEvolve-API. Focuses on **active issues**, **verification status**, and **immediate priorities**. Maintains only verified tasks and current blockers.

---

## 1. Current System State (Development Phase)

**Status**: 🟡 **DEBUGGING & DEVELOPMENT PHASE** (Feb 3, 2026)

**Core Systems**: ✅ **FUNCTIONAL**
- **Memory System**: 356+ experiences stored, semantic retrieval operational
- **Evolution System**: Fitness calculation working, boundary validation needed
- **API Server**: Production endpoints operational, dashboard endpoints WIP
- **Configuration**: Centralized config partially implemented, violations exist

**Recent Improvements**:
- ✅ **P0.24**: Evolution state persistence (6060f5d)
- ✅ **P0.25**: Memory injection consistency (6060f5d)  
- ✅ **SEMANTIC SCORING**: Length penalty eliminated (8a87f6b)
- ✅ **P1.2**: Relevance filtering implemented (6060f5d)

---

## 2. ACTIVE DEBUGGING ISSUES

### **🔥 HIGH PRIORITY BLOCKERS**

#### **P0.26 ❌ Systematic Hardcoded Value Violations**
- **Problem**: Hardcoded values violate AGENTS.md centralized config policy throughout codebase
- **Impact**: Evolution system generates invalid configurations, environment overrides ignored
- **Root Cause**: Multiple files contain hardcoded parameters instead of using ConfigManager
- **Evidence**:
  - `genotype.py:330`: `default_top_k=15` (should use config boundaries)
  - `genotype.py:264`: `default_top_k=3` (AgentKB genotype)
  - Additional hardcoded values likely exist across codebase
- **Policy Violation**: "[FORBIDDEN] Hardcoded values in any other files"
- **Fix Required**: Full codebase audit + centralized config compliance
- **Status**: ❌ IN PROGRESS

#### **P0.27 ❌ Evolution Boundary Validation Bypass**
- **Problem**: Cerebra genotype hardcoded `top_k=15` violates `TOP_K_MAX=10` environment override
- **Location**: `src/memevolve/evolution/genotype.py:330`
- **Impact**: Evolution creates invalid configurations that bypass system boundaries
- **Fix Required**: Genotype cleanup to respect dynamic boundaries
- **Status**: ❌ IN PROGRESS

### **🟡 MEDIUM PRIORITY ISSUES**

#### **P0.28 ❌ Dashboard API Implementation**
- **Problem**: Dashboard endpoints (`/dashboard`, `/health`, `/memory`) exist but incomplete
- **Impact**: No real-time system monitoring, limited observability
- **Location**: API route handlers and data collection
- **Fix Required**: Complete dashboard endpoints with real-time metrics
- **Status**: ❌ IN PROGRESS

#### **P0.29 ✅ Critical Code Quality Issues (COMPLETED - 60+ violations eliminated)**
- **Problem**: 239 linting violations blocking clean codebase, including critical runtime errors
- **COMPLETED ISSUES**:
  - ✅ **Runtime Error Fixed**: `F821 undefined name 'fitness_scores'` in `evolution_manager.py:1170`
  - ✅ **F-string Issues**: Fixed 8 `F541 f-string missing placeholders` across multiple files
  - ✅ **Unused Variables**: Cleaned 9 `F841 unused variables` (request_time, query_tokens, parsed, etc.)
  - ✅ **Import Redefinition**: Fixed 4 `F811 redefinition` errors (duplicate imports)
  - ✅ **Major Import Cleanup**: Removed 50+ `F401 unused imports` across 15+ files
    - `utils/__init__.py`: Removed 42 unused imports (drastic cleanup)
    - Multiple component files: Cleaned unused imports
- **Remaining Issues**: ~180 `E501` line length violations (>100 characters), non-blocking
- **Files Modified**: 22 total including critical system files and components
- **Impact**: Critical runtime errors eliminated, code quality significantly improved
- **Fix Applied**: Systematic cleanup of all blocking violations except line length
- **Status**: ✅ COMPLETED (major violations resolved)

#### **P0.30 ❌ Incomplete Metrics Implementation Investigation Required**
- **Problem**: `enhanced_http_client.py` get() and put() methods have incomplete timing metrics
- **Evidence**: 
  - `post()` method: Complete implementation with `start_time = time.time()` → `request_time = (time.time() - start_time) * 1000`
  - `get()` and `put()` methods: Only `start_time = time.time()` assigned, never used (F841 violation)
  - Pattern suggests copy-paste error where timing logic was partially implemented
- **Impact**: Missing timing data for GET/PUT requests may skew API performance metrics and evolution fitness calculations
- **Investigation Required**: 
  - Verify if timing data is critical for endpoint metrics collector
  - Determine if incomplete implementation affects fitness calculations in evolution system
  - Assess impact on performance monitoring and system optimization
- **Potential Fix**: Complete timing implementation or remove unused code based on investigation findings
- **Status**: ❌ PENDING - HIGH PRIORITY INVESTIGATION

### **🔍 VERIFICATION REQUIRED**

#### **Genotype Application Completeness**
- **Issue**: Only retrieval strategy verified, need to verify ALL mutations propagate
- **Check**: encoder, management, storage parameters in ConfigManager updates
- **Priority**: HIGH - Evolution system integrity verification
- **Status**: ❌ PENDING

---

## 3. IMPLEMENTATION QUEUE (Priority Order)

### **IMMEDIATE (This Session)**
1. **P0.26**: Systematic hardcoded value audit - Replace hardcoded values with ConfigManager calls
2. **P0.27**: Evolution boundary enforcement - Fix cerebra genotype to respect TOP_K_MAX
3. **P0.30**: Incomplete metrics investigation - Determine impact of missing GET/PUT timing data on evolution fitness (HIGH PRIORITY)
4. **Genotype Verification**: Validate all evolution mutations propagate through ConfigManager
5. **P0.29 (remaining)**: Complete code quality cleanup - Fix ~180 line length violations (MEDIUM PRIORITY)

### **NEXT SESSION**
1. **P0.28**: Complete dashboard API endpoints
2. **P1.3**: Unify quality scoring systems (643 → ~350 lines)
3. **P0.29 (final)**: Complete remaining line length violations (~180 non-blocking issues)

---

## 4. TESTING & VALIDATION STATUS

### **CURRENT FOCUS**
- **Relevance Filtering**: Monitor P1.2 effectiveness (target: 95%+ relevant recall)
- **Boundary Compliance**: Verify TOP_K_MAX=10 enforcement across all genotypes
- **Semantic Scoring**: Track score distribution improvements after normalization fix

### **VALIDATION METRICS**
- **Memory Relevance**: % of retrieved memories passing 0.5 threshold
- **Boundary Compliance**: % of configurations respecting environment limits
- **Evolution Integrity**: % of genotype mutations properly applied

---

## 5. RESOLVED ISSUES (Reference)

### **CRITICAL FIXES APPLIED**
- ✅ **P0.19**: Evolution negative variance fixed
- ✅ **P0.20**: Memory quality validation implemented  
- ✅ **P0.21**: Invalid configuration prevention
- ✅ **P0.22**: Upstream API health monitoring
- ✅ **P0.23**: Evolution application verified
- ✅ **P0.24**: Evolution state persistence (6060f5d)
- ✅ **P0.25**: Memory injection consistency (6060f5d)
- ✅ **P0.28-P0.29**: Memory system debugging completed
- ✅ **P0.29**: Major code quality cleanup completed (60+ violations eliminated)
- ✅ **P1.2**: Memory relevance filtering implemented (6060f5d)
- ✅ **SEMANTIC SCORING**: Vector normalization (8a87f6b)

### **RECENT COMMITS**
- `9d872e9`: Major code quality cleanup - Fixed 60+ linting violations, eliminated critical runtime errors
- `6060f5d`: Fixed P0.24, P0.25, P1.2 - Critical memory issues resolved
- `8a87f6b`: Fixed semantic scoring harshness - Vector normalization eliminates length penalty

---

## 6. DEVELOPMENT NOTES

### **CURRENT ARCHITECTURE COMPLIANCE**
- **Status**: 🟡 **PARTIALLY COMPLIANT** - Hardcoded values still exist
- **Priority**: HIGH - AGENTS.md centralized config policy restoration required
- **Blocker**: P0.26 prevents production readiness

### **PRODUCTION READINESS**
- **Current Status**: ❌ **NOT READY** - Architectural violations must be resolved
- **Requirement**: All evolution parameters must respect centralized config hierarchy
- **Timeline**: P0.26 & P0.27 completion needed for production deployment

### **🔍 Missing Context for New Developers (Not in AGENTS.md)**

### **Critical Implementation Details**:
- **Boundary Validation Pattern**: Use `min(hardcoded_value, boundary_limit)` when replacing hardcoded values
- **Evolution Sync**: ConfigManager updates use dot notation: `config_manager.update(retrieval__default_top_k=7)`
- **Testing Command**: All tests run via `./scripts/run_tests.sh` (see AGENTS.md for variants)

### **Production Blockers Context**:
- **Current Violation**: `genotype.py:330` has `default_top_k=15` that ignores `MEMEVOLVE_TOP_K_MAX=10`
- **Why P0.26 is Critical**: Without centralized config compliance, evolution system creates invalid configurations
- **Validation Priority**: Genotype verification needed beyond retrieval strategy (encoder, management, storage)