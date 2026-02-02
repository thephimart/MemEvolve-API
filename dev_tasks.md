# MemEvolve‑API — Development Tasks (Agent Runbook)

> **Purpose**: A compressed, non‑overlapping, execution‑ready task file to pair with `AGENTS.md`. This document focuses on **what is broken**, **what must be logged**, **hard metrics for validation**, and a **prioritized task list** so an agent can start immediately.

---

## 1. Current System Snapshot (Latest Progress)

**Status**: 🔄 **MAJOR PROGRESS - CORE SYSTEMS FUNCTIONAL** (Feb 2, 2026)

- **Version**: v2.0.0 (master) + FAISS integration fixes
- **API Endpoints**: Operational with core memory functionality working
- **Evolution System**: Production ready with real metrics
- **Memory System**: 216+ experiences stored, searchable, and properly incrementing
- **FAISS Integration**: Fixed and operational (IVF index, 768-dim embeddings)
- **Server Architecture**: Fixed - all endpoints respond <0.01s except LLM processing

**Critical Fixes Applied**:
- ✅ **P0.10**: Real endpoint trajectory testing implemented
- ✅ **P0.11**: Environment variable override capability restored  
- ✅ **P0.12**: Hardcoded 1024 limit removed
- ✅ **P0.13**: Storage integrity failure fixed (100% success rate)
- ✅ **P0.14-P0.16**: Server architecture fixes (cached metrics, delayed evolution, real dashboard data)
- ✅ **P0.18**: FAISS vector store integration completely fixed
- ✅ **Memory Count Issue**: ID generation bug resolved, count now properly increments

**System Status**: **CORE SYSTEMS WORKING - Remaining Type Errors**

---

## 2. COMPLETED IMPLEMENTATIONS

### **P0.10 ✅ COMPLETED** - Real Evolution Testing
- **Problem**: 18 identical fitness scores (0.837164) due to missing trajectory testing
- **Solution**: Implemented `_run_test_trajectories()` with actual endpoint performance testing
- **Result**: Evolution measures real upstream/memory/embedding API performance

### **P0.11 ✅ COMPLETED** - Config Policy Compliance  
- **Problem**: `MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD=1024` ignored, using 1000
- **Solution**: Fixed evolution manager to preserve non-evolved parameters from centralized config
- **Result**: Environment variables properly override defaults through evolution

### **P0.12 ✅ COMPLETED** - Parameter Range Expansion
- **Problem**: Mutation choices limited to `[256, 512, 1024, 2048]` 
- **Solution**: Extended to `[256, 512, 1024, 2048, 4096]` respecting config boundaries
- **Result**: Evolution can explore full parameter space up to 4096

### **P0.13 ✅ COMPLETED** - Storage Integrity Protection
- **Problem**: 717 memories lost (46.6% loss rate) during batch operations
- **Solution**: Improved `store_batch()` with individual unit storage and proper error handling
- **Result**: Each memory unit stored transactionally with failure detection

### **SERVER ARCHITECTURE FIXES ✅ COMPLETED**

#### **P0.14 ✅ COMPLETED** - Cached Health Metrics
- **Problem**: `/memory/stats` endpoint blocked for 30+ seconds with full database scan
- **Solution**: Added 30-second TTL cache to `simple_strategy.py` and `vector_store.py`
- **Result**: `/memory/stats` responds in 0.009s instead of 30s

#### **P0.15 ✅ COMPLETED** - Delayed Evolution Startup  
- **Problem**: Evolution started immediately, blocking HTTP server during initialization
- **Solution**: Added 30-second delayed startup in `server.py`
- **Result**: HTTP server ready before evolution begins, no race condition

#### **P0.16 ✅ COMPLETED** - Real Dashboard Data Collection
- **Problem**: Dashboard using mock data instead of real endpoint metrics
- **Solution**: Removed mock fallbacks, dashboard now fails hard if collector unavailable
- **Result**: Real operational data shown with actual API performance metrics

---

## 3. VALIDATION RESULTS

### **Endpoint Performance Tests (Feb 2, 2026 - Post FAISS Fix)**

| Endpoint | Response Time | Status | Functionality |
|----------|---------------|--------|---------------|
| **`/health`** | **0.006s** | ✅ | System healthy |
| **`/memory/stats`** | **0.009s** | ✅ | Real-time memory count working |
| **`/memory/search`** | **8ms** | ✅ | **FAISS semantic search operational** |
| **`/dashboard-data`** | **0.003s** | ✅ | Real data collector working |
| **`/evolution/status`** | **0.004s** | ✅ | Evolution system ready |
| **`/v1/chat/completions`** | **30-40s** | ✅ | Full memory integration working |

### **System Health Metrics**
- **Storage Integrity**: 100% success rate (all vector operations successful)
- **Memory Count**: 216+ experiences stored, **properly incrementing**
- **FAISS Integration**: Fully operational (IVF index, 768-dim embeddings)
- **Evolution History**: 1+ cycles completed, real performance testing
- **API Performance**: Upstream 20-40s, **memory retrieval <10ms**
- **Vector Operations**: Storage, search, rebuild all working correctly

---

## 4. CURRENT DEBUGGING TASKS

### **🔍 CURRENT BLOCKERS & ISSUES**

#### **P0.17 Fix Type Errors in Memory System**
- **Location**: `memory_system.py:871,881,1181`
- **Errors**:
  - Cannot access attribute "retrieval" for class "MemorySystemConfig"
  - Type "Unknown | int | None" cannot be assigned to parameter "top_k"
  - "strategy" is not a known attribute of "None"
- **Impact**: Runtime errors when accessing memory system functionality
- **Root Cause**: Missing type hints and incorrect attribute access patterns

#### **P0.19 Fix Endpoint Metrics Collector Import Errors**
- **Location**: `endpoint_metrics_collector.py:139-141`
- **Errors**:
  - "MemoryScorer", "ResponseScorer", "TokenAnalyzer" possibly unbound
- **Impact**: Dashboard data collection fails, metrics unavailable
- **Root Cause**: Import handling and optional dependency management

### **📋 P1 — KNOWN ISSUES TO INVESTIGATE**

#### **P1.1 Eliminate Generic / Meta Memories at Encode Time**
- **Problem**: Q&A formatted input causes meta‑descriptions (~19% generic)
- **Implementation**: Send assistant response only to encoder, add rule-based summarization
- **Targets**: Generic/vague memories < 10%, Actionable insights > 80%

#### **P1.2 Prevent Invalid Memory Storage**
- **Problem**: Null content and fallback chunks still stored
- **Implementation**: Add hard validation, delete known corrupted units
- **Known units to delete**: null: `unit_157`, `unit_949`; fallback: `unit_300`, `unit_317`, `unit_339`, `unit_348`, `unit_684`

#### **P1.3 Unify Quality Scoring Systems**
- **Problem**: Three separate scoring systems with overlapping functionality (643 lines total)
- **Implementation**: Create unified `ScoringEngine` class, merge duplicate logic
- **Expected**: 643 lines → ~350 lines, zero lint errors

#### **P1.4 Filter Irrelevant Memories from Injection**
- **Problem**: All retrieved memories injected regardless of relevance score
- **Implementation**: Filter memories with score > 0.5 before injection
- **Result**: Log format: "Injected X relevant memories (retrieved: Y, limit: Z)"

#### **P1.5 Remove Time-Based Evolution Trigger**
- **Problem**: Evolution triggers on time (24 hours) even when server idle
- **Implementation**: Remove time-based trigger, keep iteration-based only
- **Result**: Server idle = no evolution activity

#### **P1.6 Dynamic Performance Baseline**
- **Problem**: Response time baseline hardcoded at 1.0s vs actual 38s
- **Implementation**: Calculate baseline from rolling average of actual requests
- **Result**: Self-tuning degradation detection for any API endpoint

#### **P1.7 Complete Config Propagation for All Evolution Parameters**
- **Problem**: Only `retrieval.default_top_k` properly propagated, 15+ other parameters fail
- **Implementation**: Add missing fields to config.py dataclasses, update ConfigManager.update() calls
- **Result**: All evolvable parameters propagate through ConfigManager

#### **P1.8 Smart Index Type Management on Server Start**
- **Problem**: User converts to `flat` index, then changes `.env` to `ivf`, server ignores .env
- **Implementation**: Detect index type mismatch, offer conversion options
- **Result**: User choice between convert, keep, or abort on mismatch

---

## 5. REQUIRED LOGGING (CURRENT ISSUES)

### **🟡 LOGGING STATUS - PARTIALLY WORKING**

**Retrieval Quality Logging**: 
- ❌ **BROKEN** - Type errors prevent logging execution
- Error: `Cannot access attribute "retrieval" for class "MemorySystemConfig"`

**Management Operations Logging**:
- ✅ **WORKING** - FAISS vector operations functional
- Status: All store/retrieve operations logging correctly

**Evolution Cycle Logging**:
- ✅ Working - Evolution system logs operational

**FAISS Memory Operations Logging**:
- ✅ **WORKING** - All vector store operations logging
- Status: Storage, search, count, rebuild all functional

**Config Propagation Logs**:
- ✅ Working - Config updates logged correctly

---

## 6. Validation Checklist (CRITICAL FAILURES)

### **🟡 DEBUGGING STATUS - CORE SYSTEMS WORKING**

**Current Status**: Major components operational, remaining type errors

- ❌ **Retrieval precision/recall logging** - Type errors block execution
- ✅ **Memory storage operations** - FAISS integration fully functional  
- ✅ **Semantic search functionality** - Vector store operational
- ✅ **Memory management operations** - Store/retrieve working correctly
- ✅ **Evolution fitness varies** - Still working correctly
- ❌ **Type safety throughout system** - MemorySystem config errors remain
- ✅ **API endpoint stability** - Core memory endpoints responding correctly
- ✅ **FAISS vector operations** - All storage/search/rebuild working
- 🟡 **Dashboard metrics collection** - Import handling needs resolution

### **🎯 CURRENT DEBUGGING PRIORITIES**

1. **Resolve Type Errors** - P0.17 (Remaining: MemorySystemConfig access issues)  
2. **Fix Import Handling** - P0.19 (Dashboard metrics collection improvements)

**PROGRESS**: P0.18 (FAISS Integration) **COMPLETED** - Vector storage now fully operational with proper ID generation and memory count incrementing.

---

## 7. Agent Execution Notes

- **System Status**: **CRITICAL FAILURES** - Multiple broken components
- **Focus Areas**: Fix P0.17-P0.19 type errors and FAISS integration immediately
- **Stability Priority**: Vector storage completely broken, memory system inaccessible
- **Observability**: Logging broken due to type errors, metrics collection failing
- **Architecture**: Web layer responds but business logic components failing silently

### **DEBUGGING STRATEGY**
1. **Isolate component failures** - Test each module independently
2. **Fix type annotations** - Resolve MemorySystemConfig attribute access issues  
3. **Repair FAISS integration** - Fix vector store initialization and API calls
4. **Validate import handling** - Fix optional dependency management in metrics collector
5. **Test end-to-end flow** - Verify fixes work together without breaking other components

### **WARNING**: Do not proceed with any feature development until core component failures are resolved. System is in a broken state despite apparent API availability.

---

## 7. NEXT DEBUGGING PRIORITIES

### **🟡 P0.17-P0.19 REMAINING TASKS**

The core memory system (P0.18 FAISS integration) is now **FULLY FUNCTIONAL**. Remaining issues are type safety and import handling:

#### **P0.17 - MemorySystemConfig Type Errors**
- **Impact**: Retrieval quality logging broken
- **Approach**: Fix config dataclass field access patterns
- **Files**: `src/memevolve/memory_system.py` (lines 871, 881, 1181)

#### **P0.19 - Metrics Collector Import Handling**
- **Impact**: Dashboard scoring systems unavailable  
- **Approach**: Fix optional dependency management
- **Files**: `src/memevolve/utils/endpoint_metrics_collector.py` (lines 139-141)

### **🎯 POST-DEBUGGING DEVELOPMENT FOCUS**

Once P0.17-P0.19 are resolved, priority shifts to:

1. **P1.1 Memory Quality Improvements** - Reduce generic memories
2. **P1.2 Storage Validation** - Prevent invalid memory storage
3. **P1.3 Scoring System Unification** - Consolidate duplicate scoring logic
4. **P1.4 Memory Filtering** - Relevance-based injection filtering
5. **Evolution Parameter Management** - Complete config propagation

---

## 8. IMMEDIATE DEBUGGING TASKS

### **🟡 PRIORITY 1 - REMAINING COMPONENT FIXES**

#### **P0.17 Debug Memory System Type Errors**
**Files**: `src/memevolve/memory_system.py`
**Lines**: 871, 881, 1181
**Actions**:
1. Fix `MemorySystemConfig` attribute access - missing `retrieval` field
2. Resolve `Unknown | int | None` type assignment to `top_k` parameter
3. Fix `strategy` attribute access on `None` object
4. Add proper type hints and null checks

#### **P0.19 Debug Endpoint Metrics Collector Imports**
**Files**: `src/memevolve/utils/endpoint_metrics_collector.py`
**Lines**: 139-141  
**Actions**:
1. Fix optional import handling for `MemoryScorer`, `ResponseScorer`, `TokenAnalyzer`
2. Add proper fallback logic when scoring systems unavailable
3. Resolve "possibly unbound" type errors

#### **P0.18 ✅ COMPLETED - FAISS Integration**
**Files**: `src/memevolve/components/store/vector_store.py`  
**Status**: FULLY RESOLVED
**Fixed**:
- FAISS method signatures and null safety
- ID generation bug (memory count stuck at 212)
- IVF index training during rebuild operations
- All vector storage, search, and count operations

### **🔴 PRIORITY 2 - VALIDATION**

After fixing P0.17-P0.19:
1. **Test type safety** - Run linting to ensure no new type errors
2. **Test dashboard metrics** - Validate real data collection works
3. **Run integration tests** - Ensure component fixes don't break other systems
4. **Test memory system logging** - Verify retrieval logging works correctly
5. **Validate config propagation** - Check all MemorySystemConfig fields accessible

---

**Bottom line**: MemEvolve-API is in a **BROKEN STATE** with critical component failures. Previous endpoint tests were misleading - the web layer responded but underlying business logic components are failing due to type errors and FAISS integration issues. **Immediate debugging required before any other development.**

---

**Files Modified in Latest Session**:
- `src/memevolve/api/server.py` — Delayed evolution startup
- `src/memevolve/components/manage/simple_strategy.py` — Cached health metrics
- `src/memevolve/components/store/vector_store.py` — Health metrics cache + FAISS integration fixes  
- `src/memevolve/api/routes.py` — Removed mock fallbacks, real data collection

**Commits Applied**:
- `7cc191d` - Fix: P0.10 - Replace hardcoded fitness fallback with real trajectory testing
- `fa96f23` - CRITICAL FIX: P0.10 - Use genotype-specific parameters for trajectory testing  
- `f841935` - CRITICAL FIX: P0.10 - Fix memory endpoint URL causing 404 errors
- `247f00b` - FIX: P0.10 - Add timeout handling for trajectory testing
- `8482778` - FIX: Update dev_tasks.md to debugging focus + server architecture
- `049bda9` - CRITICAL FIX: P0.18 - Fix FAISS vector store integration errors
- `2abfa6e` - FIX: P0.18 - Fix IVF index training during rebuild operation
- `106c56b` - CRITICAL FIX: P0.18 - Fix memory count stuck at 212