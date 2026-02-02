# MemEvolve‑API — Development Tasks (Agent Runbook)

> **Purpose**: A compressed, non‑overlapping, execution‑ready task file to pair with `AGENTS.md`. This document focuses on **what is broken**, **what must be logged**, **hard metrics for validation**, and a **prioritized task list** so an agent can start immediately.

---

## 1. Current System Snapshot (Post-Fix Baseline)

**Status**: ✅ **ALL CRITICAL ISSUES RESOLVED** (Feb 2, 2026)

- **Version**: v2.0.0 (master) + Server architecture fixes
- **API Endpoints**: Fully operational with no blocking
- **Evolution System**: Production ready with real metrics
- **Memory System**: 212 experiences stored and searchable
- **Server Architecture**: Fixed - all endpoints respond <0.01s except LLM processing

**Critical Fixes Applied**:
- ✅ **P0.10**: Real endpoint trajectory testing implemented
- ✅ **P0.11**: Environment variable override capability restored  
- ✅ **P0.12**: Hardcoded 1024 limit removed
- ✅ **P0.13**: Storage integrity failure fixed (100% success rate)
- ✅ **Server Architecture**: Cached health metrics, delayed evolution startup, real dashboard data

**System Status**: **PRODUCTION READY**

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

### **Endpoint Performance Tests (Feb 2, 2026)**

| Endpoint | Response Time | Status | Functionality |
|----------|---------------|--------|---------------|
| **`/health`** | **0.006s** | ✅ | System healthy |
| **`/memory/stats`** | **0.012s** | ✅ | Cached metrics working |
| **`/memory/search`** | **8.324s** | ✅ | Semantic search operational |
| **`/dashboard-data`** | **0.003s** | ✅ | **Real data collector working** |
| **`/evolution/status`** | **0.004s** | ✅ | Evolution system ready |
| **`/v1/chat/completions`** | **34.48s** | ✅ | Full memory integration |

### **System Health Metrics**
- **Storage Integrity**: 100% success rate (67/67 operations successful)
- **Memory Count**: 212 experiences stored and searchable
- **Evolution History**: 1 cycle completed, 468 requests analyzed
- **API Performance**: 20.5s upstream, 32ms memory retrieval
- **Business Intelligence**: Token efficiency analysis and ROI monitoring

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

#### **P0.18 Fix Vector Store FAISS Integration Errors**
- **Location**: `vector_store.py:181,309,313,335`
- **Errors**:
  - Argument missing for parameter "n" 
  - "add" is not a known attribute of "None"
  - "ntotal" is not a known attribute of "None"
  - "search" is not a known attribute of "None"
- **Impact**: Vector storage completely broken, semantic search fails
- **Root Cause**: FAISS index initialization and API call mismatches

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

### **🔴 BROKEN LOGGING - NEEDS IMMEDIATE FIX**

**Retrieval Quality Logging**: 
- ❌ **BROKEN** - Type errors prevent logging execution
- Error: `Cannot access attribute "retrieval" for class "MemorySystemConfig"`

**Management Operations Logging**:
- ❌ **BROKEN** - Vector store failures prevent memory operations
- Error: `"add" is not a known attribute of "None"`

**Evolution Cycle Logging**:
- ✅ Working - Evolution system logs operational

**Slow Retrieval Warnings**:
- ❌ **BROKEN** - FAISS integration broken, no retrieval metrics
- Error: `"search" is not a known attribute of "None"`

**Config Propagation Logs**:
- ✅ Working - Config updates logged correctly

---

## 6. Validation Checklist (CRITICAL FAILURES)

### **🔴 IMMEDIATE DEBUGGING REQUIRED**

**Current Status**: Multiple component failures preventing normal operation

- ❌ **Retrieval precision/recall logging** - Type errors block execution
- ❌ **Memory storage operations** - FAISS integration completely broken  
- ❌ **Dashboard metrics collection** - Import errors prevent data flow
- ❌ **Semantic search functionality** - Vector store non-functional
- ✅ **Evolution fitness varies** - Still working correctly
- ❌ **Memory management operations** - Cannot store or retrieve memories
- ❌ **Type safety throughout system** - Multiple runtime errors
- ❌ **API endpoint stability** - Underlying component failures affect surface APIs
- ✅ **Server architecture** - Web layer responding but business logic failing
- ❌ **Real dashboard data collection** - Broken due to import/FAISS issues

### **🎯 IMMEDIATE DEBUGGING PRIORITIES**

1. **Fix FAISS Integration** - P0.18 (Critical: vector storage broken)
2. **Resolve Type Errors** - P0.17 (Critical: memory system inaccessible)  
3. **Fix Import Handling** - P0.19 (Critical: dashboard data unavailable)

**NOTE**: Previous endpoint response tests were misleading - web layer responded but underlying components are failing silently.

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

## 8. IMMEDIATE DEBUGGING TASKS

### **🔴 PRIORITY 1 - FIX BROKEN COMPONENTS**

#### **P0.17 Debug Memory System Type Errors**
**Files**: `src/memevolve/memory_system.py`
**Lines**: 871, 881, 1181
**Actions**:
1. Fix `MemorySystemConfig` attribute access - missing `retrieval` field
2. Resolve `Unknown | int | None` type assignment to `top_k` parameter
3. Fix `strategy` attribute access on `None` object
4. Add proper type hints and null checks

#### **P0.18 Debug FAISS Vector Store Integration**
**Files**: `src/memevolve/components/store/vector_store.py`  
**Lines**: 181, 309, 313, 335
**Actions**:
1. Fix missing parameter "n" in FAISS API calls
2. Resolve `None` index object issues - initialization problems
3. Fix "add", "search", "ntotal" attribute errors on None objects
4. Add proper FAISS index validation and error handling

#### **P0.19 Debug Endpoint Metrics Collector Imports**
**Files**: `src/memevolve/utils/endpoint_metrics_collector.py`
**Lines**: 139-141  
**Actions**:
1. Fix optional import handling for `MemoryScorer`, `ResponseScorer`, `TokenAnalyzer`
2. Add proper fallback logic when scoring systems unavailable
3. Resolve "possibly unbound" type errors

### **🔴 PRIORITY 2 - VALIDATION**

After fixing P0.17-P0.19:
1. **Test memory storage** - Verify vector operations work end-to-end
2. **Test semantic search** - Confirm retrieval functionality restored
3. **Test dashboard metrics** - Validate real data collection works
4. **Run integration tests** - Ensure component fixes don't break other systems
5. **Check type safety** - Run linting to ensure no new type errors

---

**Bottom line**: MemEvolve-API is in a **BROKEN STATE** with critical component failures. Previous endpoint tests were misleading - the web layer responded but underlying business logic components are failing due to type errors and FAISS integration issues. **Immediate debugging required before any other development.**

---

**Files Modified in Latest Session**:
- `src/memevolve/api/server.py` — Delayed evolution startup
- `src/memevolve/components/manage/simple_strategy.py` — Cached health metrics
- `src/memevolve/components/store/vector_store.py` — Health metrics cache infrastructure  
- `src/memevolve/api/routes.py` — Removed mock fallbacks, real data collection

**Commits Applied**:
- `7cc191d` - Fix: P0.10 - Replace hardcoded fitness fallback with real trajectory testing
- `fa96f23` - CRITICAL FIX: P0.10 - Use genotype-specific parameters for trajectory testing  
- `f841935` - CRITICAL FIX: P0.10 - Fix memory endpoint URL causing 404 errors
- `247f00b` - FIX: P0.10 - Add timeout handling for trajectory testing