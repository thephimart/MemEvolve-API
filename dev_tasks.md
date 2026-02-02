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
3. **Genotype Verification**: Validate all evolution mutations propagate through ConfigManager

### **NEXT SESSION**
1. **P0.28**: Complete dashboard API endpoints
2. **P1.3**: Unify quality scoring systems (643 → ~350 lines)

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
- ✅ **P1.2**: Memory relevance filtering implemented (6060f5d)
- ✅ **SEMANTIC SCORING**: Vector normalization (8a87f6b)

### **RECENT COMMITS**
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