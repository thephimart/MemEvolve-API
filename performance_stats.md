# MemEvolve-API Performance Statistics

> **Purpose**: Baseline performance metrics from current system state (February 10-11, 2026) for comparison during test iterations.

---

## 📊 **Encoding Pipeline Performance**

### **Success Rates**
- **Total Encoding Attempts**: 46 (40 successful, 6 failed)
- **Success Rate**: **87.0%** (↑ from previous 75% baseline)
- **Failure Rate**: **13.0%** (↓ from previous 25% baseline)
- **Primary Failure Mode**: JSON parsing errors from malformed LLM responses

### **Encoding Latency**
- **Average Encoding Time**: **13.34 seconds** (per successful experience)
- **Range**: 10.17s - 25.34s
- **Performance Issue**: 13+ second latency impacts real-time response flow
- **Target**: <2.0s per encoding for production use

### **Recent Fixes Applied**
- **Reasoning Contamination**: ✅ Fixed - Removed reasoning_content from encoder input
- **Schema Rigidity**: ✅ Fixed - Flexible 1-4 field acceptance instead of requiring all 4
- **Configuration Issues**: ✅ Fixed - Flexible encoding prompts and proper config loading

---

## 🔍 **Memory Retrieval Performance**

### **Retrieval Statistics**
- **Total Retrieval Attempts**: 39
- **Successful Memory Injections**: 17 (43.6% of attempts)
- **Failed Injections**: 22 (56.4% below threshold)
- **Memory Threshold**: 0.5 (current setting)
- **Average Retrieved Memories**: 1.5 per query

### **Memory Relevance Scores**
- **Typical Score Range**: 0.045 - 0.505
- **Most Common Score**: ~0.050 (far below 0.5 threshold)
- **Highest Score**: 0.505 (just above threshold)
- **Relevance Issue**: Most memories scored ~0.05, far below 0.5 threshold

### **Memory Store Status**
- **Memory Store Size**: 440KB (vector.index + vector.data)
- **Stored Memory Units**: ~40 successful encodings
- **Retrieval Strategy**: Hybrid (semantic + keyword)
- **Issue**: Low memory volume may impact semantic similarity scores

---

## ⚡ **API Request Pipeline Performance**

### **Request Processing**
- **Upstream API Response Time**: 75.4 seconds average
- **Memory System Overhead**: 43.7ms average (excellent)
- **Total Request Time**: ~2.16 seconds (excluding upstream)
- **Memory Retrieval**: Sub-50ms (very fast)

### **Token Usage**
- **Query Tokens**: ~16 tokens average
- **Response Tokens**: ~313 tokens average (upstream API)
- **Memory Tokens**: ~24 tokens average
- **Net Token Impact**: -610 tokens (memory adds context, not savings)

---

## 📈 **System Health Metrics**

### **Logging System**
- **Log Volume**: 616 lines in memory_system.log
- **Log Quality**: Clean, structured, properly leveled
- **Error Rate**: Low (6 encoding failures out of 46 attempts)
- **Visibility**: Good debug information available

### **Configuration Status**
- **Services Configured**: Upstream, Memory, Embedding endpoints
- **Token Limits**: 8192 (memory), properly resolved
- **API Endpoints**: http://192.168.1.61:11433 (LLM), http://192.168.1.61:11435 (embeddings)
- **Environment**: .venv activated, all dependencies loaded

### **Repository State**
- **Branch**: master (clean working tree)
- **Version**: v2.1.0 (development status)
- **Documentation**: Consistent across all files
- **Code Quality**: All changes are surgical and backwards-compatible

---

## 🎯 **Performance Targets for Test Iterations**

### **Primary Targets**
1. **Encoding Success Rate**: 87% → **95%+** (eliminate JSON parsing failures)
2. **Encoding Latency**: 13.34s → **<2.0s** (10x improvement)
3. **Memory Relevance**: 0.05 → **0.5+** average (threshold optimization)
4. **Memory Injection Rate**: 43.6% → **70%+** of retrieved memories

### **Secondary Targets**
1. **Retrieval Latency**: Maintain <50ms (currently excellent)
2. **Memory Store Growth**: 40 → 1000+ memory units
3. **Semantic Similarity**: Improve with larger memory corpus
4. **Upstream Latency**: External dependency (no direct control)

---

## 🔧 **Test Iteration Guidelines**

### **Before Running Tests**
1. **Log Scrub**: Clear `./logs/` directory to start fresh
2. **Memory Backup**: Optionally backup `./data/memory/` directory
3. **Baseline Recording**: Note current metrics from this file
4. **Configuration Freeze**: Ensure .env settings are stable

### **During Tests**
1. **Monitor Encoding Success**: Track `Encoded experience` vs `Error encoding experience`
2. **Latency Tracking**: Record `in X.XXXs` from encoding logs
3. **Memory Relevance**: Monitor `score=X.XXX` values in retrieval logs
4. **Injection Rate**: Track `Injected X relevant memories` vs total attempts

### **After Tests**
1. **Update Performance Stats**: Compare against this baseline
2. **Regression Check**: Ensure no metric degradation
3. **Success Documentation**: Record improvements in this file
4. **Next Iteration Planning**: Identify remaining performance gaps

---

## 📋 **Data Sources Used**

### **Log Files Analyzed**
- `./logs/memory_system.log` (616 lines)
- `./logs/api/enhanced_middleware.log` (encoding/retrieval metrics)
- `./data/endpoint_metrics/request_pipeline.json` (latency data)

### **Memory Store Analysis**
- `./data/memory/vector.index` (421KB)
- `./data/memory/vector.data` (14KB)
- Total stored experiences: ~40 units

### **Time Period**
- **Data Range**: February 10-11, 2026
- **System State**: Post-encoding-fixes, pre-evolution-analysis
- **Configuration**: v2.1.0 development build

---

## 🏁 **Baseline Summary**

**Current System Status**: 🟢 **ENCODING PIPELINE OPTIMIZED**
- Encoding success: 87% (improved from 75%)
- Memory retrieval functional but low relevance
- Latency acceptable for retrieval, high for encoding
- System stable and ready for evolution work

**Next Session Priorities**:
1. Test encoding pipeline end-to-end (verify 95%+ success rate)
2. Begin evolution system analysis (`src/memevolve/evolution/`)
3. Optimize memory relevance scoring and thresholds

**This baseline serves as the reference point for all performance improvements during test iterations.**