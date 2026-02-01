# MemEvolve‑API — Development Tasks (Agent Runbook)

> **Purpose**: A compressed, non‑overlapping, execution‑ready task file to pair with `AGENTS.md`. This document focuses on **what is broken**, **what must be logged**, **hard metrics for validation**, and a **prioritized task list** so an agent can start immediately.

---

## 1. Current System Snapshot (Baseline for Comparison)

These are **hard data points** to compare against future runs.

- **Version**: v2.0.0 (master)
- **Runs analyzed**: 290
- **Evolution generations**: 14 tracked
- **Memories**: 928 total
  - Clean: 926
  - Corrupted (null): 2 (unit_157, unit_949)
  - Fallback chunks: 5 (0.5%)
- **Retrieval accuracy**: ~40% (target ≥ 60%)
- **Avg retrieval score**: 0.490 (min 0.052 / max 0.864)
- **Retrieval latency**:
  - Normal: 0.2–0.3s
  - Anomaly: 8.6s (single event)
- **Evolution fitness**: **stuck at 0.687** (Gen 5–12)
- **Critical observation**: `retrieval_precision`, `retrieval_recall`, `response_quality_score` are **always 0.0**

This snapshot must be reproducible until fixes are applied.

---

## 2. Root Cause Summary (Why Evolution Is Blocked)

Evolution is not learning because **there is no feedback loop**:

1. Retrieval quality metrics are never populated
2. Auto‑management (dedup / forget) never executes
3. Evolution config updates never reach runtime

As a result:
- All genotypes are evaluated with identical retrieval scores
- Fitness plateaus regardless of configuration changes

---

## 3. CRITICAL PRIORITIES (Execution Order)

### **P0 — BLOCKING (Must Fix First)**

#### P0.1 Fix Retrieval Quality Feedback Loop

**Problem**
- Middleware does not pass precision/recall/quality to evolution manager

**Required Outcome**
- Evolution receives **non‑zero**, **varying** retrieval quality metrics per run

**Implementation Tasks**
- Compute retrieval precision / recall in middleware after retrieval
- Call:
  ```python
  evolution_manager.record_memory_retrieval(
      retrieval_time,
      success,
      memory_count,
      precision,
      recall,
      quality
  )
  ```

**Validation Metrics**
- Logs show non‑zero precision/recall values
- Evolution fitness varies between generations

---

#### P0.2 Enable Memory Management Execution

**Problem**
- Deduplication and forgetting are never invoked

**Required Outcome**
- Storage no longer grows unbounded
- Duplicate and low‑value memories are removed

**Implementation Tasks**
- Call `deduplicate()` after encoding batches
- Call `apply_forgetting()` periodically (e.g. every N requests)
- Ensure `enable_auto_management` flag actually gates execution

**Validation Metrics**
- Logs confirm management calls
- Memory count stabilizes or decreases

---

#### P0.3 Fix Config Propagation (Evolution → Runtime)

**Problem**
- Evolved values (e.g. `default_top_k = 7`) never reach retrieval code
- EvolutionManager creates separate ConfigManager instance (line 76)
- Middleware reads from different config object than EvolutionManager updates
- `ConfigManager.update()` receives dict instead of dot notation

**Required Outcome**
- Runtime retrieval reflects evolved config within one evolution cycle
- Single shared ConfigManager instance across all components
- See AGENTS.md "Configuration Architecture Rules" for priority hierarchy

**Implementation Tasks**
1. ✅ **Fix ConfigManager sharing** (evolution_manager.py:76):
   - Pass shared ConfigManager from server.py to EvolutionManager
   - Remove `config_manager or ConfigManager()` fallback
   
2. ✅ **Fix update syntax** (evolution_manager.py:1344):
   - ~~Change: `retrieval={'default_top_k': 3}`~~
   - ~~To: `retrieval__default_top_k=3` (double underscore for dot notation)~~
   - **CORRECTED**: Use `**{'retrieval.default_top_k': value}` (actual dot notation)
   - **Note**: Double underscores don't work - ConfigManager splits on dots, not underscores
   
3. ✅ **Fix middleware config access** (enhanced_middleware.py:659):
   - Change: `return self.config.retrieval.default_top_k`
   - To: `return self.config_manager.get('retrieval.default_top_k')`
   
4. ✅ **Ensure server.py passes shared ConfigManager**:
   - Create single ConfigManager instance in lifespan
   - Pass to both EvolutionManager and middleware

**Issues Discovered During Testing**
- **Critical Bug**: Used `retrieval__default_top_k` (double underscore) instead of `retrieval.default_top_k`
- **Impact**: All 20 evolution cycles failed with "Invalid config path" errors
- **Result**: Evolution stuck at Gen 3, fitness unchanged at 0.837, no config propagation
- **Fix Applied**: Changed to proper dot notation via `**{'retrieval.default_top_k': value}`

**Validation Metrics**
- Logs show runtime `top_k` changing (e.g. 5 → 3 → 5) matching evolution_state.json
- Middleware logs confirm: `Retrieval limit (top_k): X` changes after evolution cycles
- No "Invalid config path" errors in server console
- Evolution advances past Gen 3 with varying fitness scores
- **Status**: Code fixed, awaiting server restart to validate

---

### **P1 — HIGH (After P0 Is Fixed)**

#### P1.1 Eliminate Generic / Meta Memories at Encode Time

**Problem**
- Q&A formatted input causes meta‑descriptions (~19% generic)

**Implementation Tasks**
- Send **assistant response only** to encoder
- Add rule‑based summarization before LLM fallback

**Targets**
- Generic/vague memories < 10%
- Actionable insights > 80%

---

#### P1.2 Prevent Invalid Memory Storage

**Problem**
- Null content and fallback chunks still stored

**Implementation Tasks**
- Add hard validation: no `null` content, no fallback chunks
- Delete known corrupted units:
  - null: `unit_157`, `unit_949`
  - fallback: `unit_300`, `unit_317`, `unit_339`, `unit_348`, `unit_684`

---

#### P1.3 Unify Quality Scoring Systems

**Problem**
- Three separate scoring systems with overlapping functionality:
  - `ResponseQualityScorer` (utils/quality_scorer.py) - 483 lines, hardcoded weights
  - `ResponseScorer` (evaluation/response_scorer.py) - 80 lines, untested
  - `MemoryScorer` (evaluation/memory_scorer.py) - 80 lines, partial config support
- Duplicate relevance/coherence calculations
- No unified interface or exports (evaluation/ lacks __init__.py)
- ResponseQualityScorer has unused imports (json, time, Optional, Union, datetime)
- Evolution never receives response quality scores (record_response_quality never called)

**Required Outcome**
- Single `ScoringEngine` class in evaluation/scoring_engine.py
- All scoring weights/thresholds in centralized config (per AGENTS.md Configuration Architecture Rules)
- Proper exports via evaluation/__init__.py
- Response quality scores feed into evolution fitness
- 643 lines → ~350 lines, zero lint errors

**Implementation Tasks**
1. Create unified `ScoringEngine` class:
   - Merge ResponseQualityScorer reasoning/direct model parity logic
   - Merge ResponseScorer relevance/coherence/memory_utilization logic
   - Merge MemoryScorer cosine similarity/text overlap logic
   - Add `score_response()`, `score_memory_relevance()`, `score_quality()` methods
2. Create `evaluation/__init__.py` with proper exports
3. Add `ScoringConfig` dataclass to utils/config.py (per AGENTS.md: config.py is single source of truth):
   - Scoring weights (completeness, actionability, clarity, etc.)
   - Thresholds (relevance, min_quality)
   - Bias correction settings
   - Full stop words list (100+ words)
4. Update all callers:
   - enhanced_middleware.py: use ScoringEngine, call record_response_quality()
   - endpoint_metrics_collector.py: use ScoringEngine
   - routes.py: use ScoringEngine
   - utils/__init__.py: remove ResponseQualityScorer export
5. Delete old files:
   - utils/quality_scorer.py
   - evaluation/response_scorer.py
   - evaluation/memory_scorer.py
6. Create tests/test_scoring_engine.py (port 410 lines from test_quality_scorer.py)

**Validation Metrics**
- All scoring tests pass (test_scoring_engine.py)
- Zero lint errors in new scoring module
- Logs show response quality scores feeding into evolution:
  ```
  EVOLUTION | response_quality=0.72 retrieval_precision=0.65
  ```
- Config changes propagate to scoring behavior within one cycle

---

#### P1.4 Filter Irrelevant Memories from Injection

**Problem**
- All retrieved memories are injected regardless of relevance score
- Memories with score ≤ 0.5 (irrelevant) waste tokens and may confuse the LLM
- Log shows "Injected 3 memories" but doesn't distinguish relevant vs total retrieved

**Current Behavior**
```python
# Line 315-319: All memories injected, no filtering
if memories:
    enhanced_messages = self._inject_memories(messages, memories)
    request_metrics.memories_injected = len(memories)
    logger.info(f"Injected {len(memories)} memories into request (limit: {retrieval_limit})")
    # Shows: "Injected 3 memories into request (limit: 3)"
    # Both numbers are the same - no distinction between retrieved vs relevant
```

**Required Outcome**
- Only memories with score > 0.5 (relevance threshold) are injected
- Log format: "Injected X relevant memories (retrieved: Y, limit: Z)"
- Example: "Injected 1 relevant memories (retrieved: 3, limit: 3)"

**Implementation Tasks**
1. **Filter memories before injection** (enhanced_middleware.py:315-319):
   ```python
   # Filter to relevant memories only (score > 0.5)
   relevant_memories = [
       mem for mem, score in zip(memories, memory_relevance_scores)
       if score > 0.5
   ]
   
   if relevant_memories:
       enhanced_messages = self._inject_memories(messages, relevant_memories)
       request_metrics.memories_injected = len(relevant_memories)
       logger.info(
           f"Injected {len(relevant_memories)} relevant memories "
           f"(retrieved: {len(memories)}, limit: {retrieval_limit})"
       )
   ```

2. **Update metrics tracking**:
   - Track both `memories_retrieved` and `memories_injected` (relevant only)
   - Update evolution metrics to use relevant count for precision calculation

3. **Update _inject_memories call**:
   - Pass `relevant_memories` instead of `memories`
   - Ensure _inject_memories handles empty list gracefully

**Validation Metrics**
- Logs show distinction: "Injected X relevant memories (retrieved: Y, limit: Z)"
- When precision=0.00, log shows: "Injected 0 relevant memories (retrieved: 3, limit: 3)"
- When precision=0.67, log shows: "Injected 2 relevant memories (retrieved: 3, limit: 3)"
- Evolution precision metric stays the same (still calculated on all retrieved)

---

#### P0.4 Remove Time-Based Evolution Trigger

**Problem**
- Evolution triggers on time (24 hours) even when server idle
- User wants iteration-only triggers for idle server compatibility
- Sleep between generations should remain (30-min cooldown)

**Current Triggers** (evolution_manager.py:612-669)
1. ✅ Request count (50 requests since startup) — **KEEP**
2. ✅ Performance degradation (>20% slower) — **KEEP**
3. ✅ Fitness plateau (5 gens with similar fitness) — **KEEP**
4. ❌ **Time-based (24h since last evolution)** — **REMOVE**

**Current Behavior**
```
Server idle for 24 hours → Evolution triggers anyway
Active usage: 50 requests → Evolution triggers
Between generations: 30 min sleep (cycle_seconds=1800)
```

**Required Behavior**
```
Server idle for 24 hours → No evolution (wait for iterations)
Active usage: 50 requests → Evolution triggers
Between generations: 30 min sleep (unchanged)
```

**Implementation Tasks**
1. **Remove time-based trigger** (evolution_manager.py:656-663):
   ```python
   # REMOVE this entire block:
   hours_threshold = self.config.auto_evolution.hours
   if self.last_evolution_time > 0:
       hours_since_last = (time.time() - self.last_evolution_time) / 3600
       if hours_since_last >= hours_threshold:
           triggers.append(f"time_based_trigger...")
   ```

2. **Keep sleep between generations** (evolution_manager.py:1005):
   ```python
   # Keep as-is:
   self.stop_event.wait(float(self.evolution_cycle_seconds))
   ```

3. **Update .env.example**:
   ```bash
   # Change from:
   MEMEVOLVE_AUTO_EVOLUTION_HOURS=24
   
   # To:
   MEMEVOLVE_AUTO_EVOLUTION_HOURS=0  # 0 = disabled, evolution only on iterations
   ```

4. **Update config.py** to treat 0 as "disabled" for hours trigger

**Validation Metrics**
- Server idle 24h: No evolution triggers, no log activity
- 50 active requests: Evolution triggers immediately
- Between generations: Still waits 30 minutes (cycle_seconds)
- Config: `MEMEVOLVE_AUTO_EVOLUTION_HOURS=0` in .env disables time trigger

---

#### P0.5 Dynamic Performance Baseline from Actual Data

**Problem**
- Response time baseline hardcoded at 1.0s (line 638)
- Doesn't match actual API performance (38s vs 1s)
- Different API endpoints need different baselines
- Manual configuration required for each deployment

**Required Outcome**
- Baseline calculated from actual iteration data (rolling average)
- Self-tuning to any API endpoint speed
- No hardcoded or .env configuration needed
- Degradation detected relative to system's "normal" performance

**Implementation Tasks**
1. **Add baseline tracking** (evolution_manager.py:EvolutionMetrics):
   - Add `response_times_window: List[float]` field
   - Add `response_times_baseline: float` field
   - Add `baseline_window_size: int = 100` constant
   - Add `update_response_time_baseline()` method

2. **Update degradation detection** (evolution_manager.py:704-722):
   - Replace: `baseline_time = 1.0`
   - Use: `baseline = self.metrics.response_times_baseline`
   - Add minimum data check (need 10+ samples)
   - Use median for robustness against outliers

3. **Integrate with request tracking**:
   - Call `update_response_time_baseline()` in `record_api_request()`
   - Update baseline with each new request
   - Baseline established after first 100 requests

4. **Update logging**:
   - Log when baseline is established
   - Log baseline value periodically
   - Log degradation as "% above baseline"

**Validation Metrics**
- First 10 requests: No degradation triggers (insufficient data)
- Requests 10-100: Baseline established from rolling median
- Request 101+: Degradation triggers if >20% above rolling baseline
- Baseline adapts: If API slows down permanently, new baseline established
- Logs show: "Baseline established: 38.5s from last 100 requests"
- Logs show: "Degradation: 45.2s vs baseline 38.5s (+17.4%)"

**Example Behavior**
```
Requests 1-10:   No baseline yet (collecting data)
Requests 11-100: Baseline = median of last 10-100 response times
Request 101+:    Baseline = median of last 100 response times
                 Degradation if current > baseline * 1.2

API gets slower: Baseline gradually adjusts upward over 100 requests
Temporary spike: Degradation detected (current > 1.2x baseline)
```

---

#### P0.6 Complete Config Propagation for All Evolution Parameters

**Problem**
- Only `retrieval.default_top_k` is properly propagated via ConfigManager
- 15+ other evolution parameters fail with "Invalid config path" errors
- Evolution cannot test different configurations for similarity_threshold, weights, dedup settings, etc.
- All evolution cycles fail because encoder.max_tokens path doesn't exist

**Root Cause**
- Config dataclasses missing fields that genotype expects
- ConfigManager.update() only updates 2 paths instead of 15+
- Middleware doesn't read most config values from ConfigManager

**Required Outcome**
- All evolvable parameters propagate through ConfigManager
- Evolution can mutate and test any configuration
- Zero "Invalid config path" errors
- Complete evolution cycles with varying parameters

**Implementation Tasks**

1. **Add missing fields to config.py dataclasses:**
   - RetrievalConfig: similarity_threshold, enable_filters, semantic_cache_enabled, keyword_case_sensitive
   - EncoderConfig: max_tokens, batch_size, temperature, llm_model
   - ManagementConfig: deduplicate_enabled, forgetting_percentage, consolidate_enabled, consolidate_min_units, prune_max_age_days, prune_max_count

2. **Update ConfigManager.update() call** (evolution_manager.py:1342):
   - Map all genotype fields to ConfigManager paths
   - 15+ parameter updates in single call
   - Group by component (retrieval.*, encoder.*, management.*)

3. **Update middleware config access** (enhanced_middleware.py):
   - Read all retrieval parameters from ConfigManager
   - Add helper method to get live retrieval config
   - Ensure strategy creation uses ConfigManager values

4. **Update memory system reconfiguration** (memory_system.py):
   - Ensure reconfigure_component() uses ConfigManager state
   - Pass live config values when creating new strategies

**Parameters to Propagate**

| Component | Parameters |
|-----------|-----------|
| Retrieval | default_top_k, similarity_threshold, semantic_weight, keyword_weight, enable_filters, semantic_cache_enabled, keyword_case_sensitive |
| Encoder | max_tokens, batch_size, temperature, llm_model |
| Management | deduplicate_enabled, deduplicate_threshold, forgetting_strategy, forgetting_percentage, consolidate_enabled, consolidate_min_units, auto_prune_threshold, prune_max_age_days, prune_max_count |

**Validation Metrics**
- Evolution cycles 0-19 complete without errors
- Logs show parameters changing: "similarity_threshold: 0.7 → 0.5"
- Logs show weights changing: "semantic_weight: 0.7 → 0.6"
- Logs show management changing: "deduplicate_enabled: true → false"
- Fitness scores vary based on actual parameter performance
- No "Invalid config path" errors in console

---

### **P2 — MEDIUM / LOW**

- Investigate slow retrievals (>1s, >5s)
- Improve semantic retrieval precision
- Add stronger duplicate similarity thresholds

---

## 4. REQUIRED LOGGING (NON‑NEGOTIABLE)

Logging must exist **after fixes** or the task is incomplete.

### 4.1 Retrieval Quality Logging (CRITICAL)

Must log per retrieval:
- precision
- recall
- quality score

Example:
```
RETRIEVAL_QUALITY | precision=0.67 recall=0.50 quality=0.61 memories=5
```

---

### 4.2 Management Operations Logging (CRITICAL)

Must log:
- deduplication executed
- forgetting executed
- number of memories removed

Example:
```
MANAGEMENT | dedup removed=12 kept=916
MANAGEMENT | forgetting removed=8 policy=age
```

---

### 4.3 Evolution Cycle Logging (CRITICAL)

Must log per cycle:
- generation number
- genotype parameters
- fitness score
- selected genotype

Example:
```
EVOLUTION | gen=7 fitness=0.742 selected={top_k:7, dedup:true}
```

---

### 4.4 Slow Retrieval Warnings (HIGH)

Thresholds:
- >1.0s → WARNING
- >5.0s → ERROR

Example:
```
WARNING | Slow retrieval 1.42s query='why do cats purr'
ERROR   | Critical slow retrieval 8.61s
```

---

### 4.5 Config Propagation Logs (HIGH)

Must log:
- evolution config changes
- runtime application confirmation

Example:
```
CONFIG | Evolution update default_top_k: 3 → 7
CONFIG | Runtime reload confirmed
```

---

## 5. Validation Checklist (Must All Pass)

After implementation, run **50–100 iterations** and confirm:

- [ ] Retrieval precision/recall > 0.0 and logged
- [ ] Evolution fitness varies across generations
- [ ] Dedup / forgetting logs appear
- [ ] Memory count no longer grows unbounded
- [ ] Runtime `top_k` reflects evolution state
- [ ] No null or fallback memories stored
- [ ] No unflagged retrievals > 1s

If any box fails → stop and debug before proceeding.

---

## 6. Agent Execution Notes

- Prefer **small, verifiable changes** over refactors
- Add logs before adding logic where possible
- Do not collect more data until P0 issues are fixed
- Stability and observability take priority over optimization

---

**Bottom line**: Until retrieval quality feedback, management execution, and config propagation are fixed **and logged**, evolution is effectively blind. Fix observability first, then performance.

---

## 7. Session Summary (Jan 31, 2026)

### **Completed Implementations**

#### ✅ P0.1 — Retrieval Quality Feedback Loop (COMPLETE)
- **Files Modified**: `evolution_manager.py`, `enhanced_middleware.py`
- **Changes**: 
  - Extended `record_memory_retrieval()` to accept precision/recall/quality
  - Added `_compute_retrieval_metrics()` in middleware
  - Added rolling window tracking for metrics
  - Added `RETRIEVAL_QUALITY` logging
- **Validation**: 140 retrievals logged with varying precision (0.00-1.00)
- **Status**: **WORKING** — Evolution receives actual performance metrics

#### 🔄 P0.3 — Config Propagation (CODE FIXED, AWAITING RESTART)
- **Files Modified**: `server.py`, `evolution_manager.py`, `enhanced_middleware.py`, `test_evolution_api.py`
- **Changes**:
  - Created shared ConfigManager in server.py
  - Made ConfigManager required parameter in EvolutionManager
  - Fixed middleware to read from ConfigManager live state
  - **CRITICAL FIX**: Changed `retrieval__default_top_k` to `retrieval.default_top_k`
- **Issues Found**: Double underscore syntax caused 20 evolution cycle failures
- **Validation**: Pending server restart
- **Status**: **FIXED IN CODE** — Need restart to verify runtime behavior

#### 📋 P1.3 — Unify Quality Scoring (PLANNED)
- Added to dev_tasks.md with detailed implementation plan
- **Scope**: Merge 3 scoring systems (643 lines → ~350 lines)
- **Status**: **NOT STARTED** — Planned for after P0 fixes

#### 📋 P1.4 — Filter Irrelevant Memories (PLANNED)
- Added to dev_tasks.md with implementation plan
- **Scope**: Filter memories with score ≤ 0.5 before injection
- **Status**: **NOT STARTED** — Planned for after P0 fixes

#### 📋 P0.4 — Remove Time-Based Evolution Trigger (PLANNED)
- **Scope**: Remove 24-hour time trigger, keep iteration-based triggers only
- **Goal**: Server idle = no evolution activity
- **Status**: **NOT STARTED** — Planned for after P0.3 validation

#### 📋 P0.5 — Dynamic Performance Baseline (PLANNED)
- **Scope**: Calculate response time baseline from actual iteration data
- **Goal**: Self-tuning degradation detection for any API endpoint
- **Status**: **NOT STARTED** — Planned for after P0 fixes

### **Key Discoveries**

1. **ConfigManager.update() uses dots, not underscores**: 
   - ❌ `retrieval__default_top_k=3` fails
   - ✅ `**{'retrieval.default_top_k': 3}` works

2. **Evolution takes 50 iterations to trigger** (regardless of previous data)
   - Auto-evolution check interval is hardcoded at 50
   - Evolution state persists but trigger counter resets

3. **Memory grows normally** (676 memories after 140 iterations)
   - No management operations yet (triggers at 1024)
   - 8 corrupted memories found (1.2%) with fallback chunk errors

4. **Fitness calculation still hardcoded**:
   - Evolution mutations occur but fitness stays at 0.837
   - `_run_test_trajectories()` uses static values, not actual metrics
   - **Next fix needed**: Connect real metrics to fitness calculation

5. **Baseline values are inaccurate**:
   - Response time baseline: 1.0s (hardcoded) vs actual 38s
   - Degradation triggers immediately due to 32x mismatch
   - **Solution**: Dynamic baseline from rolling average (P0.5)

### **Current Blockers**

1. **P0.3 needs restart validation** — Must verify config propagation works end-to-end
2. **Fitness calculation** — Still synthetic, needs to use actual retrieval metrics
3. **Memory management** — Not yet implemented (waiting for 1024 threshold)

---

### **NEW CRITICAL FIXES (Discovered During Testing)**

#### P0.7 Fix Hardcoded Evolution Trigger Interval

**Problem**
- `auto_evolution_check_interval` hardcoded to 50 in `enhanced_middleware.py:102`
- User set `MEMEVOLVE_AUTO_EVOLUTION_REQUESTS=32` in `.env` but it's ignored
- Evolution won't trigger until request #50 regardless of configuration

**Required Outcome**
- Evolution triggers at configured interval (e.g., 32 requests)
- Respects `MEMEVOLVE_AUTO_EVOLUTION_REQUESTS` from .env

**Implementation**
```python
# enhanced_middleware.py:102
# BEFORE:
self.auto_evolution_check_interval = 50

# AFTER:
self.auto_evolution_check_interval = getattr(
    config, 'auto_evolution', None
) and getattr(config.auto_evolution, 'requests', 50) or 50
```

**Validation**
- Set `MEMEVOLVE_AUTO_EVOLUTION_REQUESTS=10` in .env
- Evolution triggers at request #10, #20, #30 (not #50)

---

#### P0.8 Reduce Hedging Penalty in Quality Scoring

**Problem**
- Quality scores for exact question answers are only 0.55-0.60 (too low)
- Hedging penalty is 0.1 per word (max 0.3), too aggressive
- Natural conversational responses penalized heavily
- Current: "You could try restarting" loses 10% for "could"

**Required Outcome**
- Exact question answers score ≥ 0.75
- Hedging penalty reduced to 0.05 per word (max 0.15)
- Conversational language not overly penalized

**Implementation**
```python
# quality_scorer.py:205
# BEFORE:
hedging_penalty = sum(0.1 for word in hedging_words if word in content.lower())

# AFTER:
hedging_penalty = sum(0.05 for word in hedging_words if word in content.lower())
```

**Validation**
- Query: "What is 2+2?" → Answer: "The answer is 4" should score ≥ 0.75
- Query: "How do I restart?" → Answer: "You could try..." should score ≥ 0.70
- Logs show quality scores consistently above 0.65 for good answers

---

### **P1 — ENHANCEMENTS (Post-Stabilization)**

#### P1.5 Smart Index Type Management on Server Start

**Problem**
- User converts to `flat` index, then changes `.env` to `ivf`
- Server silently loads `flat` index, ignores `.env` setting
- No warning or option to convert
- User expects automatic rebuild or at least a notification

**Current Behavior**
```python
# vector_store.py:46-49
if not self._load_index():  # Loads existing flat index
    self._create_index()    # Only creates new if no index exists
# .env setting (ivf) is completely ignored!
```

**Required Behavior**
1. **Parse existing index** to detect its actual type (flat/ivf/hnsw)
2. **Compare with .env setting** (`MEMEVOLVE_STORAGE_INDEX_TYPE`)
3. **If mismatch detected**, offer user options:
   - **Option A**: Load as-is, update config to match index type
   - **Option B**: Convert index to match .env setting (rebuild)
   - **Option C**: Abort startup with clear error message

**Implementation Tasks**

1. **Add index type detection** (vector_store.py):
   ```python
   def _detect_index_type(self) -> Optional[str]:
       """Detect type of loaded FAISS index."""
       if self.index is None:
           return None
       # Check index class name
       class_name = type(self.index).__name__
       if 'IndexFlat' in class_name:
           return 'flat'
       elif 'IndexIVF' in class_name:
           return 'ivf'
       elif 'IndexHNSW' in class_name:
           return 'hnsw'
       return 'unknown'
   ```

2. **Add mismatch detection and handling** (vector_store.py:__init__):
   ```python
   def __init__(self, ...):
       # ... existing init code ...
       
       # Check for index type mismatch
       if self._load_index():
           detected_type = self._detect_index_type()
           if detected_type and detected_type != self.index_type:
               self._handle_index_type_mismatch(detected_type, self.index_type)
   
   def _handle_index_type_mismatch(self, detected: str, configured: str):
       """Handle mismatch between detected and configured index types."""
       import os
       
       # Check for environment variable override
       action = os.environ.get('MEMEVOLVE_INDEX_MISMATCH_ACTION', 'prompt')
       
       if action == 'auto-convert':
           self._convert_index_type(configured)
       elif action == 'auto-keep':
           self.index_type = detected
           print(f"Index type mismatch: using detected '{detected}' (ignoring .env '{configured}')")
       elif action == 'abort':
           raise RuntimeError(
               f"Index type mismatch: index is '{detected}' but .env requests '{configured}'. "
               f"Run with MEMEVOLVE_INDEX_MISMATCH_ACTION=auto-convert to rebuild, "
               f"or auto-keep to use existing."
           )
       else:  # prompt (default)
           print(f"\n{'='*60}")
           print("INDEX TYPE MISMATCH DETECTED")
           print(f"{'='*60}")
           print(f"Existing index: {detected}")
           print(f".env setting:   {configured}")
           print(f"\nOptions:")
           print(f"  1. Keep existing ({detected}) and update config")
           print(f"  2. Convert to {configured} (rebuild index)")
           print(f"  3. Abort startup")
           # ... interactive prompt or log warning for non-interactive ...
   
   def _convert_index_type(self, target_type: str):
       """Convert existing index to new type."""
       # Rebuild index with new type
       self._create_index()  # Creates new empty index of target type
       self._rebuild_index()  # Re-adds all vectors
       self._save_index()
       print(f"Converted index to {target_type}")
   ```

3. **Add environment variable for non-interactive mode**:
   ```bash
   # .env options
   MEMEVOLVE_INDEX_MISMATCH_ACTION=prompt    # Default: interactive/warning
   MEMEVOLVE_INDEX_MISMATCH_ACTION=auto-convert  # Silently rebuild
   MEMEVOLVE_INDEX_MISMATCH_ACTION=auto-keep     # Silently use existing
   MEMEVOLVE_INDEX_MISMATCH_ACTION=abort         # Fail fast with error
   ```

**Validation**
- Start with flat index, set .env to ivf → get prompt/warning
- Set `MEMEVOLVE_INDEX_MISMATCH_ACTION=auto-convert` → automatic rebuild
- Set `MEMEVOLVE_INDEX_MISMATCH_ACTION=auto-keep` → uses flat, updates config
- No silent mismatches: user always knows what's happening

---

### **Next Steps**

1. **Restart server** with P0.3 fix
2. **Run 50+ iterations** to trigger evolution
3. **Verify**:
   - No "Invalid config path" errors
   - `top_k` changes in middleware logs
   - Evolution advances past Gen 0 (fresh start)
   - Fitness scores vary
4. **Then proceed** to P0.4, P0.5, and P1 tasks

### **Files Changed This Session**

- `src/memevolve/api/evolution_manager.py` — Retrieval metrics, ConfigManager fix, precision/recall tracking
- `src/memevolve/api/enhanced_middleware.py` — Config propagation, metric computation, RETRIEVAL_QUALITY logging
- `src/memevolve/api/server.py` — Shared ConfigManager
- `tests/test_evolution_api.py` — Updated for new signatures
- `AGENTS.md` — Added Configuration Architecture Rules
- `dev_tasks.md` — Updated P0.3 status, added P0.4, P0.5, P1.3, P1.4

---

## 8. Infrastructure & Configuration Updates (Feb 1, 2026)

### **Completed Changes**

#### **Storage Backend Standardization**

**Changes Made**:
1. **Default storage**: Changed from `json` to `vector` in `.env.example`
2. **Default index type**: Changed from `flat` to `ivf` in `.env.example`
3. **File naming standardized**:
   - Vector: `vector.index` + `vector.data` (was: `vector_index.index`)
   - JSON: `memory.json` (was: `memory_system.json`)
4. **Removed**: `MEMEVOLVE_STORAGE_PATH` from `.env` (now hardcoded)

**Rationale**:
- Vector storage provides O(log n) retrieval vs O(n) for JSON
- Pre-computed embeddings enable consistent, fast semantic search
- IVF index scales better for 1000+ memories
- Hardcoded paths prevent misconfiguration

**Migration Path**:
```bash
# Convert existing JSON memories to vector
python convert_to_vector_store.py --index-type ivf

# Update .env (already done in .env.example)
MEMEVOLVE_STORAGE_BACKEND_TYPE=vector
MEMEVOLVE_STORAGE_INDEX_TYPE=ivf
```

**Files Modified**:
- `.env.example` — Updated defaults
- `src/memevolve/memory_system.py` — Standardized paths
- `convert_to_vector_store.py` — Updated default output path

---

#### **Environment Variable Cleanup**

**Removed** (no longer needed):
- `MEMEVOLVE_STORAGE_PATH` — Paths now hardcoded per backend type

**Rationale**: Storage paths are backend-specific and should not be configurable. Each backend has an optimal location:
- Vector: `./data/memory/vector.{index,data}`
- JSON: `./data/memory/memory.json`
- Graph: Neo4j connection (no file path)

---

#### **Retrieval Scoring Investigation**

**Findings**:
1. **Storage backend mismatch**: Using JSON store with semantic strategy causes brute-force embedding computation (53s first retrieval)
2. **No pre-computed embeddings**: JSON store doesn't store embeddings, computed on-the-fly
3. **Scoring inconsistency**: Embeddings computed at query time may differ from stored embeddings
4. **Keyword matching**: Current semantic similarity catches surface words, not concepts

**Root Cause**:
- `SemanticRetrievalStrategy` calls `storage_backend.retrieve_all()` then computes embeddings
- Should use `VectorStore.search()` for O(log n) FAISS-based retrieval
- No concept-level understanding in scoring

**Solutions Implemented**:
1. ✅ **Converted to VectorStore** — Pre-computed embeddings in FAISS index
2. ✅ **IVF index** — Fast approximate search for 821+ memories
3. 📝 **P1.6** — Add concept-level relevance scoring (documented below)

---

#### **New Tasks Added**

##### P1.6 Improve Retrieval Relevance Scoring

**Problem**:
- Semantic similarity matches keywords, not concepts
- Example: "difference between dog and cat" retrieves "difference between shadows and vision" (all scored 0.77)
- No distinction between surface word overlap and semantic meaning

**Required Outcome**:
- Concept-level matching (dog/cat vs shadows/vision are different)
- Reduced false positives from keyword matching
- More accurate relevance scores for memory injection

**Implementation Options**:

**Option A: Multi-factor scoring**:
```python
def _calculate_relevance_score(query, memory):
    embedding_score = cosine_similarity(...)  # Current: 50%
    keyword_score = jaccard_similarity(...)   # New: 20%
    concept_score = concept_overlap(...)      # New: 30%
    return weighted_combination
```

**Option B: Higher threshold + filtering**:
```python
# In enhanced_middleware.py
relevant_memories = [m for m in memories if m.get('score', 0) > 0.7]
```

**Option C: LLM-guided relevance** (future):
- Use lightweight LLM to verify relevance
- Higher accuracy, higher latency

**Recommendation**: Implement Option B immediately (quick win), then Option A for long-term improvement.

---

### **Current System State (Ready for Testing)**

**Configuration**:
- Storage: Vector (IVF index)
- Memories: 821 converted with embeddings
- Evolution: Auto-trigger at 32 requests
- Scoring: Hedging penalty reduced to 0.05

**Expected Improvements**:
- Retrieval time: 53s → <0.1s
- Scoring consistency: Pre-computed embeddings
- Scalability: O(log n) search

**Next Validation**:
1. Start server with vector/ivf
2. Run 32+ iterations
3. Verify retrieval times <0.5s
4. Check evolution triggers correctly
5. Monitor retrieval quality scores

---

## 9. Evolution Verbosity Fix (Feb 1, 2026)

### **Problem**

Evolution cycle logs excessive messages - 12 reconfiguration notifications for a single evolution:
```
INFO:MemorySystem:Successfully reconfigured retrieval  (x6)
INFO:MemorySystem:Successfully reconfigured manager   (x6)
```

**Root Cause**: Each genotype parameter is applied as a separate config update, triggering component reconfiguration and logging for every parameter.

**Impact**: 
- Console spam makes it hard to see important messages
- 12 log lines per evolution cycle (should be 2-3 max)
- Happens every 32 requests during active evolution

---

### **P0.9 Batch Evolution Config Updates**

**Required Outcome**
- Single log line per component reconfiguration
- Batch all genotype parameter updates into one operation
- Log shows summary of all changes, not individual updates

**Implementation**

**Option A: Batch ConfigManager updates** (Recommended):
```python
# evolution_manager.py - _apply_genotype_to_memory_system()

# BEFORE: Individual updates (causes 12 reconfigurations)
config_updates = {
    'retrieval.default_top_k': genotype.retrieve.default_top_k,
    'retrieval.strategy_type': genotype.retrieve.strategy_type,
    'retrieval.similarity_threshold': genotype.retrieve.similarity_threshold,
    # ... 9 more parameters
}

# Apply all at once
self.config_manager.update(**config_updates)

# Log once per component
logger.info(f"Evolution applied {len(config_updates)} parameters: "
            f"top_k={genotype.retrieve.default_top_k}, "
            f"strategy={genotype.retrieve.strategy_type}, ...")
```

**Option B: Add batching to MemorySystem**:
```python
# memory_system.py

def batch_reconfigure(self, config_updates: Dict[str, Any]):
    """Batch reconfiguration to reduce verbosity."""
    # Apply all updates
    for key, value in config_updates.items():
        self._apply_config(key, value)
    
    # Log once with summary
    components = set(k.split('.')[0] for k in config_updates.keys())
    logger.info(f"Successfully reconfigured: {', '.join(components)} "
                f"({len(config_updates)} parameters)")
```

**Files to Modify**:
- `src/memevolve/api/evolution_manager.py` - Batch genotype application
- `src/memevolve/memory_system.py` - Add batch reconfiguration method (optional)

**Validation**:
- Evolution cycle shows 1-3 log lines instead of 12
- Log format: `Successfully reconfigured retrieval, manager (12 parameters)`
- All parameters still applied correctly
- No functional changes, only logging reduction

---

### **Current Status**

**Working**:
- ✅ Evolution triggers at 32 requests
- ✅ Evolution state created and persisted
- ✅ Parameters mutated (top_k: 3→5, strategy: hybrid→semantic)
- ✅ No config errors

**Needs Fix**:
- 📝 Evolution verbosity (P0.9)
- 📝 Retrieval quality scoring (P1.6)
- 📝 Concept-level relevance (future)


---

#### P0.10 Implement Missing _run_test_trajectories Method

**Problem**
- All 18 evolution generations have identical fitness score: 0.837164
- Evolution is not actually optimizing anything
- Selection pressure is meaningless - no fitness differences
- System appears to work but has zero learning effect

**Root Cause**
- `_run_test_trajectories()` method missing from `selection.py`
- `evaluate()` falls back to hardcoded values:
  ```python
  performance = performance_data.get(genome_id, 0.5)  # ALWAYS 0.5!
  retrieval_accuracy = performance_data.get(genome_id, 0.5)  # ALWAYS 0.5!
  ```
- Fitness calculation becomes deterministic formula: 0.837164
- No actual performance testing occurs

**Evidence**
- 18 generations, all identical fitness ✅
- Parameters ARE changing (top_k: 3→5→10) ✅
- But fitness stays constant (0.837164) ❌
- No improvement detected in any generation (improvement=0.0)

**Required Outcome**
- Evolution actually measures real performance
- Different genotypes get different fitness scores
- Selection pressure drives optimization
- Fitness scores vary based on actual system performance

**Implementation Location**: `src/memevolve/evolution/selection.py`

**Core Implementation Steps**
1. **Add missing `_run_test_trajectories()` method**:
   ```python
   def _run_test_trajectories(self, genotype: MemoryGenotype) -> Dict[str, float]:
       """Run actual test trajectories with the given genotype."""
       # This should:
       # 1. Apply genotype to memory system
       # 2. Run test queries 
       # 3. Measure actual performance metrics
       # 4. Return real performance scores
   ```

2. **Real performance testing**:
   - Apply genotype configuration to memory system
   - Run benchmark queries from trajectory_tester.py
   - Measure actual retrieval quality, response time, etc.
   - Calculate genuine performance metrics

3. **Remove hardcoded fallbacks**:
   ```python
   # DELETE these lines in evaluate():
   performance = performance_data.get(genome_id, 0.5)     # WRONG
   retrieval_accuracy = performance_data.get(genome_id, 0.5)  # WRONG
   ```

4. **Integration with TrajectoryTester**:
   ```python
   from ..utils.trajectory_tester import TrajectoryTester
   tester = TrajectoryTester(self.memory_system, self.config)
   fitness_scores = tester.run_comprehensive_test(genotype)
   return {
       'performance': fitness_scores[0],      # task success rate
       'retrieval_accuracy': fitness_scores[1],  # retrieval quality
   }
   ```

**Validation**
- Generation 0 fitness: 0.837164 (baseline)
- Generation 1+ fitness: Different values based on actual performance
- Improvement scores vary (some positive, some negative)
- Evolution selects genuinely better performing genotypes
- System actually learns and optimizes

**Priority**: **CRITICAL** - Evolution is completely broken until fixed


---

#### P0.10 Implement Missing _run_test_trajectories Method

**Problem**
- All 18 evolution generations have identical fitness score: 0.837164
- Evolution is not actually optimizing anything
- Selection pressure is meaningless - no fitness differences
- System appears to work but has zero learning effect

**Root Cause**
- `_run_test_trajectories()` method missing from `selection.py`
- `evaluate()` falls back to hardcoded values
- Fitness calculation becomes deterministic formula, not based on real performance

**Evidence**
- 18 generations, all identical fitness score: 0.837164 ✅
- Parameters ARE changing (top_k: 3→5→10) ✅
- But fitness stays constant (0.837164) ❌
- No improvement detected in any generation (improvement=0.0) ✅

**Required Outcome**
- Evolution actually measures real performance
- Different genotypes get different fitness scores
- Selection pressure drives optimization
- Fitness scores vary based on actual system performance

**Implementation Location**: `src/memevolve/evolution/selection.py`

**Core Implementation Steps**
1. **Add missing `_run_test_trajectories()` method**:
   ```python
   def _run_test_trajectories(self, genotype: MemoryGenotype) -> Dict[str, float]:
       """Run actual test trajectories with the given genotype."""
       # This should:
       # 1. Apply genotype configuration to memory system
       # 2. Run test queries 
       # 3. Measure actual performance metrics
       # 4. Return real performance scores
   ```

2. **Real performance testing**:
   - Apply genotype configuration to memory system
   - Run benchmark queries from trajectory_tester.py
   - Measure actual retrieval quality, response time, etc.
   - Calculate genuine performance metrics

3. **Remove hardcoded fallbacks**:
   ```python
   # DELETE these lines in evaluate():
   performance = performance_data.get(genome_id, 0.5)     # WRONG
   retrieval_accuracy = performance_data.get(genome_id, 0.5)  # WRONG
   ```

4. **Integration with TrajectoryTester**:
   ```python
   from ..utils.trajectory_tester import TrajectoryTester
   tester = TrajectoryTester(self.memory_system, self.config)
   fitness_scores = tester.run_comprehensive_test(genotype)
   return {
       'performance': fitness_scores[0],      # task success rate
       'retrieval_accuracy': fitness_scores[1],  # retrieval quality
   }
   ```

**Validation**
- Generation 0 fitness: 0.837164 (baseline)
- Generation 1+ fitness: Different values based on actual performance
- Improvement scores vary (some positive, some negative)
- Evolution selects genuinely better performing genotypes
- System actually learns and optimizes

**Priority**: **CRITICAL** - Evolution is completely broken until fixed


#### P0.11 Centralized Config Policy Violation

**Problem**
- User set `MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD=1024` in `.env` 
- System is still using `auto_prune_threshold=1000` (hardcoded default)
- 717 memories lost (46.6% loss) because management never triggered
- Centralized config system failing to apply environment override

**Root Cause**
- Multiple hardcoded `1000` defaults exist in config system:
  1. `ManagementConfig` (line 79): `auto_prune_threshold: int = 1000`
  2. `EvolutionBoundaryConfig` (line 588): `fallback_top_k_min: int = 3`
  3. `EvolutionBoundaryConfig` (line 624): `fallback_top_k_max: int = 20`

**The Issue**
- ConfigManager should prioritize environment variables over hardcoded defaults
- `ConfigManager._load_config()` has proper environment mapping:
  ```python
  "MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD": (("management", "auto_prune_threshold"), int)
  ```
- But `memory_system.py:175` is using config directly instead of going through ConfigManager

**Evidence**
- `.env` contains: `MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD=1024` ✅
- System uses: `auto_prune_threshold=1000` ❌
- Result: 717 memories lost to silent storage failures
- 99.6% memory loss from management system that should have pruned

**Required Outcome**
- Environment variables should override hardcoded defaults
- ConfigManager should handle all parameter loading
- Centralized architecture should be enforced consistently

**Implementation Location**: Centralized config loading in `ConfigManager` and component initialization

**Validation**
- Check that `.env` variables take precedence over hardcoded defaults
- Verify that all components use `ConfigManager.get()` instead of accessing `config.<parameter>`
- Test that `MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD=1024` works as expected

**Priority**: **CRITICAL** - Fix centralized config violations


#### P0.12 Fix Centralized Config Policy Violation

**Problem**
- Mutation system still including `1024` as possible evolution parameter
- `genotype.py:376`: Random choice includes `1024` despite removal from mutation ranges
- P0.10 bug (fitness evaluation) makes this mutation impact analysis meaningless
- All mutations could create `auto_prune_threshold=1024`, breaking the 1000+ threshold

**Root Cause**
- Mutation code not properly respecting the removal of `auto_prune_threshold` from mutation ranges
- Even though parameters are removed from mutation, the random choice still includes the old value
- Mutation ranges need to be updated to exclude the removed parameter

**Evidence**
- `genotype.py:376` includes `1024` in choice list:
  ```python
  value = random.choice([256, 512, 1024, 2048])  # STILL THERE!
  ```
- Mutation documentation (line 281-282) claims it was removed:
  ```python
  # Note: auto_prune_threshold is evolved (triggers auto-management)
  ```

**Inconsistency**: Documentation vs Implementation

**Required Outcome**
- Remove `1024` from mutation choice list completely
- Ensure all mutation ranges exclude removed parameters
- Prevent mutations that violate centralized configuration architecture

**Implementation Location**: `src/memevolve/evolution/genotype.py`

**Validation**
- Verify `auto_prune_threshold` no longer appears in any mutation
- Check that all parameter ranges properly exclude removed items
- Test that mutations cannot create invalid threshold values

**Priority**: **HIGH** - Prevents invalid parameter mutations

#### P0.13 Fix Memory Storage Integrity Failure

**Problem**
- 717 memories lost from 821 created during 386 iterations (46.6% loss rate)
- Memory system claimed 1,538 storages but vector store only contains 821
- 717 memories silently vanished during batch processing operations
- Vector index bloat: 7.23MB instead of expected ~4.6MB for 821 memories

**Root Cause**
- Silent storage backend failures (no error logging)
- Vector storage corruption or duplication during index rebuilds
- Missing transaction/rollback mechanisms for failed storage operations
- No consistency checks between claimed and actual storage

**Evidence**
- Memory log: 1,538 storage operations claimed ✅
- Vector store: 821 memories actually stored ❌
- Missing unit IDs: [108, 265, 340, 356, 381, 404, 428, 509, 645, 654, 108] ❌
- Index-to-data ratio: 28.79:1 (should be ~3:1) ❌
- No error logs during storage failures ❌

**Impact**
- Evolution operates on incomplete dataset
- 717 missing memories create knowledge gaps
- Retrieval quality compromised by missing relevant data
- Performance optimization based on corrupted foundation

**Required Outcome**
- Implement transactional storage with rollback on failure
- Add comprehensive error logging for storage operations
- Implement storage verification (claim vs reality)
- Fix vector index corruption/duplication issues
- Add retry logic with exponential backoff for failed operations

**Priority**: **CRITICAL** - Data integrity foundation is compromised

#### P0.14 Evolution Backups Cleanup

**Problem**
- 80 evolution backup files stored (~25MB total) with redundant data
- No cleanup policy for old backup files
- Potential disk space waste and performance impact on backup operations

**Required Outcome**
- Implement backup rotation (keep only last N generations)
- Implement backup compression for storage efficiency
- Add backup cleanup policy (remove old/invalid backups)

**Priority**: **MEDIUM** - Backup management for system health

---

## 🎯 CRITICAL TASKS OVERVIEW (Updated)

| Priority | Task | Status |
|---------|------|--------|
| CRITICAL | P0.10 - Missing `_run_test_trajectories()` | **NOT STARTED** |
| CRITICAL | P0.11 - Centralized Config Policy Violation | **NOT STARTED** |
| CRITICAL | P0.12 - Mutation Still Includes 1024 | **NOT STARTED** |
| CRITICAL | P0.13 - Memory Storage Integrity Failure | **NOT STARTED** |
| CRITICAL | P0.9 - Batch Evolution Config Updates | **NOT STARTED** |
| HIGH | P1.6 - Concept-level Retrieval Scoring | **NOT STARTED** |

## 🚨 EXECUTION PRIORITY

1. **P0.10** (CRITICAL) - Evolution completely broken
2. **P0.11** (CRITICAL) - Centralized config not working
3. **P0.12** (CRITICAL) - Mutation violates architecture
4. **P0.13** (CRITICAL) - 717 memories lost

**All other tasks are BLOCKED** until these four CRITICAL issues are resolved.

**Root Cause Chain**:
- Storage failures → corrupted data → evolution working on incomplete dataset
- Config failures → wrong thresholds → management never triggered
- Evolution evaluation failures → no meaningful optimization
- Mutation violations → invalid parameters being evolved

**Immediate Action Required**:
1. Fix P0.10 (missing trajectory testing) → enables real evolution
2. Fix P0.13 (storage integrity) → preserves data foundation  
3. Fix P0.11 (config policy) → enables proper thresholds

**Once P0.10 is working, evolution will actually start optimizing and the other issues can be addressed in priority order.**

