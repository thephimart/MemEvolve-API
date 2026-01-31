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
