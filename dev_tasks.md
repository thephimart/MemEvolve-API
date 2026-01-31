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

**Required Outcome**
- Runtime retrieval reflects evolved config within one evolution cycle

**Implementation Tasks**
- Verify `ConfigManager.update()` propagation chain
- Ensure retrieval components read live config (not cached defaults)

**Validation Metrics**
- Logs show runtime `top_k` changing (e.g. 3 → 7)
- Retrieval behavior changes accordingly

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
