# IVF Vector Training Fix - Implementation Plan

## Executive Summary

**Problem**: The IVF index training logic uses synthetic/generic text patterns instead of actual stored memory vectors, causing poor clustering quality and search validation failures.

**Root Cause**: Training functions generate arbitrary text like "lesson learned from experience" instead of extracting embeddings from real stored units.

**Solution**: Implement a phased training strategy that:
1. Uses synthetic data for initial startup (when no real data exists)
2. Transitions to using actual stored vectors as data grows
3. Scales training target based on `max_units` configuration

---

## Background

### Current Architecture

```
Vector Store Flow:
┌─────────────────────────────────────────────────────────────────────────┐
│  add(unit)                                                              │
│    ├── index.add(embedding)          ← Adds to FAISS                  │
│    └── _accumulate_training_data()    ← Caches in _training_embeddings_cache │
│                                                                          │
│  Triggers:                                                               │
│    ├── _should_retrain_progressively()  ← Checks cache + units added   │
│    ├── _progressive_retrain_index()     ← Uses cache + synthetic        │
│    ├── _auto_rebuild_index()             ← Uses cache + synthetic       │
│    └── _train_ivf_if_needed()           ← Uses SYNTHETIC ONLY           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Issues Identified

| Issue | Location | Problem |
|-------|----------|---------|
| Synthetic-only training | `_train_ivf_if_needed` (line 1014) | Never uses real data |
| Mixed synthetic + cache | `_progressive_retrain_index` (line 185) | Adds 20 synthetic vectors to real cache |
| Synthetic fallback | `_auto_rebuild_index` (line 426) | Falls back to 100% synthetic when cache small |
| Wrong multiplier | `_generate_system_aligned_training_data` (line 961) | Uses `nlist * 2` instead of `nlist * 39` |

### Evidence from Logs

```
2026-02-13 15:50:21 - [IVF_TRAINING] Generated training data: (100, 768)
2026-02-13 15:50:21 - [IVF_TRAINING] Training IVF index with 100 vectors for 10 centroids
WARNING clustering 100 points to 10 centroids: please provide at least 390 training points
WARNING - [IVF_TRAINING] ⚠️ Training completed but comprehensive validation failed
```

**Data available**: ~425 stored units  
**Training used**: 100 synthetic vectors  
**FAISS recommendation**: 390+ (10 centroids × 39 vectors/centroid)  

---

## Training Strategy

### Phase 1: Initial Startup

**Condition**: `data_size < MIN_DATA_THRESHOLD` (default: 50 units)

- **Source**: Synthetic patterns from `_generate_system_aligned_training_data()`
- **Target**: `min_training_vectors` (default: 100)
- **Rationale**: No real data exists yet, synthetic provides basic clustering

### Phase 2: Progressive Retraining

**Condition**: `units_added >= retrain_threshold` (adaptive: 50-200 based on data size)

- **Source**: Actual stored units from `self.data`
- **Target**: `nlist * vectors_per_centroid` (e.g., 10 × 39 = 390)
- **Rationale**: Real data distribution improves IVF partition quality

### Phase 3: Scaled Training

**Condition**: `data_size > SCALE_THRESHOLD` (e.g., 1000 units)

- **Source**: Actual stored units (sampled if exceeding max)
- **Target**: `min(data_size, nlist * vectors_per_centroid)`
- **Rationale**: Maintain training quality as dataset grows

---

## Configuration Variables

### Existing (.env.example)

| Variable | Default | Purpose |
|----------|---------|---------|
| `MEMEVOLVE_STORAGE_MAX_UNITS` | 50000 | Cap for scaling calculations |
| `MEMEVOLVE_STORAGE_VECTORS_PER_CENTROID` | 39 | Quality target |
| `MEMEVOLVE_STORAGE_MIN_TRAINING_VECTORS` | 100 | Initial training size |
| `MEMEVOLVE_STORAGE_ENABLE_AUTO_RETRAIN` | true | Enable progressive retraining |

### Required Additions

| Variable | Default | Purpose |
|----------|---------|---------|
| `MEMEVOLVE_STORAGE_RETRAIN_THRESHOLD` | 100 | Units added before retrain triggers |
| `MEMEVOLVE_STORAGE_RETRAIN_MIN_DATA` | 50 | Min data size before using real vectors |

---

## Implementation Details

### 1. Update Config (config.py)

Add new fields to `StorageConfig`:
```python
# Retraining thresholds
retrain_threshold: int = 100  # Units added before retrain triggers
retrain_min_data_threshold: int = 50  # Min data before using real for training
```

Add environment variable loading in `__post_init__`:
```python
retrain_threshold_env = os.getenv("MEMEVOLVE_STORAGE_RETRAIN_THRESHOLD")
if retrain_threshold_env:
    self.retrain_threshold = int(retrain_threshold_env)

retrain_min_data_env = os.getenv("MEMEVOLVE_STORAGE_RETRAIN_MIN_DATA")
if retrain_min_data_env:
    self.retrain_min_data_threshold = int(retrain_min_data_env)
```

### 2. Create Training Data Extraction Function

New method in `VectorStore`:
```python
def _generate_training_from_actual_data(self, min_count: int, max_count: int = None) -> np.ndarray:
    """Extract training data from actual stored units.
    
    Args:
        min_count: Minimum number of vectors to return
        max_count: Maximum number of vectors (for large datasets)
    
    Returns:
        Stack of embedding vectors from stored units
    """
    embeddings = []
    
    for unit_id, unit in self.data.items():
        text = self._unit_to_text(unit)
        embedding = self._get_embedding(text)
        embeddings.append(embedding)
    
    if len(embeddings) < min_count:
        logger.warning(
            f"[TRAINING] Only {len(embeddings)} real vectors available, "
            f"need {min_count} - falling back to synthetic")
        # Return empty to trigger synthetic fallback
        return np.array([]).reshape(0, self.embedding_dim).astype('float32')
    
    # Sample if exceeding max_count
    if max_count and len(embeddings) > max_count:
        step = len(embeddings) // max_count
        embeddings = embeddings[::step][:max_count]
    
    return np.vstack(embeddings)
```

### 3. Update `_train_ivf_if_needed()`

Location: `vector_store.py` around line 1014

```python
def _train_ivf_if_needed(self, embedding: np.ndarray):
    """Train IVF index with appropriate data source based on data size."""
    if self.index_type == 'ivf':
        try:
            import faiss
            
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                nlist = getattr(self.index, 'nlist', 4)
                
                # Determine if we have enough real data
                min_data_threshold = getattr(
                    self.storage_config, 'retrain_min_data_threshold', 50)
                data_size = len(self.data)
                vectors_per_centroid = getattr(
                    self.storage_config, 'vectors_per_centroid', 39)
                
                if data_size >= min_data_threshold:
                    # Use actual stored data for training
                    target_count = min(data_size, nlist * vectors_per_centroid)
                    train_data = self._generate_training_from_actual_data(target_count)
                    
                    if len(train_data) > 0:
                        logger.info(
                            f"[IVF_TRAINING] Using {len(train_data)} real vectors "
                            f"(data_size={data_size})")
                    else:
                        # Fallback to synthetic if extraction failed
                        train_data = self._generate_system_aligned_training_data(
                            nlist * vectors_per_centroid)
                        logger.warning(
                            f"[IVF_TRAINING] Using synthetic (extraction failed)")
                else:
                    # Initial startup: use synthetic
                    train_data = self._generate_system_aligned_training_data(
                        nlist * vectors_per_centroid)
                    logger.info(
                        f"[IVF_TRAINING] Using synthetic (data_size={data_size} < {min_data_threshold})")
                
                # ... rest of training and validation logic
```

### 4. Update `_progressive_retrain_index()`

Location: `vector_store.py` around line 167

**Remove** synthetic mixing (lines 185-198):
```python
# BEFORE (broken):
synthetic_data = self._generate_system_aligned_training_data(20)  # REMOVE
cached_embeddings = np.vstack(self._training_embeddings_cache)
# Mix 70/30... REMOVE
```

**Replace** with pure actual data:
```python
# AFTER (fixed):
def _progressive_retrain_index(self) -> bool:
    if self.index_type != 'ivf':
        return False
    
    # Always use actual stored data, not cache + synthetic mix
    vectors_per_centroid = getattr(
        self.storage_config, 'vectors_per_centroid', 39)
    nlist = getattr(self.index, 'nlist', 10)
    target_count = min(len(self.data), nlist * vectors_per_centroid)
    
    train_data = self._generate_training_from_actual_data(target_count)
    
    if len(train_data) == 0:
        logger.warning("[PROGRESSIVE_TRAINING] No real training data available")
        return False
    
    logger.info(
        f"[PROGRESSIVE_TRAINING] Training with {len(train_data)} real vectors")
    
    # Create new index and train
    import faiss
    quantizer = faiss.IndexFlatL2(self.embedding_dim)
    new_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
    new_index.train(train_data)
    
    # ... rest of method (add existing vectors, replace index)
```

### 5. Update `_auto_rebuild_index()`

Location: `vector_store.py` around line 426

**Remove** synthetic fallback:
```python
# BEFORE (broken):
else:
    synthetic_data = self._generate_system_aligned_training_data(100)
    self.index.train(synthetic_data)
```

**Replace** with actual data extraction:
```python
# AFTER (fixed):
# Always use actual data, even if cache is small
actual_count = min(len(backup_data), 200)  # Cap at 200 for rebuild
train_data = self._generate_training_from_actual_data(
    min_count=actual_count,
    max_count=actual_count)

if len(train_data) == 0:
    logger.warning("[AUTO_REBUILD] No real training data available, using synthetic")
    train_data = self._generate_system_aligned_training_data(
        getattr(self.index, 'nlist', 10) * vectors_per_centroid)

self.index.train(train_data)
```

### 6. Fix Target Count Multiplier

Location: `_generate_system_aligned_training_data()` line 961-964

```python
# BEFORE:
target_count = max(
    nlist * 2,  # BUG: hardcoded
    min_training_vectors,
    len(training_texts))

# AFTER:
vectors_per_centroid = getattr(
    self.storage_config, 'vectors_per_centroid', 39)
target_count = max(
    nlist * vectors_per_centroid,  # FIXED: use config
    min_training_vectors,
    len(training_texts))
```

### 7. Add Scaling Function

New method for calculating training target based on max_units:
```python
def _calculate_training_target(self) -> int:
    """Calculate optimal training target based on data size and max_units."""
    vectors_per_centroid = getattr(
        self.storage_config, 'vectors_per_centroid', 39)
    max_units = getattr(
        self.storage_config, 'max_units', 50000)
    data_size = len(self.data)
    
    # Calculate nlist based on current data size
    nlist = self._calculate_optimal_nlist(data_size)
    
    # Target: vectors_per_centroid * nlist
    target = nlist * vectors_per_centroid
    
    # Cap at actual data size
    target = min(target, data_size)
    
    # Scale with max_units for large datasets
    if data_size > max_units * 0.5:
        target = min(target, max_units // 2)
    
    return max(target, 100)  # Minimum 100
```

---

## File Changes Summary

| File | Changes |
|------|---------|
| `config.py` | Add `retrain_threshold`, `retrain_min_data_threshold` fields and env loading |
| `vector_store.py` | Update `_train_ivf_if_needed`, `_progressive_retrain_index`, `_auto_rebuild_index`, fix multiplier |
| `.env.example` | Add new env variables (optional, defaults work) |

---

## Testing Plan

### Unit Tests
1. `_generate_training_from_actual_data` returns correct count
2. Synthetic fallback triggers when data insufficient
3. Training target scales with data size

### Integration Tests
1. Initial startup uses synthetic
2. After 50+ units added, retrain uses real data
3. Validation passes after retraining with real data

### Manual Verification
1. Check logs for training source (synthetic vs real)
2. Verify validation success rate improves
3. Monitor search accuracy metrics

---

## Success Criteria

| Metric | Before | After |
|--------|--------|-------|
| Training data source | 100% synthetic | Real vectors (when available) |
| Validation success rate | 0% | >50% |
| IVF partition quality | Poor (random-like) | Good (reflects actual data distribution) |
| Search accuracy | Degraded | Improved |

---

## Rollback Plan

If issues arise:
1. Set `MEMEVOLVE_STORAGE_ENABLE_AUTO_RETRAIN=false` to disable retraining
2. Set `MEMEVOLVE_STORAGE_RETRAIN_MIN_DATA=1000` to delay real-data training
3. Revert to synthetic-only by setting data size threshold high

---

**Document Version**: 1.0  
**Created**: 2026-02-13  
**Status**: Ready for Implementation
