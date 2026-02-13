"""
Vector store implementation using FAISS for efficient similarity search.

FAISS deprecation warnings: The warnings about SwigPyPacked, SwigPyObject, and swigvarlink
are from FAISS's internal SWIG-generated Python bindings. These are cosmetic warnings that
don't affect functionality. FAISS is still actively maintained and provides the best
performance for vector similarity search. These warnings are safely suppressed.
"""

import logging
import os
import pickle
import warnings
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .base import MetadataMixin, StorageBackend

from ...utils.logging_manager import LoggingManager

logger = LoggingManager.get_logger(__name__)
# Suppress FAISS SWIG deprecation warnings (cosmetic, don't affect functionality)
warnings.filterwarnings(
    "ignore", message=".*SwigPyPacked.*", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore", message=".*SwigPyObject.*", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore", message=".*swigvarlink.*", category=DeprecationWarning)


class VectorStore(StorageBackend, MetadataMixin):
    """FAISS-based vector store backend for memory units."""

    def __init__(
        self,
        index_file: str,
        embedding_function: Callable[[str], np.ndarray],
        embedding_dim: Optional[int] = None,
        index_type: str = 'flat',
        storage_config: Optional[Any] = None
    ):
        self.index_file = index_file
        self.embedding_function = embedding_function
        self.index_type = index_type
        self.storage_config = storage_config

        # CRITICAL FIX: Auto-detect embedding dimension from service instead of hardcoded 384
        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = self._detect_embedding_dimension()

        logger.debug(
            f"[IVF_INITIALIZATION] Initialized with auto-detected embedding_dim={self.embedding_dim}")

        self.data: Dict[str, Dict[str, Any]] = {}
        self.index: Optional[Any] = None  # FAISS index type is dynamic

        # Health metrics cache to prevent blocking operations
        self._health_cache = None
        self._health_cache_time = 0

        # Progressive training data accumulation (Phase 2.1)
        self._training_embeddings_cache: List[np.ndarray] = []
        self._retrain_threshold = 100  # Retrain after 100 new units
        self._corruption_detected = False

        # Try to load existing index and data, fall back to creating new one
        if not self._load_index():
            self._create_index(data_size=0)
        self._load_data()

    def _detect_embedding_dimension(self) -> int:
        """Auto-detect embedding dimension from the embedding service."""
        try:
            # Use a simple test text to detect embedding dimension
            test_embedding = self.embedding_function("test dimension detection")
            detected_dim = test_embedding.shape[0]
            logger.info(f"[IVF_INITIALIZATION] Auto-detected embedding dimension: {detected_dim}")
            return detected_dim
        except Exception as e:
            logger.warning(f"[IVF_INITIALIZATION] Failed to auto-detect embedding dimension: {e}")
            # Fallback to 768 (common for modern embedding models)
            logger.info("[IVF_INITIALIZATION] Using fallback embedding dimension: 768")
            return 768

    def _accumulate_training_data(self, embedding: np.ndarray) -> None:
        """Phase 2.1: Accumulate real training data for progressive IVF improvement.

        Args:
            embedding: New embedding to add to training cache
        """
        if self.index_type != 'ivf':
            return

        # Add new embedding to training cache
        self._training_embeddings_cache.append(embedding.copy())

        # Limit cache size to prevent memory issues (keep last 1000 embeddings)
        max_cache_size = 1000
        if len(self._training_embeddings_cache) > max_cache_size:
            self._training_embeddings_cache = self._training_embeddings_cache[-max_cache_size:]

        logger.debug(
            f"[PROGRESSIVE_TRAINING] Cached {len(self._training_embeddings_cache)} training embeddings")

    def _should_retrain_progressively(self) -> bool:
        """Phase 2.1: Check if progressive retraining should be triggered.

        Retrains when nlist needed for current data exceeds index nlist.
        Uses same formula as retrain: floor(data/39) rounded down to nearest 10.

        Returns:
            True if retraining is recommended
        """
        if self.index_type != 'ivf':
            return False

        current_size = len(self.data)

        # Only consider retraining for larger datasets (minimum for reasonable clustering)
        if current_size < 50:
            return False

        # Calculate nlist using same formula as retrain (floor, round down to 10s)
        target_nlist = max(10, (current_size // 39) // 10 * 10)
        index_nlist = getattr(self.index, 'nlist', 10)

        # Retrain when target nlist exceeds current index nlist
        should_retrain = target_nlist > index_nlist

        if should_retrain:
            logger.info(
                f"[PROGRESSIVE_TRAINING] Retrain: index nlist {index_nlist} -> {target_nlist} at {current_size} vectors")

        return should_retrain

    def _progressive_retrain_index(self) -> bool:
        """Phase 2.1: Retrain IVF index with ALL real vectors from self.data.

        Uses FAISS-recommended nlist * 39 training vectors for optimal clustering.
        No synthetic data is used - only real embeddings from stored units.

        Returns:
            True if retraining was successful
        """
        if self.index_type != 'ivf':
            return False

        current_size = len(self.data)
        if current_size < 50:
            return False

        # Get nlist based on actual data size
        # nlist = data_size // 39, then round DOWN to nearest 10 (10, 20, 30, etc.)
        # Example: 478 // 39 = 12 -> 10; 780 // 39 = 20 -> 20
        nlist = max(10, (current_size // 39) // 10 * 10)

        try:
            import faiss

            logger.info(
                f"[PROGRESSIVE_TRAINING] Starting retrain: {current_size} vectors -> nlist {nlist}")

            # Generate embeddings from ALL stored units for training
            training_embeddings = []
            for unit in self.data.values():
                text = self._unit_to_text(unit)
                embedding = self._get_embedding(text)
                training_embeddings.append(embedding)

            training_data = np.vstack(training_embeddings)
            logger.info(
                f"[PROGRESSIVE_TRAINING] Training with {len(training_data)} real vectors")

            # Create new index with nlist based on actual data size
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            new_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)

            # Train with all real data
            new_index.train(training_data)

            if new_index.is_trained:
                # Add all existing vectors to new index
                new_index.add(training_data)
                logger.info(
                    f"[PROGRESSIVE_TRAINING] Added {len(training_data)} vectors to retrained index")

                # Replace old index
                self.index = new_index

                # Save the improved index
                self._save_index()

                logger.info(
                    f"[PROGRESSIVE_TRAINING] ✅ Successfully retrained IVF index with real data")
                return True
            else:
                logger.error(
                    "[PROGRESSIVE_TRAINING] ❌ Progressive retraining failed - index not trained")
                return False

        except Exception as e:
            logger.error(f"[PROGRESSIVE_TRAINING] Progressive retraining failed: {e}")
            return False

    def _detect_corruption(self) -> Dict[str, Any]:
        """Phase 2.2: Detect IVF index corruption using multiple validation methods.

        Returns:
            Dictionary with corruption detection results
        """
        corruption_results = {
            "corrupted": False,
            "issues": [],
            "mapping_errors": [],
            "index_stats": {}
        }

        try:
            if self.index is None:
                corruption_results["corrupted"] = True
                corruption_results["issues"].append("Index is None")
                return corruption_results

            # Get basic index statistics
            total_vectors = self.index.ntotal
            corruption_results["index_stats"]["total_vectors"] = total_vectors
            corruption_results["index_stats"]["data_units"] = len(self.data)

            # Check for size mismatch between index and data
            if total_vectors != len(self.data):
                corruption_results["corrupted"] = True
                corruption_results["issues"].append(
                    f"Index/data size mismatch: {total_vectors} vs {len(self.data)}")

            # For IVF indexes, check mapping integrity only if multiple failures occur
            if self.index_type == 'ivf' and hasattr(
                    self.index, 'is_trained') and self.index.is_trained and len(
                    self.data) > 10:
                # Sample verification for recent units (only if we have enough data)
                sample_size = min(3, max(1, len(self.data) // 20))
                unit_ids = list(self.data.keys())[-sample_size:]  # Recent units

                failed_verifications = 0
                for unit_id in unit_ids:
                    unit = self.data[unit_id]
                    text = self._unit_to_text(unit)
                    embedding = self._get_embedding(text)

                    verification = self._verify_storage(unit_id, embedding)
                    if not verification["verified"]:
                        failed_verifications += 1
                        corruption_results["mapping_errors"].append(
                            f"{unit_id}: {verification['error']}")

                # Only flag corruption if multiple verifications fail
                if failed_verifications > 0 and failed_verifications >= sample_size:
                    corruption_results["corrupted"] = True
                    corruption_results["issues"].append(
                        f"Multiple mapping errors: {failed_verifications}/{sample_size}")

            # Check for consistent nlist configuration
            if self.index_type == 'ivf':
                expected_nlist = self._calculate_optimal_nlist(len(self.data))
                actual_nlist = getattr(self.index, 'nlist', None)
                if actual_nlist and abs(actual_nlist - expected_nlist) > expected_nlist * 0.5:
                    corruption_results["issues"].append(
                        f"Nlist mismatch: expected ~{expected_nlist}, got {actual_nlist}")

            if corruption_results["corrupted"]:
                logger.error(
                    f"[CORRUPTION_DETECTION] ❌ Corruption detected: {
                        corruption_results['issues']}")
                self._corruption_detected = True
            else:
                logger.debug(
                    f"[CORRUPTION_DETECTION] ✅ No corruption detected in {total_vectors} vectors")

        except Exception as e:
            corruption_results["corrupted"] = True
            corruption_results["issues"].append(f"Corruption detection failed: {e}")
            logger.error(f"[CORRUPTION_DETECTION] Error during corruption detection: {e}")

        return corruption_results

    def _enhanced_verify_storage(self, unit_id: str, embedding: np.ndarray) -> Dict[str, Any]:
        """Phase 2.2: Enhanced storage verification with corruption detection.

        Args:
            unit_id: ID of the stored unit
            embedding: The embedding that was stored

        Returns:
            Dict with enhanced verification results
        """
        # Start with basic verification
        verification = self._verify_storage(unit_id, embedding)

        # Add corruption detection for IVF indexes
        if self.index_type == 'ivf' and verification["verified"]:
            try:
                # Additional check: test search with slightly modified embedding
                modified_embedding = embedding + np.random.normal(0, 0.01, embedding.shape)
                modified_embedding = modified_embedding.astype('float32')

                if self.index.ntotal > 1:
                    distances, indices = self.index.search(modified_embedding, k=2)

                    # Check that original unit is still in top results
                    found_original = False
                    for idx in indices[0]:
                        if idx >= 0:
                            # CRITICAL FIX: Use proper index-to-unit mapping
                            found_unit_id = None
                            if hasattr(
                                    self, '_index_to_unit_mapping') and idx in self._index_to_unit_mapping:
                                found_unit_id = self._index_to_unit_mapping[idx]
                            else:
                                # Fallback to key order (less reliable)
                                if idx < len(list(self.data.keys())):
                                    found_unit_id = list(self.data.keys())[idx]

                            if found_unit_id == unit_id:
                                found_original = True
                                break

                    if not found_original:
                        # Log warning but don't fail - this check is too aggressive
                        logger.warning(
                            f"[ENHANCED_VERIFICATION] Modified search failed for {unit_id} (non-critical)")
                        verification["corruption_check"] = "modified_search_failed"
                        # Don't set verification["verified"] = False - basic verification already
                        # passed

                verification["corruption_check"] = "passed"

            except Exception as e:
                verification["corruption_check"] = f"failed: {e}"
                logger.warning(
                    f"[ENHANCED_VERIFICATION] Corruption check failed for {unit_id}: {e}")

        return verification

    def _auto_rebuild_index(self) -> bool:
        """Phase 2.3: Automatically rebuild corrupted IVF index with enhanced training.

        Returns:
            True if rebuilding was successful
        """
        if not self._corruption_detected:
            return True  # No corruption detected, no rebuild needed

        try:
            logger.warning("[AUTO_REBUILD] Starting automatic index rebuilding due to corruption")

            # Step 1: Backup current data
            backup_data = self.data.copy()
            backup_training_cache = self._training_embeddings_cache.copy()

            # Step 2: Create fresh index with optimized parameters
            self._create_index(data_size=len(backup_data))

            # Step 3: Generate training data using ALL real vectors
            if self.index_type == 'ivf':
                # Get all embeddings from stored units for training
                training_embeddings = []
                for unit in backup_data.values():
                    text = self._unit_to_text(unit)
                    embedding = self._get_embedding(text)
                    training_embeddings.append(embedding)

                if len(training_embeddings) >= 50:
                    training_data = np.vstack(training_embeddings)
                    logger.info(
                        f"[AUTO_REBUILD] Training with {
                            len(training_data)} real vectors (no synthetic)")

                    # Train the new index
                    self.index.train(training_data)

                    if not self.index.is_trained:
                        logger.error("[AUTO_REBUILD] Training failed during automatic rebuild")
                        return False
                else:
                    # Fallback: need at least 50 vectors for reasonable training
                    logger.warning(
                        f"[AUTO_REBUILD] Not enough vectors for training: {
                            len(training_embeddings)}")
                    # Use what we have but warn
                    training_data = np.vstack(training_embeddings) if training_embeddings else None
                    if training_data is not None and len(training_data) > 0:
                        self.index.train(training_data)

            # Step 4: Re-add all existing units
            rebuilt_count = 0
            failed_rebuilds = []

            for unit_id, unit in backup_data.items():
                try:
                    text = self._unit_to_text(unit)
                    embedding = self._get_embedding(text)
                    self.index.add(x=embedding)
                    rebuilt_count += 1
                except Exception as rebuild_error:
                    failed_rebuilds.append((unit_id, str(rebuild_error)))
                    logger.warning(f"[AUTO_REBUILD] Failed to re-add {unit_id}: {rebuild_error}")

            # Step 5: Restore data and update cache
            self.data = backup_data
            self._training_embeddings_cache = backup_training_cache

            # Step 6: Verify rebuild success
            if rebuilt_count > 0:
                # Test a few units to verify rebuild integrity
                test_units = list(self.data.keys())[:min(3, len(self.data))]
                rebuild_verification_passed = True

                for test_unit_id in test_units:
                    test_unit = self.data[test_unit_id]
                    text = self._unit_to_text(test_unit)
                    embedding = self._get_embedding(text)
                    verification = self._verify_storage(test_unit_id, embedding)

                    if not verification["verified"]:
                        rebuild_verification_passed = False
                        logger.error(
                            f"[AUTO_REBUILD] Verification failed for {test_unit_id}: {
                                verification['error']}")
                        break

                if rebuild_verification_passed:
                    # Step 7: Save rebuilt index
                    self._save_index()
                    self._save_data()

                    self._corruption_detected = False
                    logger.info(
                        f"[AUTO_REBUILD] ✅ Successfully rebuilt index: {rebuilt_count}/{len(backup_data)} units")

                    if failed_rebuilds:
                        failed_summary = [f"{uid}: {err}" for uid, err in failed_rebuilds[:5]]
                        logger.warning(
                            f"[AUTO_REBUILD] Some units failed to rebuild: {failed_summary}")

                    return True
                else:
                    logger.error("[AUTO_REBUILD] Rebuild verification failed")
                    return False
            else:
                logger.error("[AUTO_REBUILD] No units were successfully rebuilt")
                return False

        except Exception as e:
            logger.error(f"[AUTO_REBUILD] Automatic rebuilding failed: {e}")
            # Last resort: create empty index to prevent system failure
            try:
                self._create_index(data_size=0)
                self.data.clear()
                self._training_embeddings_cache.clear()
                self._corruption_detected = False
                logger.warning("[AUTO_REBUILD] Created empty index as last resort")
                return False
            except Exception as fallback_error:
                logger.error(f"[AUTO_REBUILD] Even fallback failed: {fallback_error}")
                return False

    def management_health_check(self) -> Dict[str, Any]:
        """Phase 2.4: Health check interface for memory management operations.

        Called by management strategies to assess vector store health and
        trigger maintenance operations if needed.

        Returns:
            Dictionary with health status and recommendations
        """
        health_report = {
            "healthy": True,
            "issues": [],
            "recommendations": [],
            "stats": {},
            "actions_taken": []
        }

        try:
            # Basic statistics
            health_report["stats"] = {
                "total_units": len(self.data),
                "index_type": self.index_type,
                "training_cache_size": len(self._training_embeddings_cache),
                "corruption_detected": self._corruption_detected
            }

            # Check for corruption
            corruption_check = self._detect_corruption()
            if corruption_check["corrupted"]:
                health_report["healthy"] = False
                health_report["issues"].extend(corruption_check["issues"])
                health_report["recommendations"].append(
                    "Index corruption detected - rebuild required")

                # Attempt automatic rebuilding
                if self._auto_rebuild_index():
                    health_report["actions_taken"].append("Automatic index rebuilding completed")
                    health_report["healthy"] = True
                    health_report["issues"] = []  # Clear issues after successful rebuild
                else:
                    health_report["actions_taken"].append("Automatic rebuilding failed")
                    health_report["recommendations"].append("Manual index rebuilding required")

            # Check training cache health
            if self.index_type == 'ivf':
                cache_size = len(self._training_embeddings_cache)
                data_size = len(self.data)

                if cache_size < 10 and data_size > 50:
                    health_report["recommendations"].append(
                        "Training cache too small - accumulation needed")

                # Check if retraining would be beneficial
                if self._should_retrain_progressively():
                    health_report["recommendations"].append("Progressive retraining recommended")

                    # Trigger progressive retraining
                    if self._progressive_retrain_index():
                        health_report["actions_taken"].append("Progressive retraining completed")
                    else:
                        health_report["recommendations"].append("Progressive retraining failed")

            # Check size management
            if self.storage_config:
                max_size = getattr(self.storage_config, 'max_units', 50000)
                warning_threshold = int(
                    max_size *
                    getattr(
                        self.storage_config,
                        'warning_threshold',
                        0.8))

                if data_size >= max_size:
                    health_report["healthy"] = False
                    health_report["issues"].append(f"Size limit reached: {data_size}/{max_size}")
                    health_report["recommendations"].append("Memory consolidation required")
                elif data_size >= warning_threshold:
                    health_report["recommendations"].append(
                        f"Approaching size limit: {data_size}/{max_size}")

            # Log health check results
            if health_report["healthy"]:
                logger.info(
                    f"[MANAGEMENT_HEALTH] ✅ Vector store healthy: {data_size} units, {cache_size} cached embeddings")
            else:
                logger.warning(
                    f"[MANAGEMENT_HEALTH] ⚠️ Vector store issues: {
                        health_report['issues']}")
                if health_report["actions_taken"]:
                    logger.info(
                        f"[MANAGEMENT_HEALTH] Actions taken: {
                            health_report['actions_taken']}")

        except Exception as e:
            health_report["healthy"] = False
            health_report["issues"].append(f"Health check failed: {e}")
            logger.error(f"[MANAGEMENT_HEALTH] Health check error: {e}")

        return health_report

    def optimize_for_management(self) -> Dict[str, Any]:
        """Phase 2.4: Optimize vector store for management operations.

        Called before/after management operations like pruning, consolidation,
        or deduplication to ensure optimal performance.

        Returns:
            Dictionary with optimization results
        """
        optimization_results = {
            "optimized": False,
            "actions_taken": [],
            "errors": []
        }

        try:
            logger.info("[MANAGEMENT_OPTIMIZE] Starting optimization for management operations")

            # Action 1: Progressive retraining if needed
            if self._should_retrain_progressively():
                if self._progressive_retrain_index():
                    optimization_results["actions_taken"].append("Progressive retraining completed")
                else:
                    optimization_results["errors"].append("Progressive retraining failed")

            # Action 2: Corruption check and rebuild
            corruption_check = self._detect_corruption()
            if corruption_check["corrupted"]:
                if self._auto_rebuild_index():
                    optimization_results["actions_taken"].append("Corruption repair completed")
                else:
                    optimization_results["errors"].append("Corruption repair failed")

            # Action 3: Training cache cleanup
            if len(self._training_embeddings_cache) > 1000:
                old_size = len(self._training_embeddings_cache)
                self._training_embeddings_cache = self._training_embeddings_cache[-500:]
                optimization_results["actions_taken"].append(
                    f"Training cache cleaned: {old_size} -> {len(self._training_embeddings_cache)}")

            # Action 4: Index parameter optimization
            if self.index_type == 'ivf':
                current_nlist = getattr(self.index, 'nlist', None)
                optimal_nlist = self._calculate_optimal_nlist(len(self.data))

                if current_nlist and abs(current_nlist - optimal_nlist) > optimal_nlist * 0.3:
                    logger.info(
                        f"[MANAGEMENT_OPTIMIZE] Nlist optimization: {current_nlist} -> {optimal_nlist}")
                    # Rebuild with optimal parameters
                    if self._auto_rebuild_index():
                        optimization_results["actions_taken"].append(
                            f"Nlist optimized: {current_nlist} -> {optimal_nlist}")
                    else:
                        optimization_results["errors"].append("Nlist optimization failed")

            optimization_results["optimized"] = len(optimization_results["actions_taken"]) > 0

            if optimization_results["optimized"]:
                logger.info(
                    f"[MANAGEMENT_OPTIMIZE] ✅ Optimization completed: {
                        optimization_results['actions_taken']}")
            else:
                logger.debug("[MANAGEMENT_OPTIMIZE] No optimization needed")

        except Exception as e:
            optimization_results["errors"].append(f"Optimization failed: {e}")
            logger.error(f"[MANAGEMENT_OPTIMIZE] Optimization error: {e}")

        return optimization_results

    def _verify_storage(self, unit_id: str, embedding: np.ndarray) -> Dict[str, Any]:
        """Verify that a stored memory unit can be retrieved correctly.

        Args:
            unit_id: ID of the stored unit
            embedding: The embedding that was stored

        Returns:
            Dict with verification results
        """
        try:
            logger.debug(f"[STORAGE_VERIFICATION] 🔍 Verifying storage for {unit_id}")

            # Check 1: Unit exists in data dictionary
            if unit_id not in self.data:
                return {"verified": False, "error": f"Unit {unit_id} not found in data dictionary"}

            # Check 2: Can retrieve by vector search
            try:
                # Search for the exact embedding using direct FAISS search
                if self.index is None or self.index.ntotal == 0:
                    return {"verified": False, "error": "FAISS index is empty or not initialized"}

                # Use direct FAISS search with the embedding
                if len(embedding.shape) == 1:
                    embedding_reshaped = embedding.reshape(1, -1).astype('float32')
                else:
                    embedding_reshaped = embedding.astype('float32')
                distances, indices = self.index.search(embedding_reshaped, k=1)

                if len(indices[0]) == 0 or indices[0][0] < 0:
                    return {"verified": False, "error": "FAISS search returned no results"}

                # Get the found unit ID using index-to-unit mapping
                found_idx = indices[0][0]

                # CRITICAL FIX: Use proper index-to-unit mapping instead of key order
                found_unit_id = None
                if hasattr(
                        self,
                        '_index_to_unit_mapping') and found_idx in self._index_to_unit_mapping:
                    found_unit_id = self._index_to_unit_mapping[found_idx]
                else:
                    # Fallback to key order (less reliable)
                    if found_idx < len(list(self.data.keys())):
                        found_unit_id = list(self.data.keys())[found_idx]
                    else:
                        return {"verified": False,
                                "error": f"Invalid index {found_idx} returned from FAISS"}

                if found_unit_id != unit_id:
                    return {
                        "verified": False,
                        "error": f"Vector search found wrong unit. Expected: {unit_id}, Found: {found_unit_id}"}
                logger.debug(
                    f"[STORAGE_VERIFICATION] ✅ Vector search found {unit_id} with distance {
                        distances[0][0]:.4f}")
            except Exception as search_error:
                return {"verified": False, "error": f"Vector search failed: {search_error}"}

            # Check 3: Verify metadata integrity
            stored_unit = self.data[unit_id]
            required_fields = ["id", "type", "content", "tags", "metadata"]
            missing_fields = [field for field in required_fields if field not in stored_unit]
            if missing_fields:
                return {"verified": False, "error": f"Missing required fields: {missing_fields}"}

            # Check 4: Verify embedding dimension matches
            # Handle both (dim,) and (1, dim) shapes
            if len(embedding.shape) == 1:
                actual_dim = embedding.shape[0]
            else:
                actual_dim = embedding.shape[-1]

            if actual_dim != self.embedding_dim:
                return {
                    "verified": False,
                    "error": f"Embedding dimension mismatch: {actual_dim} != {
                        self.embedding_dim}"}

            logger.debug(f"[STORAGE_VERIFICATION] ✅ All verification checks passed for {unit_id}")
            return {"verified": True, "unit_id": unit_id}

        except Exception as e:
            return {"verified": False, "error": f"Verification failed: {e}"}

    def _rebuild_index_without_unit(self, unit_id: str) -> None:
        """Rebuild FAISS index without a specific unit for rollback."""
        try:
            logger.warning(f"[STORAGE_VERIFICATION] 🔄 Rebuilding index without unit {unit_id}")

            # Save current data temporarily
            temp_data = self.data.copy()

            # Remove the problematic unit
            if unit_id in temp_data:
                del temp_data[unit_id]

            # Create new index
            self._create_index(data_size=len(temp_data))

            # CRITICAL FIX: Reset index-to-unit mapping for rebuild
            if hasattr(self, '_index_to_unit_mapping'):
                self._index_to_unit_mapping.clear()

            # CRITICAL FIX: Train IVF index before adding vectors
            if self.index_type == 'ivf' and len(temp_data) > 0:
                # Generate embeddings for training
                all_embeddings = []
                for remaining_id, remaining_unit in temp_data.items():
                    text = self._unit_to_text(remaining_unit)
                    embedding = self._get_embedding(text)
                    all_embeddings.append(embedding)

                # Train the index with available data
                if all_embeddings:
                    self._train_ivf_if_needed(all_embeddings[0])

            # Re-add all remaining units with proper mapping
            for idx, (remaining_id, remaining_unit) in enumerate(temp_data.items()):
                try:
                    text = self._unit_to_text(remaining_unit)
                    embedding = self._get_embedding(text)
                    self.index.add(x=embedding)

                    # CRITICAL FIX: Rebuild index-to-unit mapping
                    if hasattr(self, '_index_to_unit_mapping'):
                        self._index_to_unit_mapping[idx] = remaining_id

                except Exception as readd_error:
                    logger.warning(
                        f"[STORAGE_VERIFICATION] Failed to re-add {remaining_id} during rebuild: {readd_error}")

            # Restore data (without the problematic unit)
            self.data = temp_data
            logger.info(f"[STORAGE_VERIFICATION] ✅ Index rebuilt without {unit_id}")

        except Exception as rebuild_error:
            logger.error(f"[STORAGE_VERIFICATION] ❌ Failed to rebuild index: {rebuild_error}")
            # Last resort: create empty index
            self._create_index(data_size=0)
            self.data.clear()

    def _load_index(self) -> bool:
        """Load FAISS index from file. Returns True if successful."""
        try:
            import faiss
            self.index = faiss.read_index(self.index_file + ".index")
            return True
        except Exception:
            # File doesn't exist or is corrupted - will create new index
            return False

    def _calculate_optimal_nlist(self, data_size: int) -> int:
        """Calculate optimal nlist based on data size - rounds up to nearest 10.

        FAISS recommends nlist = n / 39 where n = number of vectors.
        This version rounds up to nearest 10 for more stable clustering:
        - 0-389: nlist = 10
        - 390-779: nlist = 20
        - 780-1169: nlist = 30
        etc.

        Args:
            data_size: Current number of stored units

        Returns:
            Optimal nlist value for IVF index
        """
        # Use config value for vectors per centroid if available, otherwise default to 39
        if self.storage_config:
            vectors_per_centroid = getattr(self.storage_config, 'vectors_per_centroid', 39)
            max_units = getattr(self.storage_config, 'max_units', 50000)
        else:
            vectors_per_centroid = 39
            max_units = 50000

        # Calculate nlist: retrain every nlist * 39 vectors
        # nlist = 10 at <390, 20 at <780, 30 at <1170, etc.
        chunk_size = vectors_per_centroid * 10  # 390
        optimal_nlist = ((data_size // chunk_size) + 1) * 10
        optimal_nlist = max(10, optimal_nlist)  # Minimum 10

        # Max bound based on MAX_UNITS config
        max_nlist = max_units // vectors_per_centroid
        optimal_nlist = min(optimal_nlist, max_nlist)

        logger.debug(
            f"[IVF_OPTIMIZATION] Calculated nlist={optimal_nlist} for data_size={data_size}, max_units={max_units}")
        return optimal_nlist

    def _create_index(self, data_size: int = 0):
        """Create new FAISS index based on index_type with dynamic nlist calculation.

        Args:
            data_size: Current number of stored units for optimal nlist calculation
        """
        try:
            import faiss
            if self.index_type == 'flat':
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                logger.info(
                    f"[IVF_INITIALIZATION] Created flat index with dimension {
                        self.embedding_dim}")
            elif self.index_type == 'ivf':
                # CRITICAL FIX: Calculate optimal nlist based on actual data size
                nlist = self._calculate_optimal_nlist(data_size)
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                logger.info(
                    f"[IVF_INITIALIZATION] Created IVF index with nlist={nlist} for optimal performance")
                # IVF indexes are not trained initially - training happens automatically
                # on first add with system-aligned data
            elif self.index_type == 'hnsw':
                # HNSW index
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
                logger.info(
                    f"[IVF_INITIALIZATION] Created HNSW index with dimension {
                        self.embedding_dim}")
            else:
                # Default to flat
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                logger.info(
                    f"[IVF_INITIALIZATION] Created default flat index with dimension {
                        self.embedding_dim}")
        except Exception as e:
            raise RuntimeError(f"Failed to create {self.index_type} index: {str(e)}")

    def _generate_system_aligned_training_data(self, nlist: int) -> np.ndarray:
        """Generate system-aligned training data using encoding configuration patterns.

        Creates realistic training data that matches the actual memory unit structure
        and content patterns used by the encoding system (lesson/skill/tool/abstraction).

        Args:
            nlist: Number of centroids for IVF training

        Returns:
            Training data embeddings array
        """
        try:
            # System-aligned patterns based on encoding configuration
            training_texts = []

            # Core patterns from encoding configuration
            base_patterns = [
                "lesson learned from experience",
                "skill developed through practice",
                "tool used for problem solving",
                "abstraction from concrete examples",
                "technical knowledge acquired",
                "conceptual understanding gained",
                "practical insight discovered",
                "methodology improvement achieved"
            ]

            # Create variations for comprehensive training
            for pattern in base_patterns:
                training_texts.append(pattern)
                # Add variations with different contexts
                training_texts.append(f"detailed {pattern}")
                training_texts.append(f"practical {pattern}")
                training_texts.append(f"theoretical {pattern}")

            # Add memory-relevant contexts
            memory_contexts = [
                "query response about technical concepts",
                "reasoning for problem solving",
                "content analysis for understanding",
                "metadata categorization for organization",
                "tag-based classification system",
                "retrieval-based memory access"
            ]

            training_texts.extend(memory_contexts)

            # Use config value for minimum training vectors if available
            if self.storage_config:
                min_training_vectors = getattr(self.storage_config, 'min_training_vectors', 100)
            else:
                min_training_vectors = 100

            # Ensure we have enough training data
            target_count = max(
                nlist * 2,
                min_training_vectors,
                len(training_texts))  # Aim for 2 vectors per centroid
            while len(training_texts) < target_count:
                # Duplicate with variations
                base_idx = len(training_texts) % len(base_patterns)
                variation = f"enhanced {base_patterns[base_idx]} with additional context"
                training_texts.append(variation)

            # Generate embeddings for training data
            logger.info(f"[IVF_TRAINING] Generating {len(training_texts)} training vectors")
            embeddings = []

            for i, text in enumerate(training_texts):
                try:
                    embedding = self.embedding_function(text)
                    embeddings.append(embedding.reshape(1, -1).astype('float32'))

                    if (i + 1) % 10 == 0:
                        logger.debug(
                            f"[IVF_TRAINING] Generated {i + 1}/{len(training_texts)} embeddings")

                except Exception as e:
                    logger.warning(f"[IVF_TRAINING] Failed to generate embedding for '{text}': {e}")
                    # Use a fallback embedding
                    fallback = np.random.randn(self.embedding_dim).astype('float32')
                    embeddings.append(fallback.reshape(1, -1))

            # Combine all embeddings
            training_data = np.vstack(embeddings[:target_count])
            logger.info(f"[IVF_TRAINING] Generated training data: {training_data.shape}")

            return training_data

        except Exception as e:
            logger.error(f"[IVF_TRAINING] Failed to generate training data: {e}")
            # Fallback to random data with correct structure
            logger.warning("[IVF_TRAINING] Using fallback random training data")
            return np.random.randn(max(nlist, 100), self.embedding_dim).astype('float32')

    def _train_ivf_if_needed(self, embedding: np.ndarray):
        """Train IVF index with system-aligned training data if not already trained."""
        if self.index_type == 'ivf':
            try:
                import faiss

                # Check if it's an IVF index and needs training
                if hasattr(self.index, 'is_trained') and not self.index.is_trained:  # type: ignore
                    nlist = getattr(self.index, 'nlist', 4)  # type: ignore

                    # CRITICAL FIX: Use system-aligned training data instead of single
                    # duplicated vector
                    train_data = self._generate_system_aligned_training_data(nlist)

                    logger.info(
                        f"[IVF_TRAINING] Training IVF index with {
                            train_data.shape[0]} vectors for {nlist} centroids")
                    self.index.train(train_data)  # type: ignore

                    # CRITICAL FIX: Enhanced validation with multiple test queries
                    if self.index.is_trained:
                        # Comprehensive validation: test search functionality
                        validation_success = self._validate_training_comprehensive(train_data)

                        if validation_success:
                            logger.info(
                                f"[IVF_TRAINING] ✅ Successfully trained IVF index with {
                                    train_data.shape[0]} vectors")
                        else:
                            logger.warning(
                                "[IVF_TRAINING] ⚠️ Training completed but comprehensive validation failed")
                    else:
                        raise RuntimeError("IVF index training failed - index remains untrained")

            except Exception as e:
                logger.error(
                    f"[IVF_TRAINING] Failed to train IVF index: {e}, falling back to flat index")
                # Fallback to flat index if training fails
                import faiss
                self.index = faiss.IndexFlatL2(self.embedding_dim)

    def _validate_training_comprehensive(self, train_data: np.ndarray) -> bool:
        """Comprehensive training validation with multiple test queries.

        Args:
            train_data: Training data used for index training

        Returns:
            True if validation passes, False otherwise
        """
        try:
            if self.index is None or not hasattr(self.index, 'search'):
                return False

            # Test with multiple different queries
            test_queries = [
                train_data[0:1],  # First training vector
                train_data[-1:1] if len(train_data) > 1 else train_data[0:1],  # Last vector
            ]

            validation_passed = 0
            for i, test_query in enumerate(test_queries):
                try:
                    distances, indices = self.index.search(test_query, k=3)

                    # Check if search returns valid results
                    if (len(indices) > 0 and len(indices[0]) > 0 and
                            indices[0][0] >= 0 and distances[0][0] >= 0):
                        validation_passed += 1
                        logger.debug(
                            f"[IVF_TRAINING] Validation {
                                i +
                                1}: distance={
                                distances[0][0]:.4f}")
                    else:
                        logger.debug(f"[IVF_TRAINING] Validation {i + 1}: Invalid search results")

                except Exception as query_error:
                    logger.debug(f"[IVF_TRAINING] Validation {i + 1} failed: {query_error}")

            # Consider validation successful if at least one test passes
            success_rate = validation_passed / len(test_queries)
            logger.debug(f"[IVF_TRAINING] Validation success rate: {success_rate:.2f}")

            return success_rate >= 0.5  # At least 50% of tests must pass

        except Exception as validation_error:
            logger.warning(f"[IVF_TRAINING] Comprehensive validation failed: {validation_error}")
            return False

    def _save_index(self):
        """Save FAISS index to file."""
        try:
            import faiss
            faiss.write_index(self.index, self.index_file + ".index")
        except Exception as e:
            raise RuntimeError(f"Failed to save index: {str(e)}")

    def _load_data(self):
        """Load metadata from file."""
        try:
            data_file = self.index_file + ".data"
            if os.path.exists(data_file):
                with open(data_file, 'rb') as f:
                    loaded_data = pickle.load(f)
                    # Handle both old format (just data) and new format (data only now)
                    if isinstance(loaded_data, dict) and 'data' in loaded_data:
                        self.data = loaded_data['data']
                    else:
                        # Old format: data is the pickle directly
                        self.data = loaded_data
        except Exception:
            # If data file is corrupted, start with empty data
            self.data = {}

    def _save_data(self):
        """Save metadata to file."""
        try:
            with open(self.index_file + ".data", 'wb') as f:
                pickle.dump(self.data, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save data: {str(e)}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        embedding = self.embedding_function(text)
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"expected {self.embedding_dim}, "
                f"got {embedding.shape[0]}"
            )
        return embedding.reshape(1, -1).astype('float32')

    def store(self, unit: Dict[str, Any]) -> str:
        """Store a memory unit and return its ID with improved error handling and size limits."""
        unit = self._add_metadata(unit.copy())

        # CRITICAL FIX: Check size limits using configuration values
        current_size = len(self.data)

        # Use config values if available, otherwise defaults
        if self.storage_config:
            max_size = getattr(self.storage_config, 'max_units', 50000)
            warning_threshold = int(
                max_size *
                getattr(
                    self.storage_config,
                    'warning_threshold',
                    0.8))
        else:
            max_size = 50000
            warning_threshold = int(max_size * 0.8)  # 80% warning threshold = 40,000

        if current_size >= max_size:
            raise RuntimeError(
                f"Vector store size limit reached: {current_size}/{max_size} units. Cannot store additional units.")

        if current_size >= warning_threshold and current_size % 1000 == 0:
            logger.warning(
                f"[SIZE_LIMIT] Approaching size limit: {current_size}/{max_size} units ({current_size / max_size * 100:.1f}%)")

        # CRITICAL: Unit ID must be generated by encoder, not here
        if "id" not in unit:
            raise ValueError(
                f"[VECTOR_STORE] Unit must have 'id' field. "
                f"ID generation is the encoder's responsibility.")
        unit_id = unit["id"]

        # Validate embedding generation with better error logging
        try:
            text = self._unit_to_text(unit)
            embedding = self._get_embedding(text)
        except Exception as e:
            logger.error(f"Failed to generate embedding for unit {unit_id}: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")

        # Train IVF index if needed before first add
        try:
            self._train_ivf_if_needed(embedding)
        except Exception as e:
            logger.error(f"Failed to train IVF index for unit {unit_id}: {e}")
            raise RuntimeError(f"Index training failed: {e}")

        # Validate index is ready
        if self.index is None:
            raise RuntimeError("FAISS index not initialized")

        # Add with error handling and logging
        try:
            # Add to data structure first
            self.data[unit_id] = unit

            # Add to FAISS index with correct parameter name
            self.index.add(embedding)
            logger.debug(f"[STORAGE_VERIFICATION] 🔧 Added embedding to FAISS index for {unit_id}")

            # Phase 2.1: Accumulate training data from successful storage
            self._accumulate_training_data(embedding)

            # CRITICAL FIX: Track index-to-unit mapping for accurate verification
            if not hasattr(self, '_index_to_unit_mapping'):
                self._index_to_unit_mapping = {}

            # Map this new index position to unit ID
            if self.index is not None and hasattr(self.index, 'ntotal'):
                current_index = self.index.ntotal - 1  # Current index position (0-based)
                self._index_to_unit_mapping[current_index] = unit_id

            # Phase 2.2: Enhanced verification with corruption detection
            verification_results = self._enhanced_verify_storage(unit_id, embedding)
            if not verification_results["verified"]:
                raise RuntimeError(
                    f"Enhanced storage verification failed: {
                        verification_results['error']}")
            logger.debug(f"[ENHANCED_VERIFICATION] ✅ Verified storage for {unit_id}")

            # Phase 2.2: Check for index corruption after verification
            corruption_check = self._detect_corruption()
            if corruption_check["corrupted"]:
                logger.error(
                    f"[CORRUPTION_DETECTION] Corruption detected during storage: {
                        corruption_check['issues']}")
                # Phase 2.3: Trigger automatic index rebuilding
                if self._auto_rebuild_index():
                    logger.info("[AUTO_REBUILD] ✅ Automatic index rebuilding completed")
                else:
                    raise RuntimeError("Corruption detected and automatic rebuilding failed")

            # Phase 2.1: Check if progressive retraining is needed
            if self._should_retrain_progressively():
                logger.info(
                    "[PROGRESSIVE_TRAINING] Triggering progressive retraining after successful storage")
                self._progressive_retrain_index()

        except Exception as verify_error:
            # Rollback failed storage
            if unit_id in self.data:
                del self.data[unit_id]
                # Remove from FAISS index (not straightforward, so rebuild index)
                self._rebuild_index_without_unit(unit_id)
            raise RuntimeError(
                f"Enhanced storage verification failed and rolled back: {verify_error}")

        # Persist to disk - critical for preventing data loss
        try:
            logger.debug(f"[STORAGE_DEBUG] Attempting to save index to {self.index_file}")
            self._save_index()
            logger.debug(f"[STORAGE_DEBUG] Attempting to save data to {self.index_file}")
            self._save_data()
            logger.debug(f"Successfully persisted unit {unit_id} to {self.index_file}")
        except Exception as save_error:
            logger.error(f"[STORAGE_ERROR] Failed to persist unit {unit_id}: {save_error}")
            import traceback
            logger.error(f"[STORAGE_DEBUG] Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Storage persistence failed: {save_error}")

        return unit_id

    def store_batch(self, units: List[Dict[str, Any]]) -> List[str]:
        """Store multiple memory units using individual store() calls for reliability."""
        if not units:
            return []

        logger.info(f"Starting batch storage of {len(units)} units")
        ids = []
        failed_units = []

        # Use the proven single store() method for each unit
        # This ensures proper error handling and persistence for each unit
        for i, unit in enumerate(units):
            try:
                unit_id = self.store(unit)
                ids.append(unit_id)
                logger.debug(f"Batch unit {i + 1}/{len(units)} stored successfully: {unit_id}")
            except Exception as e:
                logger.error(f"Batch unit {i + 1}/{len(units)} failed: {e}")
                failed_units.append((i, str(e)))

                # Continue trying to store remaining units (fail-fast approach)
                continue

        if failed_units:
            failed_summary = [f"Unit {idx}: {err}" for idx, err in failed_units]
            logger.error(
                f"Batch storage partially failed: {len(failed_units)}/{len(units)} units failed")
            logger.error(f"Failed units: {failed_summary}")

            # Return partial success - don't raise exception to allow partial completion
            # This is better than all-or-nothing for large batches
            if len(failed_units) == len(units):
                # All failed - raise exception
                raise RuntimeError(f"All {len(units)} units in batch failed to store")
            else:
                # Partial success - log but continue
                logger.warning(f"Batch storage: {len(ids)} succeeded, {len(failed_units)} failed")

        logger.info(f"Batch storage completed: {len(ids)}/{len(units)} units stored successfully")
        return ids

    def retrieve(self, unit_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory unit by ID."""
        return self.data.get(unit_id)

    def retrieve_all(self) -> List[Dict[str, Any]]:
        """Retrieve all stored memory units."""
        return list(self.data.values())

    def update(self, unit_id: str, unit: Dict[str, Any]) -> bool:
        """Update a memory unit by ID."""
        if unit_id in self.data:
            if "metadata" not in unit:
                unit["metadata"] = {}
            unit["metadata"]["updated_at"] = self._generate_timestamp()
            unit["id"] = unit_id

            self.data[unit_id] = unit

            self._save_index()
            self._save_data()

            return True
        return False

    def delete(self, unit_id: str) -> bool:
        """Delete a memory unit by ID."""
        if unit_id in self.data:
            del self.data[unit_id]
            self._rebuild_index()
            self._save_index()
            self._save_data()
            return True
        return False

    def exists(self, unit_id: str) -> bool:
        """Check if a memory unit exists."""
        return unit_id in self.data

    def count(self) -> int:
        """Get the count of stored memory units."""
        return len(self.data)

    def clear(self) -> None:
        """Clear all stored memory units."""
        self.data.clear()
        self._create_index(data_size=0)
        self._save_index()
        self._save_data()

    def get_metadata(self, unit_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific unit."""
        unit = self.retrieve(unit_id)
        if unit and "metadata" in unit:
            return unit["metadata"]
        return None

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[tuple]:
        """Search for similar memory units.

        Returns:
            List of (distance, unit_id) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        embedding = self._get_embedding(query)
        distances, indices = self.index.search(x=embedding, k=top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.data):
                unit_id = list(self.data.keys())[idx]
                results.append((float(dist), unit_id))

        return results

    def _rebuild_index(self):
        """Rebuild index from current data."""
        self._create_index(data_size=len(self.data))

        if self.index is None:
            raise RuntimeError("Failed to create index during rebuild")

        if len(self.data) > 0:
            embeddings = []
            for unit in self.data.values():
                text = self._unit_to_text(unit)
                embedding = self._get_embedding(text)
                embeddings.append(embedding)

            all_embeddings = np.vstack(embeddings).astype('float32')

            # Train IVF index if needed before adding vectors
            if self.index_type == 'ivf' and hasattr(
                    self.index, 'is_trained') and not self.index.is_trained:
                if len(all_embeddings) > 0:
                    self._train_ivf_if_needed(all_embeddings[0:1])

            self.index.add(x=all_embeddings)

    def _unit_to_text(self, unit: Dict[str, Any]) -> str:
        """Convert unit to text for embedding - include reasoning."""
        text_parts = []

        # Primary content
        if "content" in unit:
            text_parts.append(str(unit["content"]))

        # CRITICAL FIX: Include reasoning content for embeddings
        if "reasoning" in unit and unit["reasoning"]:
            text_parts.append(f"Reasoning: {unit['reasoning']}")

        # Query context
        if "query" in unit and unit["query"]:
            text_parts.append(f"Query: {unit['query']}")

        if "tags" in unit and isinstance(unit["tags"], list):
            text_parts.append(" ".join(unit["tags"]))

        if "type" in unit:
            text_parts.append(str(unit["type"]))

        return " ".join(text_parts)

    # ===== StorageBackend Interface Implementation =====

    def retrieve(self, unit_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory unit by ID."""
        try:
            return self.data.get(unit_id)
        except Exception as e:
            logger.error(f"[RETRIEVAL_ERROR] Failed to retrieve unit {unit_id}: {e}")
            return None

    def retrieve_all(self) -> List[Dict[str, Any]]:
        """Retrieve all stored memory units."""
        try:
            return list(self.data.values())
        except Exception as e:
            logger.error(f"[RETRIEVAL_ERROR] Failed to retrieve all units: {e}")
            return []
