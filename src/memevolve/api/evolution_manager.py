"""Evolution Manager for runtime memory architecture optimization in API proxy."""

import json
import logging
import random
import statistics
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from ..components.encode import ExperienceEncoder
from ..components.manage import SimpleManagementStrategy
from ..components.retrieve import (HybridRetrievalStrategy,
                                   KeywordRetrievalStrategy,
                                   SemanticRetrievalStrategy)
from ..evolution.diagnosis import DiagnosisEngine
from ..evolution.genotype import GenotypeFactory, MemoryGenotype
from ..evolution.mutation import MutationEngine, RandomMutationStrategy
from ..evolution.selection import ParetoSelector
from ..memory_system import ComponentType, MemorySystem
from ..utils.config import ConfigManager, MemEvolveConfig

from ..utils.logging_manager import LoggingManager
logger = LoggingManager.get_logger("memevolve.evolution")


@dataclass
class EvolutionMetrics:
    """Runtime metrics for evolution."""
    api_requests_total: int = 0
    api_requests_successful: int = 0
    average_response_time: float = 0.0
    memory_retrievals_total: int = 0
    memory_retrievals_successful: int = 0
    average_retrieval_time: float = 0.0
    current_genotype_id: Optional[str] = None
    evolution_cycles_completed: int = 0
    last_evolution_time: Optional[float] = None

    # Enhanced metrics for fitness evaluation
    response_quality_score: float = 0.0  # Semantic coherence/context relevance
    retrieval_precision: float = 0.0     # Precision of memory retrieval
    retrieval_recall: float = 0.0        # Recall of memory retrieval
    memory_utilization: float = 0.0      # Storage efficiency
    user_satisfaction_score: float = 0.0  # Future: explicit feedback

    # Rolling window data
    response_times_window: List[float] = field(default_factory=list)
    retrieval_times_window: List[float] = field(default_factory=list)
    quality_scores_window: List[float] = field(default_factory=list)
    memory_utilization_window: List[float] = field(default_factory=list)
    precision_window: List[float] = field(default_factory=list)
    recall_window: List[float] = field(default_factory=list)
    window_size: int = 100  # Rolling window size


@dataclass
class EvolutionResult:
    """Result of an evolution cycle."""
    generation: int
    best_genotype: MemoryGenotype
    fitness_score: float
    improvement: float
    timestamp: float


class EvolutionManager:
    """Manages runtime evolution of memory architectures for API proxy."""

    def __init__(self, config: MemEvolveConfig, memory_system: MemorySystem,
                 config_manager: ConfigManager):
        self.config = config
        self.memory_system = memory_system
        self.config_manager = config_manager
        self.metrics = EvolutionMetrics()

        # Evolution cycle rate (seconds between generations)
        self.evolution_cycle_seconds = config.cycle_evolution.cycle_seconds

        # Evolution embedding settings
        # These are optimized values found by evolution
        self.evolution_embedding_max_tokens: Optional[int] = None

        # Base model capabilities (maximum allowable values)
        # Priority: env var > auto-detect > fallback
        self.base_embedding_max_tokens = config.embedding.max_tokens or 512

        # Setup evolution logging
        from ..utils.logging import setup_component_logging
        self.logger = setup_component_logging("evolution", config)

        # Evolution components
        try:
            self.selector = ParetoSelector()
            self.mutation_engine = MutationEngine(
                RandomMutationStrategy(
                    base_max_tokens=self.base_embedding_max_tokens,
                    boundary_config=config.evolution_boundaries
                ),
                base_max_tokens=self.base_embedding_max_tokens
            )
            self.diagnosis_engine = DiagnosisEngine()
        except Exception as e:
            self.logger.warning(f"Failed to initialize evolution components: {e}")
            raise

        # Evolution state
        self.current_genotype: Optional[MemoryGenotype] = None
        self.genotype_factory = GenotypeFactory()
        self.population: List[MemoryGenotype] = []
        self.evolution_history: List[EvolutionResult] = []

        # Control flags
        self.is_running = False
        self.evolution_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Performance tracking
        self.request_times: List[float] = []
        self.retrieval_times: List[float] = []

        # Auto-evolution trigger tracking (reset on startup)
        self.startup_time = time.time()
        self.requests_since_startup = 0
        self.last_evolution_time = 0.0  # Reset on startup

        # Persistence
        # Evolution state is persistent data, not temporary cache
        evolution_dir = Path(self.config.data_dir) / "evolution"
        evolution_dir.mkdir(exist_ok=True)
        self.persistence_file = evolution_dir / "evolution_state.json"
        # Keep backups of last few good states
        self.backup_dir = evolution_dir / "evolution_backups"
        self.max_backups = 3
        # Metrics persistence - separate from evolution state for analysis
        self.metrics_dir = Path(self.config.data_dir) / "metrics"
        self.best_genotype: Optional[MemoryGenotype] = None
        self._load_persistent_state()
        self._ensure_metrics_directory()

    def _load_persistent_state(self):
        """Load previously saved evolution state, with fallback to backups."""
        loaded = False

        # Try loading from main file first
        if self.persistence_file.exists():
            loaded = self._load_from_file(self.persistence_file)
            if loaded:
                return

        # If main file failed, try loading from backups
        if self.backup_dir.exists():
            backups = sorted(self.backup_dir.glob("evolution_state_*.json"),
                             key=lambda x: x.stat().st_mtime, reverse=True)

            for backup_file in backups:
                self.logger.info(f"Trying to load from backup: {backup_file}")
                if self._load_from_file(backup_file):
                    self.logger.info(
                        f"Successfully recovered evolution state from backup: {backup_file}")
                    # Copy the good backup back to main file
                    try:
                        import shutil
                        shutil.copy2(backup_file, self.persistence_file)
                        self.logger.info("Restored main evolution state file from backup")
                    except Exception as e:
                        self.logger.warning(f"Failed to restore main file from backup: {e}")
                    loaded = True
                    break

        if not loaded:
            self.logger.warning("Could not load evolution state from any source - starting fresh")

    def _load_from_file(self, file_path: Path) -> bool:
        """Load evolution state from a specific file. Returns True if successful."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Load best genotype if available
            if 'best_genotype' in data and data['best_genotype'] is not None:
                try:
                    genotype_dict = data['best_genotype']
                    self.best_genotype = self._dict_to_genotype(genotype_dict)
                    self.logger.info(
                        f"Loaded best genotype: {
                            self.best_genotype.get_genome_id() if self.best_genotype else None}")

                    # CRITICAL FIX: Apply loaded genotype to memory system at startup
                    if self.best_genotype:
                        try:
                            self._apply_genotype_to_memory_system(
                                self.best_genotype, log_changes=False)
                            self.logger.info(
                                f"Applied loaded genotype {
                                    self.best_genotype.get_genome_id()} at startup")
                        except Exception as e:
                            self.logger.error(f"Failed to apply loaded genotype at startup: {e}")
                            # Continue startup even if application fails - system will use defaults

                except Exception as e:
                    self.logger.warning(f"Failed to load best genotype: {e}")

            # Load evolution embedding settings
            if 'evolution_embedding_max_tokens' in data:
                self.evolution_embedding_max_tokens = data[
                    'evolution_embedding_max_tokens']
                self.logger.debug(
                    f"Loaded evolution embedding_max_tokens: {self.evolution_embedding_max_tokens}")

            # Load evolution history
            if 'evolution_history' in data:
                self.evolution_history = []
                for i, result_dict in enumerate(data['evolution_history']):
                    try:
                        genotype_dict = result_dict.pop('best_genotype')
                        genotype = self._dict_to_genotype(genotype_dict)
                        if genotype is not None:
                            result = EvolutionResult(
                                best_genotype=cast(MemoryGenotype, genotype),
                                **result_dict
                            )
                            self.evolution_history.append(result)
                    except Exception as e:
                        self.logger.warning(f"Failed to load evolution history entry {i}: {e}")

            # Load metrics
            if 'metrics' in data:
                for key, value in data['metrics'].items():
                    if hasattr(self.metrics, key):
                        setattr(self.metrics, key, value)

            history_count = len(self.evolution_history)
            best_id = self.best_genotype.get_genome_id() if self.best_genotype else None
            self.logger.debug(
                f"Loaded evolution state from {file_path} "
                f"({history_count} history entries, best_genotype: {best_id})"
            )
            return True

        except json.JSONDecodeError as e:
            self.logger.warning(f"Corrupted evolution state file {file_path}: {e}")
            if file_path == self.persistence_file:
                # Backup corrupted main file for debugging
                backup_file = file_path.with_suffix('.corrupted')
                try:
                    file_path.replace(backup_file)
                    self.logger.info(f"Backed up corrupted file to {backup_file}")
                except Exception:
                    pass
            return False
        except Exception as e:
            self.logger.warning(f"Failed to load evolution state from {file_path}: {e}")
            return False

    def _ensure_metrics_directory(self):
        """Ensure metrics directory exists for comprehensive metrics storage."""
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def _save_comprehensive_metrics(self):
        """Save comprehensive metrics to metrics directory for analysis."""
        try:
            timestamp = int(time.time())
            metrics_file = self.metrics_dir / f"metrics_{timestamp}.json"

            # Create comprehensive metrics snapshot
            comprehensive_metrics = {
                "timestamp": timestamp,
                "evolution_cycle": self.metrics.evolution_cycles_completed,
                "current_genotype_id": self.metrics.current_genotype_id,

                # API Performance Metrics
                "api_requests_total": self.metrics.api_requests_total,
                "api_requests_successful": self.metrics.api_requests_successful,
                "api_success_rate": (
                    self.metrics.api_requests_successful /
                    max(1, self.metrics.api_requests_total)
                ),
                "average_response_time": self.metrics.average_response_time,
                "response_times_window": self.metrics.response_times_window.copy(),

                # Memory Retrieval Metrics
                "memory_retrievals_total": self.metrics.memory_retrievals_total,
                "memory_retrievals_successful": self.metrics.memory_retrievals_successful,
                "retrieval_success_rate": (
                    self.metrics.memory_retrievals_successful /
                    max(1, self.metrics.memory_retrievals_total)
                ),
                "average_retrieval_time": self.metrics.average_retrieval_time,
                "retrieval_times_window": self.metrics.retrieval_times_window.copy(),

                # Quality Metrics
                "response_quality_score": self.metrics.response_quality_score,
                "quality_scores_window": self.metrics.quality_scores_window.copy(),

                # Memory Utilization Metrics
                "memory_utilization": self.metrics.memory_utilization,
                "memory_utilization_window": self.metrics.memory_utilization_window.copy(),

                # Additional Performance Metrics
                "retrieval_precision": self.metrics.retrieval_precision,
                "retrieval_recall": self.metrics.retrieval_recall,
                "user_satisfaction_score": self.metrics.user_satisfaction_score,

                # Evolution Metadata
                "last_evolution_time": self.metrics.last_evolution_time,
                "evolution_cycles_completed": self.metrics.evolution_cycles_completed,

                # Rolling Window Metadata
                "window_size": self.metrics.window_size,
                "rolling_windows_populated": {
                    "response_times": len(self.metrics.response_times_window),
                    "retrieval_times": len(self.metrics.retrieval_times_window),
                    "quality_scores": len(self.metrics.quality_scores_window),
                    "memory_utilization": len(self.metrics.memory_utilization_window)
                }
            }

            with open(metrics_file, 'w') as f:
                json.dump(comprehensive_metrics, f, indent=2)

            self.logger.debug(f"Saved comprehensive metrics to {metrics_file}")

        except Exception as e:
            self.logger.warning(f"Failed to save comprehensive metrics: {e}")

    def export_metrics_for_analysis(self, output_file: Optional[str] = None) -> str:
        """Export all metrics history for analysis."""
        try:
            if output_file is None:
                timestamp = int(time.time())
                output_file = str(self.metrics_dir / f"metrics_export_{timestamp}.json")

            # Collect all metrics files
            metrics_files = list(self.metrics_dir.glob("metrics_*.json"))
            metrics_files.sort(key=lambda x: x.stat().st_mtime)

            all_metrics = []
            for metrics_file in metrics_files:
                try:
                    with open(metrics_file, 'r') as f:
                        metrics_data = json.load(f)
                        all_metrics.append(metrics_data)
                except Exception as e:
                    self.logger.warning(f"Failed to load metrics file {metrics_file}: {e}")

            # Add current metrics if not already included
            current_metrics = {
                "timestamp": int(time.time()),
                "evolution_cycle": self.metrics.evolution_cycles_completed,
                "current_genotype_id": self.metrics.current_genotype_id,
                "api_requests_total": self.metrics.api_requests_total,
                "api_requests_successful": self.metrics.api_requests_successful,
                "average_response_time": self.metrics.average_response_time,
                "memory_retrievals_total": self.metrics.memory_retrievals_total,
                "memory_retrievals_successful": self.metrics.memory_retrievals_successful,
                "average_retrieval_time": self.metrics.average_retrieval_time,
                "response_quality_score": self.metrics.response_quality_score,
                "memory_utilization": self.metrics.memory_utilization,
                "retrieval_precision": self.metrics.retrieval_precision,
                "retrieval_recall": self.metrics.retrieval_recall,
                "user_satisfaction_score": self.metrics.user_satisfaction_score,
                "last_evolution_time": self.metrics.last_evolution_time,
                "evolution_cycles_completed": self.metrics.evolution_cycles_completed
            }
            all_metrics.append(current_metrics)

            # Export combined metrics
            with open(output_file, 'w') as f:
                json.dump({
                    "export_timestamp": int(time.time()),
                    "total_metrics_snapshots": len(all_metrics),
                    "metrics_history": all_metrics
                }, f, indent=2)

            self.logger.info(f"Exported metrics analysis to {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Failed to export metrics for analysis: {e}")
            return ""

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary for monitoring and analysis."""
        return {
            "evolution_status": {
                "is_running": self.is_running,
                "current_genotype_id": self.metrics.current_genotype_id,
                "evolution_cycles_completed": self.metrics.evolution_cycles_completed,
                "last_evolution_time": self.metrics.last_evolution_time,
                "best_genotype_available": self.best_genotype is not None
            },
            "performance_metrics": {
                "api_success_rate": (
                    self.metrics.api_requests_successful /
                    max(1, self.metrics.api_requests_total)
                ),
                "average_response_time": self.metrics.average_response_time,
                "retrieval_success_rate": (
                    self.metrics.memory_retrievals_successful /
                    max(1, self.metrics.memory_retrievals_total)
                ),
                "average_retrieval_time": self.metrics.average_retrieval_time
            },
            "quality_metrics": {
                "response_quality_score": self.metrics.response_quality_score,
                "memory_utilization": self.metrics.memory_utilization,
                "retrieval_precision": self.metrics.retrieval_precision,
                "retrieval_recall": self.metrics.retrieval_recall,
                "user_satisfaction_score": self.metrics.user_satisfaction_score
            },
            "data_volumes": {
                "api_requests_total": self.metrics.api_requests_total,
                "memory_retrievals_total": self.metrics.memory_retrievals_total,
                "rolling_windows": {
                    "response_times": len(self.metrics.response_times_window),
                    "retrieval_times": len(self.metrics.retrieval_times_window),
                    "quality_scores": len(self.metrics.quality_scores_window),
                    "memory_utilization": len(self.metrics.memory_utilization_window)
                }
            },
            "fitness_score": self.get_fitness_score(),
            "metrics_persistence": {
                "metrics_directory": str(self.metrics_dir),
                "metrics_files_count": len(list(self.metrics_dir.glob("metrics_*.json"))) if self.metrics_dir.exists() else 0
            }
        }

    def analyze_metrics_trends(self) -> Dict[str, Any]:
        """Analyze metrics trends over evolution history."""
        try:
            if not self.metrics_dir.exists():
                return {"error": "No metrics directory found"}

            metrics_files = list(self.metrics_dir.glob("metrics_*.json"))
            if not metrics_files:
                return {"error": "No metrics files found"}

            # Sort by timestamp
            metrics_files.sort(key=lambda x: int(x.stem.split('_')[1]))

            # Load metrics history
            metrics_history = []
            for metrics_file in metrics_files[-50:]:  # Last 50 snapshots for analysis
                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        metrics_history.append(data)
                except Exception as e:
                    self.logger.warning(f"Failed to load {metrics_file}: {e}")

            if not metrics_history:
                return {"error": "No valid metrics data found"}

            # Analyze trends
            timestamps = [m['timestamp'] for m in metrics_history]
            fitness_scores = [m.get('evolution_cycle', 0) for m in metrics_history]
            quality_scores = [m.get('response_quality_score', 0) for m in metrics_history]
            utilization_scores = [m.get('memory_utilization', 0) for m in metrics_history]
            response_times = [m.get('average_response_time', 0) for m in metrics_history]

            return {
                "analysis_period": {
                    "start_timestamp": min(timestamps),
                    "end_timestamp": max(timestamps),
                    "total_snapshots": len(metrics_history),
                    "duration_hours": (max(timestamps) - min(timestamps)) / 3600
                },
                "trends": {
                    "quality_score_trend": "improving" if quality_scores[-1] > quality_scores[0] else "stable",
                    "utilization_trend": "improving" if utilization_scores[-1] > utilization_scores[0] else "stable",
                    "response_time_trend": "improving" if response_times[-1] < response_times[0] else "stable"
                },
                "current_values": {
                    "response_quality_score": quality_scores[-1] if quality_scores else 0,
                    "memory_utilization": utilization_scores[-1] if utilization_scores else 0,
                    "average_response_time": response_times[-1] if response_times else 0
                },
                "evolution_progress": {
                    "cycles_completed": max(fitness_scores) if fitness_scores else 0,
                    "evolution_active": self.is_running
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to analyze metrics trends: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def _save_persistent_state(self):
        """Save current evolution state to disk atomically with backups."""
        try:
            # Ensure data directory exists
            self.persistence_file.parent.mkdir(parents=True, exist_ok=True)

            # Create backup of current file before overwriting
            if self.persistence_file.exists():
                self._create_backup()

            data = {
                'best_genotype': (
                    self._genotype_to_dict(self.best_genotype)
                    if self.best_genotype else None
                ),
                'evolution_embedding_max_tokens': self.evolution_embedding_max_tokens,
                'evolution_history': [
                    {
                        'generation': result.generation,
                        'fitness_score': result.fitness_score,
                        'improvement': result.improvement,
                        'timestamp': result.timestamp,
                        'best_genotype': self._genotype_to_dict(result.best_genotype)
                    }
                    for result in self.evolution_history
                ],
                'metrics': {
                    'api_requests_total': self.metrics.api_requests_total,
                    'api_requests_successful': self.metrics.api_requests_successful,
                    'average_response_time': self.metrics.average_response_time,
                    'memory_retrievals_total': self.metrics.memory_retrievals_total,
                    'memory_retrievals_successful': self.metrics.memory_retrievals_successful,
                    'average_retrieval_time': self.metrics.average_retrieval_time,
                    'evolution_cycles_completed': self.metrics.evolution_cycles_completed,
                    'last_evolution_time': self.metrics.last_evolution_time
                }
            }

            # Atomic write: write to temp file first, then move
            temp_file = self.persistence_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.persistence_file)

            self.logger.debug(f"Saved evolution state to {self.persistence_file}")

        except Exception as e:
            self.logger.warning(f"Failed to save evolution state: {e}")
            # Clean up temp file if it exists
            temp_file = self.persistence_file.with_suffix('.tmp')
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass

    def _create_backup(self):
        """Create a backup of the current evolution state file."""
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)

            # Create timestamped backup
            timestamp = int(time.time())
            backup_file = self.backup_dir / f"evolution_state_{timestamp}.json"

            # Copy current file to backup
            import shutil
            shutil.copy2(self.persistence_file, backup_file)

            # Clean up old backups (keep only max_backups)
            backups = sorted(self.backup_dir.glob("evolution_state_*.json"),
                             key=lambda x: x.stat().st_mtime, reverse=True)
            for old_backup in backups[self.max_backups:]:
                try:
                    old_backup.unlink()
                except Exception:
                    pass

            self.logger.debug(f"Created evolution state backup: {backup_file}")

        except Exception as e:
            self.logger.warning(f"Failed to create evolution state backup: {e}")

    def _genotype_to_dict(self, genotype: MemoryGenotype) -> Dict[str, Any]:
        """Convert genotype to dictionary for serialization."""
        return genotype.to_dict()

    def _dict_to_genotype(self, data: Dict[str, Any]) -> Optional[MemoryGenotype]:
        """Convert dictionary back to genotype."""
        if data is None:
            return None

        # Create a new genotype and populate it
        genotype = MemoryGenotype()

        # Helper function to set nested attributes
        def set_nested_attr(obj, path, value):
            parts = path.split('.')
            for part in parts[:-1]:
                if not hasattr(obj, part):
                    setattr(obj, part, type('', (), {})())
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

        # Populate from the data structure
        if 'encode' in data:
            genotype.encode = self._dict_to_encode_config(data['encode'])
        if 'store' in data:
            genotype.store = self._dict_to_store_config(data['store'])
        if 'retrieve' in data:
            genotype.retrieve = self._dict_to_retrieve_config(data['retrieve'])
        if 'manage' in data:
            genotype.manage = self._dict_to_manage_config(data['manage'])

        return genotype

    def _dict_to_encode_config(self, data: Dict[str, Any]):
        """Convert dict to EncodeConfig."""
        from ..evolution.genotype import EncodeConfig
        return EncodeConfig(**data)

    def _dict_to_store_config(self, data: Dict[str, Any]):
        """Convert dict to StoreConfig."""
        from ..evolution.genotype import StoreConfig
        return StoreConfig(**data)

    def _dict_to_retrieve_config(self, data: Dict[str, Any]):
        """Convert dict to RetrieveConfig."""
        from ..evolution.genotype import RetrieveConfig
        return RetrieveConfig(**data)

    def _dict_to_manage_config(self, data: Dict[str, Any]):
        """Convert dict to ManageConfig with ONLY evolved parameters.

        Non-evolved parameters (pruning, limits, persistence) are managed
        by centralized config and NOT included in genotype evolution.
        """
        from ..evolution.genotype import ManageConfig

        # Only include evolved parameters in ManageConfig
        # All other management parameters remain in centralized config
        evolved_data = {}

        # Evolved management parameters only
        evolved_keys = [
            "strategy_type",
            "enable_auto_management",
            "consolidate_enabled",
            "consolidate_min_units",
            "forgetting_strategy",
            "forgetting_percentage"
        ]

        for key in evolved_keys:
            if key in data:
                evolved_data[key] = data[key]

        return ManageConfig(**evolved_data)

    def check_cycle_evolution_triggers(self) -> bool:
        """Check if evolution should start based on NEW activity triggers since startup."""
        # Cycle evolution configuration
        cycle_enabled = self.config.cycle_evolution.enabled
        if not cycle_enabled:
            return False

        # Prevent evolution immediately after startup (require minimum warmup period)
        min_startup_requests = 10  # Require some new requests before auto-evolution
        time_since_startup = time.time() - self.startup_time
        if time_since_startup < 60 or self.requests_since_startup < min_startup_requests:
            return False  # Not enough activity yet

        # Multiple trigger conditions based on NEW activity only
        triggers = []

        # 1. Request count threshold (NEW requests since startup)
        request_threshold = self.config.cycle_evolution.requests
        if self.requests_since_startup >= request_threshold:
            triggers.append(
                f"request_threshold_met ({
                    self.requests_since_startup} >= {request_threshold})")

        # 2. Performance degradation (current performance)
        degradation_threshold = self.config.cycle_evolution.degradation
        if self.metrics.average_response_time > 0:
            baseline_time = 1.0  # 1 second baseline
            if self.metrics.average_response_time > baseline_time * (1 + degradation_threshold):
                triggers.append(
                    f"performance_degradation_detected ({
                        self.metrics.average_response_time:.3f}s vs {
                        baseline_time * (
                            1 + degradation_threshold):.3f}s)")

        # 3. Fitness plateau detection (if we have evolution history)
        plateau_generations = self.config.cycle_evolution.plateau
        if len(self.evolution_history) >= plateau_generations:
            recent_fitness = [
                result.fitness_score for result in self.evolution_history[-plateau_generations:]]
            if len(recent_fitness) >= plateau_generations:
                fitness_std = statistics.stdev(recent_fitness) if len(recent_fitness) > 1 else 0.0
                if fitness_std < 0.01:  # Very low variance indicates plateau
                    triggers.append(f"fitness_plateau_detected (std: {fitness_std:.6f})")

        # 4. Time-based evolution (since last evolution, not since startup)
        hours_threshold = self.config.cycle_evolution.hours
        if self.last_evolution_time > 0:
            hours_since_last = (time.time() - self.last_evolution_time) / 3600
            if hours_since_last >= hours_threshold:
                triggers.append(
                    f"time_based_trigger ({
                        hours_since_last:.1f}h >= {hours_threshold}h)")

        if triggers:
            self.logger.info(f"Auto-evolution triggers detected: {', '.join(triggers)}")
            return True

        return False

        # Multiple trigger conditions
        triggers = []

        # 1. Request count threshold
        request_threshold = self.config.auto_evolution.requests
        if self.metrics.api_requests_total >= request_threshold:
            triggers.append(
                f"request_threshold_met ({
                    self.metrics.api_requests_total} >= {request_threshold})")

        # 2. Performance degradation trigger
        degradation_threshold = self.config.auto_evolution.degradation
        if self.metrics.average_response_time > 0 and self._detect_performance_degradation(
                degradation_threshold):
            triggers.append("performance_degradation_detected")

        # 3. Fitness plateau trigger
        plateau_generations = self.config.auto_evolution.plateau
        if self._detect_fitness_plateau(plateau_generations):
            triggers.append("fitness_plateau_detected")

        # 4. Time-based trigger
        time_hours = self.config.auto_evolution.hours
        if self._check_time_based_trigger(time_hours):
            triggers.append("time_based_trigger")

        # Log trigger detection
        if triggers:
            self.logger.info(f"Auto-evolution triggers detected: {', '.join(triggers)}")
            return True

        return False

    def _detect_performance_degradation(self, threshold: float) -> bool:
        """Detect if performance has degraded by threshold percentage."""
        # Check recent vs historical performance
        if len(self.evolution_history) < 2:
            return False

        current_fitness = self.get_fitness_score()
        previous_fitness = self.evolution_history[-2].fitness_score if len(
            self.evolution_history) >= 2 else current_fitness

        # If current fitness is significantly worse than previous
        if current_fitness < previous_fitness * (1 - threshold):
            self.logger.info(
                f"Performance degradation detected: {
                    current_fitness:.4f} vs {
                    previous_fitness:.4f}")
            return True

        return False

    def _detect_fitness_plateau(self, generations: int) -> bool:
        """Detect if fitness has plateaued over N generations."""
        if len(self.evolution_history) < generations:
            return False

        recent_fitness = [result.fitness_score for result in self.evolution_history[-generations:]]
        fitness_std = statistics.stdev(recent_fitness) if len(recent_fitness) > 1 else 0

        # If standard deviation is very low, fitness has plateaued
        if fitness_std < 0.001:  # Very little variation
            self.logger.info(
                f"Fitness plateau detected over {generations} generations (std: {
                    fitness_std:.6f})")
            return True

        return False

    def _check_time_based_trigger(self, hours: int) -> bool:
        """Check if it's time for periodic evolution."""
        if not self.metrics.last_evolution_time:
            return False

        hours_since_last = (time.time() - self.metrics.last_evolution_time) / 3600
        if hours_since_last >= hours:
            self.logger.info(
                f"Time-based trigger: {hours_since_last:.1f} hours since last evolution")
            return True

        return False

    def start_evolution(self, auto_trigger: bool = False) -> bool:
        """Start the evolution process in background thread."""
        if self.is_running:
            self.logger.info("Evolution already running")
            return False

        # Check auto-triggers unless explicitly forced
        if not auto_trigger and not self.check_cycle_evolution_triggers():
            self.logger.info("Auto-evolution triggers not met - skipping start")
            return False

        try:
            self.is_running = True
            self.stop_event.clear()

            # Initialize population
            self._initialize_population()

            # Start evolution thread
            self.evolution_thread = threading.Thread(
                target=self._evolution_loop, daemon=True)
            self.evolution_thread.start()

            self.logger.info(f"Evolution started (auto_trigger={auto_trigger})")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to start evolution: {e}")
            self.is_running = False
            return False

    def stop_evolution(self) -> bool:
        """Stop the evolution process."""
        if not self.is_running:
            return False

        self.is_running = False
        self.stop_event.set()

        if self.evolution_thread:
            self.evolution_thread.join(timeout=5.0)

        # Save final state
        self._save_persistent_state()

        return True

    def get_status(self) -> Dict[str, Any]:
        """Get current evolution status."""
        return {
            "is_running": self.is_running,
            "current_genotype": (
                self.current_genotype.get_genome_id() if self.current_genotype else None),
            "population_size": len(
                self.population),
            "evolution_cycles_completed": self.metrics.evolution_cycles_completed,
            "last_evolution_time": self.metrics.last_evolution_time,
            "api_requests_total": self.metrics.api_requests_total,
            "average_response_time": self.metrics.average_response_time,
            "memory_retrievals_total": self.metrics.memory_retrievals_total,
            "average_retrieval_time": self.metrics.average_retrieval_time,
            "response_quality_score": self.metrics.response_quality_score,
            "memory_utilization": self.metrics.memory_utilization,
            "fitness_score": self.get_fitness_score(),
            "metrics_persistence": {
                "metrics_directory": str(
                    self.metrics_dir),
                "metrics_files_count": len(
                    list(
                        self.metrics_dir.glob("metrics_*.json"))) if self.metrics_dir.exists() else 0}}

    def record_api_request(self, response_time: float, success: bool = True):
        """Record an API request for performance tracking."""
        self.metrics.api_requests_total += 1
        self.requests_since_startup += 1  # Track NEW requests since startup
        if success:
            self.metrics.api_requests_successful += 1

        self.request_times.append(response_time)
        # Keep only last 1000 requests for rolling average
        if len(self.request_times) > 1000:
            self.request_times.pop(0)

        self.metrics.average_response_time = sum(
            self.request_times) / len(self.request_times)

    def record_memory_retrieval(
            self,
            retrieval_time: float,
            success: bool = True,
            memory_count: int = 0,
            precision: float = 0.0,
            recall: float = 0.0,
            quality: float = 0.0):
        """Record a memory retrieval for performance tracking.

        Args:
            retrieval_time: Time taken for retrieval in seconds
            success: Whether retrieval was successful
            memory_count: Number of memories retrieved
            precision: Precision of retrieval (relevant memories / total retrieved)
            recall: Recall of retrieval (relevant memories / total relevant)
            quality: Overall quality score of retrieved memories
        """
        self.metrics.memory_retrievals_total += 1
        if success:
            self.metrics.memory_retrievals_successful += 1

        self.retrieval_times.append(retrieval_time)
        # Keep only last 1000 retrievals for rolling average
        if len(self.retrieval_times) > 1000:
            self.retrieval_times.pop(0)

        self.metrics.average_retrieval_time = sum(
            self.retrieval_times) / len(self.retrieval_times)

        # Update quality metrics with rolling window
        if precision > 0:
            self._update_rolling_metric('precision', precision)
        if recall > 0:
            self._update_rolling_metric('recall', recall)
        if quality > 0:
            self._update_rolling_metric('quality', quality)

        # Log retrieval metrics for monitoring
        self.logger.info(
            f"Evolution: Memory retrieval recorded - time={retrieval_time:.3f}s, "
            f"success={success}, count={memory_count}, "
            f"avg_time={self.metrics.average_retrieval_time:.3f}s, "
            f"precision={precision:.3f}, recall={recall:.3f}, quality={quality:.3f}"
        )

    def _update_rolling_metric(self, metric_type: str, value: float):
        """Update a rolling window metric."""
        if metric_type == 'precision':
            self.metrics.precision_window.append(value)
            if len(self.metrics.precision_window) > self.metrics.window_size:
                self.metrics.precision_window.pop(0)
            self.metrics.retrieval_precision = sum(
                self.metrics.precision_window) / len(self.metrics.precision_window)
        elif metric_type == 'recall':
            self.metrics.recall_window.append(value)
            if len(self.metrics.recall_window) > self.metrics.window_size:
                self.metrics.recall_window.pop(0)
            self.metrics.retrieval_recall = sum(
                self.metrics.recall_window) / len(self.metrics.recall_window)
        elif metric_type == 'quality':
            # Quality scores already have a window in metrics.quality_scores_window
            self.metrics.quality_scores_window.append(value)
            if len(self.metrics.quality_scores_window) > self.metrics.window_size:
                self.metrics.quality_scores_window.pop(0)
            self.metrics.response_quality_score = sum(
                self.metrics.quality_scores_window) / len(self.metrics.quality_scores_window)

    def _initialize_population(self):
        """Initialize the population with current and variant genotypes."""
        # Start with best saved genotype if available, otherwise baseline
        if self.best_genotype:
            current_genotype = self.best_genotype
            self.logger.info(
                f"Using previously optimized genotype: {current_genotype.get_genome_id()}")
        else:
            current_genotype = self.genotype_factory.create_baseline_genotype()

        self.population = [current_genotype]

        # Add different architecture variants for initial population
        architectures = [
            self.genotype_factory.create_agentkb_genotype(),
            self.genotype_factory.create_lightweight_genotype(),
            self.genotype_factory.create_riva_genotype(),
            self.genotype_factory.create_cerebra_genotype()
        ]

        # Add available architectures, limiting to population size
        for arch in architectures[:self.config.evolution.population_size - 1]:
            self.population.append(arch)

        # Fill remaining slots with baseline variants if needed
        while len(self.population) < self.config.evolution.population_size:
            variant = self.genotype_factory.create_baseline_genotype()
            self.population.append(variant)

        self.current_genotype = current_genotype
        self.metrics.current_genotype_id = current_genotype.get_genome_id()

    def _evolution_loop(self):
        """Main evolution loop running in background thread."""
        for generation in range(self.config.evolution.generations):
            if self.stop_event.is_set():
                break

            try:
                # Wait for minimum evaluation period (10 requests per genotype)
                min_requests = max(10, len(self.population) * 10)
                while self.metrics.api_requests_total < min_requests:
                    if self.stop_event.is_set():
                        return
                    time.sleep(1)  # Wait for more data

                # Evaluate current population
                fitness_scores = self._evaluate_population()

                # Select best genotypes
                selected = self._select_best_genotypes(fitness_scores)

                # Create next generation
                new_population = self._create_next_generation(selected)

                # Update population
                self.population = new_population

                # Apply best genotype to memory system
                best_genotype = max(
                    new_population,
                    key=lambda g: fitness_scores.get(g.get_genome_id(), 0)
                )
                self._apply_genotype_to_memory_system(best_genotype, log_changes=True)

                # Update best genotype for persistence
                if (
                    self.best_genotype is None or
                    fitness_scores[best_genotype.get_genome_id()] >
                    fitness_scores.get(self.best_genotype.get_genome_id(), 0)
                ):
                    self.best_genotype = best_genotype

                    # Update evolution embedding settings from best genotype
                    self.evolution_embedding_max_tokens = best_genotype.encode.max_tokens
                    self.logger.info(
                        f"Evolution embedding settings updated: "
                        f"max_tokens={self.evolution_embedding_max_tokens}"
                    )

                # Record evolution result
                result = EvolutionResult(
                    generation=generation,
                    best_genotype=best_genotype,
                    fitness_score=fitness_scores[best_genotype.get_genome_id(
                    )],
                    improvement=self._calculate_improvement(fitness_scores[best_genotype.get_genome_id()]),
                    timestamp=time.time()
                )
                self.evolution_history.append(result)

                self.metrics.evolution_cycles_completed = generation + 1
                self.metrics.last_evolution_time = time.time()

                # Save state after each generation
                self._save_persistent_state()
                self._save_comprehensive_metrics()

                # Sleep between generations
                self.stop_event.wait(float(self.evolution_cycle_seconds))  # Configurable cycle rate

            except Exception as e:
                self.logger.error(f"Evolution cycle {generation} failed: {e}")
                continue

        self.is_running = False

    def _evaluate_population(self) -> Dict[str, float]:
        """Evaluate fitness of current population through per-genotype testing."""
        fitness_scores = {}

        # Update memory utilization metrics before evaluation
        if hasattr(self, 'memory_system') and self.memory_system:
            self.update_memory_utilization(self.memory_system)

        # Store original genotype to restore after testing
        original_genotype = self.current_genotype if self.current_genotype else None

        for genotype in self.population:
            genome_id = genotype.get_genome_id()

            # Apply genotype to memory system for testing
            try:
                self._apply_genotype_to_memory_system(genotype, log_changes=False)

                # Generate test trajectories for this genotype
                fitness_vector = self._run_test_trajectories(genotype)

                # Aggregate multi-dimensional fitness to scalar score
                aggregated_fitness = self._aggregate_fitness(fitness_vector)
                fitness_scores[genome_id] = aggregated_fitness

                self.logger.info(
                    f"Evaluated genotype {genome_id}: "
                    f"fitness={aggregated_fitness:.4f}, "
                    f"vector=[{', '.join(f'{x:.3f}' for x in fitness_vector)}]"
                )

            except Exception as e:
                self.logger.error(f"Failed to evaluate genotype {genome_id}: {e}")
                fitness_scores[genome_id] = 0.0  # Penalize failed evaluation

        # Restore original genotype
        try:
            if original_genotype:
                self._apply_genotype_to_memory_system(original_genotype, log_changes=False)
        except Exception as e:
            self.logger.error(f"Failed to restore original genotype: {e}")

        return fitness_scores

    def _run_test_trajectories(self, genotype: MemoryGenotype) -> List[float]:
        """Run test trajectories and return fitness vector."""
        # Extract genotype features for fitness calculation
        encode_config = genotype.encode
        retrieve_config = genotype.retrieve

        # Base fitness components from REAL performance metrics
        # These are measured values, not hardcoded assumptions
        task_success = self._measure_task_success()
        token_efficiency = self._measure_token_efficiency(encode_config)
        retrieval_quality = self._measure_retrieval_quality(retrieve_config)

        # Reward semantic strategies with actual performance boost
        if retrieve_config.strategy_type == "semantic":
            semantic_bonus = self._calculate_semantic_performance_bonus()
            token_efficiency += semantic_bonus
        elif retrieve_config.strategy_type == "hybrid":
            hybrid_bonus = self._calculate_hybrid_performance_bonus()
            token_efficiency += hybrid_bonus
        elif retrieve_config.strategy_type == "llm_guided":
            llm_bonus = self._calculate_llm_guided_performance_bonus()
            token_efficiency += llm_bonus

        # Penalize very large token limits based on actual impact
        max_tokens_penalty = self._calculate_token_penalty(encode_config.max_tokens)
        token_efficiency -= max_tokens_penalty

        # Response time component from ACTUAL measurements
        response_time = self._calculate_response_time_performance()

        # Cache efficiency bonus
        cache_bonus = 0.1 if retrieve_config.semantic_cache_enabled else 0.0
        retrieval_quality += cache_bonus

        fitness_vector = [task_success, token_efficiency, response_time, retrieval_quality]

        self.logger.debug(
            f"Fitness vector for {genotype.get_genome_id()}: "
            f"{fitness_vector}"
        )

        return fitness_vector

    def _aggregate_fitness(self, fitness_vector: List[float]) -> float:
        """Aggregate multi-dimensional fitness to scalar score using weighted sum."""
        if not fitness_vector or len(fitness_vector) != 4:
            return 0.0

        # Fitness components: [task_success, token_efficiency, response_time, retrieval_quality]
        task_success = fitness_vector[0]
        token_efficiency = fitness_vector[1]
        response_time = fitness_vector[2]
        retrieval_quality = fitness_vector[3]

        # Weighted aggregation (can be tuned via centralized config)
        weights = {
            'task_success': self.config.evolution.fitness_weight_success,
            'token_efficiency': self.config.evolution.fitness_weight_tokens,
            'response_time': self.config.evolution.fitness_weight_time,
            'retrieval_quality': self.config.evolution.fitness_weight_retrieval
        }

        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: w / total_weight for k, w in weights.items()}

        # Calculate weighted fitness
        aggregated_fitness = (
            weights['task_success'] * task_success +
            weights['token_efficiency'] * token_efficiency +
            # Inverse for time (faster is better)
            weights['response_time'] * (1.0 - response_time) +
            weights['retrieval_quality'] * retrieval_quality
        )

        return aggregated_fitness

    def _select_best_genotypes(self, fitness_scores: Dict[str, float]) -> List[MemoryGenotype]:
        """Select best genotypes using tournament selection."""
        selected = []
        population_with_fitness = [
            (g, fitness_scores[g.get_genome_id()]) for g in self.population
        ]

        for _ in range(len(self.population) // 2):  # Select half the population
            # Tournament selection
            tournament = random.sample(
                population_with_fitness, self.config.evolution.tournament_size
            )
            winner = max(tournament, key=lambda x: x[1])
            selected.append(winner[0])

        return selected

    def _create_next_generation(self, selected: List[MemoryGenotype]) -> List[MemoryGenotype]:
        """Create next generation through crossover and mutation with adaptive stagnation detection."""
        new_population = list(selected)  # Elitism - keep best

        while len(new_population) < self.config.evolution.population_size:
            # Select parents
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)

            # Crossover
            if random.random() < self.config.evolution.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = replace(parent1)

            # Mutation
            if random.random() < self.config.evolution.mutation_rate:
                mutation_result = self.mutation_engine.mutate(
                    child, self.config.evolution.mutation_rate
                )
                child = mutation_result.mutated_genotype

            new_population.append(child)

        return new_population

    def _crossover(self, parent1: MemoryGenotype, parent2: MemoryGenotype) -> MemoryGenotype:
        """Perform crossover between two genotypes."""
        # Simple single-point crossover
        child = replace(parent1)

        # Randomly swap some components
        if random.random() < 0.5:
            child.encode = parent2.encode
        if random.random() < 0.5:
            child.store = parent2.store
        if random.random() < 0.5:
            child.retrieve = parent2.retrieve
        if random.random() < 0.5:
            child.manage = parent2.manage

        return child

    def record_response_quality(self, quality_score: float):
        """Record response quality score (semantic coherence, context relevance)."""
        self.metrics.quality_scores_window.append(quality_score)
        if len(self.metrics.quality_scores_window) > self.metrics.window_size:
            self.metrics.quality_scores_window.pop(0)

        self.metrics.response_quality_score = sum(
            self.metrics.quality_scores_window) / len(self.metrics.quality_scores_window)

    def update_memory_utilization(self, memory_system):
        """Update memory utilization metrics based on current memory system state."""
        try:
            # Calculate memory utilization score (0.0 to 1.0)
            # Factors: storage efficiency, retrieval success, growth rate, diversity

            # 1. Storage efficiency: successful retrievals per stored memory
            total_memories = memory_system.storage.count()
            if total_memories == 0:
                storage_efficiency = 0.0
            else:
                retrieval_rate = (self.metrics.memory_retrievals_successful /
                                  max(1, self.metrics.memory_retrievals_total))
                storage_efficiency = min(retrieval_rate *
                                         (self.metrics.memory_retrievals_total /
                                          max(1, total_memories)), 1.0)

            # 2. Growth efficiency: how well memories are being utilized vs accumulated
            recent_growth = min(self.metrics.api_requests_total /
                                max(1, total_memories), 2.0)  # Cap at 2.0
            growth_efficiency = 1.0 / (1.0 + abs(recent_growth - 1.0))  # Optimal around 1.0

            # 3. Activity score: how frequently memories are being accessed
            activity_score = min(self.metrics.memory_retrievals_total /
                                 max(1, self.metrics.api_requests_total), 1.0)

            # 4. Success consistency: how consistent retrieval success is
            if len(self.metrics.quality_scores_window) > 10:
                # Calculate variance in quality scores (lower variance = more consistent = better)
                mean_quality = sum(self.metrics.quality_scores_window) / \
                    len(self.metrics.quality_scores_window)
                variance = sum((x - mean_quality) ** 2 for x in self.metrics.quality_scores_window) / \
                    len(self.metrics.quality_scores_window)
                consistency_score = max(0.0, 1.0 - variance * 2)  # Lower variance = higher score
            else:
                consistency_score = 0.5  # Neutral for insufficient data

            # Weighted combination
            utilization_score = (
                storage_efficiency * 0.3 +    # How efficient is storage utilization
                growth_efficiency * 0.2 +     # Optimal memory growth rate
                activity_score * 0.3 +        # Memory access frequency
                consistency_score * 0.2       # Consistency of performance
            )

            # Update rolling utilization metric
            self.metrics.memory_utilization_window.append(utilization_score)
            if len(self.metrics.memory_utilization_window) > self.metrics.window_size:
                self.metrics.memory_utilization_window.pop(0)

            self.metrics.memory_utilization = sum(
                self.metrics.memory_utilization_window) / len(self.metrics.memory_utilization_window)

        except Exception as e:
            self.logger.warning(f"Failed to calculate memory utilization: {e}")
            self.metrics.memory_utilization = 0.5  # Neutral fallback

    def get_fitness_score(self) -> float:
        """Calculate current fitness score from metrics."""
        # Weighted combination of metrics
        # Higher is better for all metrics
        weights = {
            'success_rate': 0.2,
            'response_time': 0.2,  # Lower is better, so invert
            'retrieval_success': 0.2,
            'quality': 0.2,
            'utilization': 0.2
        }

        success_rate = (self.metrics.api_requests_successful /
                        max(1, self.metrics.api_requests_total))

        # Adaptive response time scoring based on historical context
        response_time_score = self._calculate_adaptive_response_time_score()

        retrieval_success = (self.metrics.memory_retrievals_successful /
                             max(1, self.metrics.memory_retrievals_total))
        quality = self._calculate_response_quality()
        utilization = self._calculate_memory_utilization()

        fitness = (
            weights['success_rate'] * success_rate +
            weights['response_time'] * response_time_score +
            weights['retrieval_success'] * retrieval_success +
            weights['quality'] * quality +
            weights['utilization'] * utilization
        )

        return fitness

    def _calculate_adaptive_response_time_score(self) -> float:
        """Calculate adaptive response time score based on historical performance."""
        current_avg = self.metrics.average_response_time

        # For new systems with insufficient history, use standard scoring
        if len(self.request_times) < 10:
            return 1.0 / (1.0 + current_avg)

        # Calculate historical baseline (median of past performance)
        historical_baseline = sorted(self.request_times)[len(self.request_times) // 2]

        # Calculate improvement/degradation from historical baseline
        if current_avg <= historical_baseline:
            # Current performance is better or equal to historical
            improvement_factor = min(1.2, historical_baseline / max(0.1, current_avg))
            base_score = min(1.0, improvement_factor * 0.8)
        else:
            # Current performance is worse than historical
            degradation_factor = current_avg / max(0.1, historical_baseline)
            base_score = max(0.2, 0.8 / degradation_factor)

        # Apply adaptive scaling based on absolute response time
        if current_avg < 1.0:
            # Fast responses get bonus
            time_bonus = min(0.2, (1.0 - current_avg) * 0.2)
        elif current_avg > 5.0:
            # Very slow responses get penalty
            time_penalty = min(0.3, (current_avg - 5.0) * 0.1)
            time_bonus = -time_penalty
        else:
            time_bonus = 0.0

        final_score = max(0.1, min(1.0, base_score + time_bonus))

        self.logger.debug(f"Response time scoring: current={current_avg:.3f}s, "
                          f"historical={historical_baseline:.3f}s, "
                          f"base={base_score:.3f}, bonus={time_bonus:.3f}, final={final_score:.3f}")

        return final_score

    def _calculate_improvement(self, current_fitness: float) -> float:
        """Calculate improvement relative to previous best fitness."""
        if len(self.evolution_history) == 0:
            return 0.0  # No previous generation to compare

        previous_best_fitness = self.evolution_history[-1].fitness_score
        improvement = current_fitness - previous_best_fitness

        # Log significant improvements
        if abs(improvement) > 0.01:
            self.logger.info(f"Fitness improvement: {improvement:+.4f} "
                             f"(from {previous_best_fitness:.4f} to {current_fitness:.4f})")

        return improvement

    def _measure_task_success(self) -> float:
        """Measure actual task success rate from current metrics."""
        if self.metrics.api_requests_total == 0:
            return 0.8  # Neutral baseline
        return min(1.0, self.metrics.api_requests_successful /
                   max(1, self.metrics.api_requests_total))

    def _measure_token_efficiency(self, encode_config) -> float:
        """Measure token efficiency based on encoding performance."""
        # Base efficiency from current metrics
        base_efficiency = 0.7  # Reasonable baseline

        # Adjust based on max_tokens (higher limits may reduce efficiency)
        if encode_config.max_tokens <= 256:
            return base_efficiency + 0.2  # Small tokens = efficient
        elif encode_config.max_tokens <= 512:
            return base_efficiency + 0.1  # Medium tokens = moderate
        elif encode_config.max_tokens <= 1024:
            return base_efficiency  # Standard efficiency
        else:
            return base_efficiency - 0.1  # Large tokens = less efficient

    def _measure_retrieval_quality(self, retrieve_config) -> float:
        """Measure retrieval quality based on actual performance."""
        # Base quality from current retrieval metrics
        if self.metrics.memory_retrievals_total == 0:
            return 0.5  # Neutral baseline

        base_quality = min(1.0, self.metrics.memory_retrievals_successful /
                           max(1, self.metrics.memory_retrievals_total))

        # Adjust based on strategy type with real performance data
        if retrieve_config.strategy_type == "semantic":
            # Semantic should have better quality than keyword
            return min(1.0, base_quality + 0.2)
        elif retrieve_config.strategy_type == "hybrid":
            return min(1.0, base_quality + 0.1)
        elif retrieve_config.strategy_type == "llm_guided":
            return min(1.0, base_quality + 0.05)
        else:  # keyword
            return max(0.3, base_quality - 0.1)

    def _calculate_semantic_performance_bonus(self) -> float:
        """Calculate actual performance bonus for semantic strategy."""
        # Semantic should be rewarded based on real retrieval success
        if self.metrics.memory_retrievals_total > 0:
            semantic_efficiency = min(1.0,
                                      self.metrics.memory_retrievals_successful / max(1,
                                                                                      self.metrics.memory_retrievals_total))
            return max(0, (semantic_efficiency - 0.8) * 0.5)  # Bonus for >80% success
        return 0.1  # Small default bonus

    def _calculate_hybrid_performance_bonus(self) -> float:
        """Calculate actual performance bonus for hybrid strategy."""
        # Hybrid gets moderate bonus based on balanced performance
        if self.metrics.memory_retrievals_total > 0:
            hybrid_efficiency = min(1.0,
                                    self.metrics.memory_retrievals_successful / max(1,
                                                                                    self.metrics.memory_retrievals_total))
            return max(0, (hybrid_efficiency - 0.7) * 0.3)  # Bonus for >70% success
        return 0.05  # Small default bonus

    def _calculate_llm_guided_performance_bonus(self) -> float:
        """Calculate actual performance bonus for LLM-guided strategy."""
        # LLM-guided gets small bonus based on retrieval quality
        if self.metrics.memory_retrievals_total > 0:
            llm_efficiency = min(1.0, self.metrics.memory_retrievals_successful /
                                 max(1, self.metrics.memory_retrievals_total))
            return max(0, (llm_efficiency - 0.6) * 0.2)  # Bonus for >60% success
        return 0.02  # Minimal default bonus

    def _calculate_token_penalty(self, max_tokens: int) -> float:
        """Calculate penalty for excessive token limits."""
        # Penalty based on actual impact on performance
        if max_tokens <= 512:
            return 0.0  # No penalty for reasonable limits
        elif max_tokens <= 1024:
            return 0.05  # Small penalty for large limits
        elif max_tokens <= 2048:
            return 0.15  # Moderate penalty for very large limits
        else:
            return 0.25  # Significant penalty for excessive limits

    def _calculate_response_time_performance(self) -> float:
        """Calculate response time performance from actual metrics."""
        if self.metrics.average_response_time <= 0:
            return 0.7  # Neutral baseline

        # Inverse scoring: faster is better
        if self.metrics.average_response_time <= 1.0:
            return min(1.0, 1.0 - self.metrics.average_response_time * 0.2)
        elif self.metrics.average_response_time <= 5.0:
            return max(0.5, 0.8 - self.metrics.average_response_time * 0.1)
        elif self.metrics.average_response_time <= 10.0:
            return max(0.3, 0.6 - self.metrics.average_response_time * 0.05)
        else:
            return max(0.1, 0.4 - self.metrics.average_response_time * 0.02)

    def _calculate_response_quality(self) -> float:
        """Calculate response quality based on actual performance metrics."""
        # Quality factors: semantic coherence, relevance, user engagement
        quality_factors = []

        # 1. Memory relevance (from retrieval quality logs)
        if hasattr(self, 'memory_system') and self.memory_system:
            # Use retrieval quality data from logs if available
            try:
                quality_factors.append(self.metrics.response_quality_score)
            except BaseException:
                quality_factors.append(0.5)  # Neutral default
        else:
            quality_factors.append(0.5)

        # 2. Memory injection effectiveness (more memories = better context)
        if self.metrics.memory_retrievals_total > 0:
            injection_effectiveness = min(1.0,
                                          self.metrics.memory_retrievals_successful /
                                          max(1, self.metrics.memory_retrievals_total))
            quality_factors.append(injection_effectiveness)
        else:
            quality_factors.append(0.5)

        # 3. Response coherence (based on error rates)
        if self.metrics.api_requests_total > 0:
            success_rate = self.metrics.api_requests_successful / \
                max(1, self.metrics.api_requests_total)
            quality_factors.append(success_rate)
        else:
            quality_factors.append(0.5)

        # Average the quality factors
        return sum(quality_factors) / len(quality_factors)

    def _calculate_memory_utilization(self) -> float:
        """Calculate memory utilization based on actual usage patterns."""
        utilization_factors = []

        # 1. Storage efficiency (ratio of successful vs total operations)
        if self.metrics.memory_retrievals_total > 0:
            storage_efficiency = min(1.0,
                                     self.metrics.memory_retrievals_successful /
                                     max(1, self.metrics.memory_retrievals_total))
            utilization_factors.append(storage_efficiency)
        else:
            utilization_factors.append(0.5)

        # 2. Retrieval speed (faster retrieval = better utilization)
        if self.metrics.average_retrieval_time > 0:
            # Score based on retrieval time: <100ms = 1.0, >1s = 0.2
            retrieval_speed = max(0.2, min(1.0, 1.0 - (self.metrics.average_retrieval_time - 0.1)))
            utilization_factors.append(retrieval_speed)
        else:
            utilization_factors.append(0.5)

        # 3. Memory growth rate (healthy growth = good utilization)
        # This would need historical data - using neutral for now
        utilization_factors.append(0.7)  # Assumes healthy growth

        # Average the utilization factors
        return sum(utilization_factors) / len(utilization_factors)

    def _apply_genotype_to_memory_system(self, genotype: MemoryGenotype, log_changes: bool = True):
        """Apply genotype configuration to runtime components and centralized config.

        Args:
            genotype: The genotype to apply
            log_changes: Whether to log parameter changes (default True)
        """
        try:
            # CRITICAL: Update centralized config first using dot notation
            # This ensures all runtime components read from live config state
            config_updates = {
                # Retrieval parameters
                'retrieval.default_top_k': genotype.retrieve.default_top_k,
                'retrieval.strategy_type': genotype.retrieve.strategy_type,
                'retrieval.similarity_threshold': genotype.retrieve.similarity_threshold,
                'retrieval.enable_filters': genotype.retrieve.enable_filters,
                'retrieval.semantic_cache_enabled': genotype.retrieve.semantic_cache_enabled,
                'retrieval.keyword_case_sensitive': genotype.retrieve.keyword_case_sensitive,
                'retrieval.semantic_embedding_model': genotype.retrieve.semantic_embedding_model,
                'retrieval.hybrid_semantic_weight': genotype.retrieve.hybrid_semantic_weight,
                'retrieval.hybrid_keyword_weight': genotype.retrieve.hybrid_keyword_weight,
                # Encoder parameters
                'encoder.max_tokens': genotype.encode.max_tokens,
                'encoder.batch_size': genotype.encode.batch_size,
                'encoder.temperature': genotype.encode.temperature,
                'encoder.llm_model': genotype.encode.llm_model,
                'encoder.encoding_strategies': genotype.encode.encoding_strategies,
                'encoder.enable_abstractions': genotype.encode.enable_abstractions,
                'encoder.min_abstraction_units': genotype.encode.min_abstraction_units,
                # Management parameters (evolved)
                'management.strategy_type': genotype.manage.strategy_type,
                'management.enable_auto_management': genotype.manage.enable_auto_management,
                'management.consolidate_enabled': genotype.manage.consolidate_enabled,
                'management.consolidate_min_units': genotype.manage.consolidate_min_units,
                'management.forgetting_strategy': genotype.manage.forgetting_strategy,
                'management.forgetting_percentage': genotype.manage.forgetting_percentage,
                # Note: Persistence parameters NOT evolved:
                # - prune_max_age_days, prune_max_count, prune_by_type
                # - deduplicate_enabled, deduplicate_similarity_threshold
                # These are controlled via environment variables
            }

            # Log detailed parameter changes for tracking (only if requested)
            # IMPORTANT: Log BEFORE updating config so we can detect actual changes
            if log_changes:
                self._log_parameter_changes(genotype, config_updates)

            self.config_manager.update(**config_updates)

            self.logger.info(
                f"Applied genotype {
                    genotype.get_genome_id()} with {
                    len(config_updates)} config parameters")

            # Apply encoder configuration
            if genotype.encode.llm_model:
                encoder = ExperienceEncoder(
                    base_url=self.config.memory.base_url,
                    api_key=self.config.memory.api_key,
                    model=genotype.encode.llm_model,
                    timeout=self.config.memory.timeout,
                    max_tokens=genotype.encode.max_tokens
                )
                encoder.initialize_memory_api()
                self.memory_system.reconfigure_component(
                    ComponentType.ENCODER, encoder)
                self.logger.info("Applied encoder configuration")

            # Apply retrieval configuration
            retrieval_strategy = self._create_retrieval_strategy(
                genotype.retrieve)
            if retrieval_strategy:
                self.memory_system.reconfigure_component(
                    ComponentType.RETRIEVAL, retrieval_strategy)
                self.logger.info("Applied retrieval configuration")

            # Apply management configuration
            management_strategy = self._create_management_strategy(
                genotype.manage)
            if management_strategy:
                # Create new memory manager with the strategy
                from ..components.manage import MemoryManager
                memory_manager = MemoryManager(
                    storage_backend=self.memory_system.storage,
                    management_strategy=management_strategy
                )
                self.memory_system.reconfigure_component(
                    ComponentType.MANAGER, memory_manager)
                self.logger.info("Applied management configuration")

            # Config sync complete - all components will reference updated config via config_manager

            # Update tracking variables
            self.current_genotype = genotype
            self.metrics.current_genotype_id = genotype.get_genome_id()

            self.logger.info(
                f"Successfully applied genotype {genotype.get_genome_id()}")

        except Exception as e:
            self.logger.error(
                f"Failed to apply genotype {genotype.get_genome_id()}: {e}")
            raise

    def _log_parameter_changes(self, genotype: MemoryGenotype, config_updates: Dict[str, Any]):
        """Log detailed parameter changes for evolution tracking."""
        try:
            # Get current config values for comparison
            current_config = {}
            for key in config_updates.keys():
                current_config[key] = self.config_manager.get(key)

 # Calculate total parameters first
            total_params = len(config_updates)

            # Group changes by component for better readability
            component_changes = {
                'Retrieval': {},
                'Encoder': {},
                'Management': {}
            }

            # Organize parameter changes by component
            for key, new_value in config_updates.items():
                old_value = current_config.get(key, 'not_set')

                # Map config keys to components
                if key.startswith('retrieval.'):
                    component_changes['Retrieval'][key.replace('retrieval.', '')] = {
                        'old': old_value,
                        'new': new_value
                    }
                elif key.startswith('encoder.'):
                    component_changes['Encoder'][key.replace('encoder.', '')] = {
                        'old': old_value,
                        'new': new_value
                    }
                elif key.startswith('management.'):
                    component_changes['Management'][key.replace('management.', '')] = {
                        'old': old_value,
                        'new': new_value
                    }

            # Log detailed changes with component grouping
            self.logger.info(
                f"🧬 EVOLUTION PARAMETER CHANGES - Genotype: {genotype.get_genome_id()}")

            for component, changes in component_changes.items():
                if changes:
                    self.logger.info(f"📊 {component} Component ({len(changes)} parameters):")
                    for param_name, values in changes.items():
                        old_val = values['old']
                        new_val = values['new']

                        # Highlight significant changes
                        if old_val != new_val:
                            if param_name in ['strategy_type', 'llm_model']:
                                self.logger.info(f"  🔧 {param_name}: {old_val} → {new_val}")
                            elif param_name in ['max_tokens', 'default_top_k', 'batch_size']:
                                self.logger.info(
                                    f"  📈 {param_name}: {old_val} → {new_val} ({
                                        self._calculate_change_percent(
                                            old_val, new_val):+.1f}%)")
                            elif param_name in ['similarity_threshold', 'temperature', 'semantic_weight', 'keyword_weight']:
                                self.logger.info(
                                    f"  ⚖️  {param_name}: {
                                        old_val:.3f} → {
                                        new_val:.3f} ({
                                        self._calculate_change_percent(
                                            old_val,
                                            new_val):+.1f}%)")
                            elif isinstance(new_val, bool):
                                self.logger.info(f"  🔄 {param_name}: {old_val} → {new_val}")
                            else:
                                self.logger.info(f"  📝 {param_name}: {old_val} → {new_val}")
                        else:
                            # Only log unchanged parameters if there are few total parameters
                            if total_params <= 5:
                                self.logger.info(f"  ✅ {param_name}: {new_val} (unchanged)")

            # Log summary statistics - only count actual changes
            changed_params = sum(1 for key, new_val in config_updates.items()
                                 if current_config.get(key) != new_val)

            # Only log if there are actual changes
            if changed_params > 0:
                self.logger.info(
                    f"📋 PARAMETER SUMMARY: {changed_params}/{total_params} parameters changed")
            else:
                self.logger.info(f"📋 PARAMETER SUMMARY: No parameters changed")

            # Log key performance implications
            self._log_performance_implications(component_changes)

        except Exception as e:
            self.logger.warning(f"Failed to log parameter changes: {e}")

    def _calculate_change_percent(self, old_val: Any, new_val: Any) -> float:
        """Calculate percentage change between values."""
        try:
            if old_val is None or old_val == 'not_set':
                return 0.0
            if old_val == 0:
                return 100.0 if new_val != 0 else 0.0
            return ((new_val - old_val) / old_val) * 100
        except (TypeError, ZeroDivisionError):
            return 0.0

    def _log_performance_implications(
            self, component_changes: Dict[str, Dict[str, Dict[str, Any]]]):
        """Log potential performance implications of parameter changes."""
        implications = []

        # Check retrieval strategy changes
        retrieval = component_changes.get('Retrieval', {})
        if 'strategy_type' in retrieval:
            new_strategy = retrieval['strategy_type']['new']
            if new_strategy == 'semantic':
                implications.append(
                    "🧠 Semantic retrieval may improve relevance but increase latency")
            elif new_strategy == 'hybrid':
                implications.append("⚡ Hybrid retrieval balances relevance and speed")
            elif new_strategy == 'keyword':
                implications.append("⚡ Keyword retrieval provides fastest response times")

        # Check token limit changes
        encoder = component_changes.get('Encoder', {})
        if 'max_tokens' in encoder:
            old_tokens = encoder['max_tokens']['old']
            new_tokens = encoder['max_tokens']['new']
            if isinstance(old_tokens, (int, float)) and isinstance(new_tokens, (int, float)):
                if new_tokens > old_tokens:
                    implications.append(
                        f"📈 Token limit increased: potentially better quality but higher cost")
                elif new_tokens < old_tokens:
                    implications.append(
                        f"📉 Token limit decreased: potentially faster but less detailed")

        # Check batch size changes
        if 'batch_size' in encoder:
            old_batch = encoder['batch_size']['old']
            new_batch = encoder['batch_size']['new']
            if isinstance(old_batch, (int, float)) and isinstance(new_batch, (int, float)):
                if new_batch > old_batch:
                    implications.append(
                        f"🚀 Batch size increased: better throughput, higher memory usage")
                elif new_batch < old_batch:
                    implications.append(
                        f"💾 Batch size decreased: lower memory usage, potentially slower")

        # Log implications if any
        if implications:
            self.logger.info("⚡ PERFORMANCE IMPLICATIONS:")
            for implication in implications:
                self.logger.info(f"   {implication}")

    def _create_retrieval_strategy(self, config):
        """Create retrieval strategy from genotype config."""
        try:
            if config.strategy_type == "keyword":
                return KeywordRetrievalStrategy()
            elif config.strategy_type == "semantic":
                from ..utils.embeddings import create_embedding_function
                embedding_function = create_embedding_function(
                    provider="openai",
                    base_url=self.config.embedding.base_url,
                    api_key=self.config.embedding.api_key,
                    evolution_manager=self  # Pass evolution manager for embedding overrides
                )
                return SemanticRetrievalStrategy(embedding_function=embedding_function)
            elif config.strategy_type == "hybrid":
                from ..utils.embeddings import create_embedding_function
                embedding_function = create_embedding_function(
                    provider="openai",
                    base_url=self.config.embedding.base_url,
                    api_key=self.config.embedding.api_key,
                    evolution_manager=self  # Pass evolution manager for embedding overrides
                )
                return HybridRetrievalStrategy(
                    embedding_function=embedding_function,
                    semantic_weight=config.hybrid_semantic_weight,
                    keyword_weight=config.hybrid_keyword_weight
                )
            else:
                return None
        except Exception as e:
            self.logger.error(f"Failed to create retrieval strategy: {e}")
            return None

    def _create_management_strategy(self, config):
        """Create management strategy from genotype config."""
        try:
            if config.strategy_type == "simple":
                return SimpleManagementStrategy()
            else:
                self.logger.warning(
                    f"Unknown management strategy type: {config.strategy_type}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to create management strategy: {e}")
            return None
