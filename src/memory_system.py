from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

from components.encode import ExperienceEncoder
from components.store import StorageBackend
from components.retrieve import RetrievalStrategy, RetrievalContext
from components.manage import ManagementStrategy, MemoryManager, HealthMetrics


@dataclass
class MemorySystemConfig:
    """Configuration for MemorySystem."""

    llm_base_url: str = "http://192.168.1.61:11434/v1"
    llm_api_key: str = "dummy-key"
    llm_model: Optional[str] = None
    storage_backend: Optional[StorageBackend] = None
    retrieval_strategy: Optional[RetrievalStrategy] = None
    management_strategy: Optional[ManagementStrategy] = None
    default_retrieval_top_k: int = 5
    enable_auto_management: bool = True
    auto_prune_threshold: int = 1000
    log_level: str = "INFO"
    on_encode_complete: Optional[Callable] = None
    on_retrieve_complete: Optional[Callable] = None
    on_manage_complete: Optional[Callable] = None


class MemorySystem:
    """Main memory system integrating all four components."""

    def __init__(self, config: Optional[MemorySystemConfig] = None):
        self.config = config or MemorySystemConfig()
        self._setup_logging()

        self.encoder: Optional[ExperienceEncoder] = None
        self.storage: Optional[StorageBackend] = None
        self.retrieval_context: Optional[RetrievalContext] = None
        self.memory_manager: Optional[MemoryManager] = None

        self.operation_log: List[Dict[str, Any]] = []
        self._initialize_components()

    def _setup_logging(self):
        """Configure logging based on config."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("MemorySystem")

    def _initialize_components(self):
        """Initialize all components based on configuration."""
        try:
            self._initialize_encoder()
            self._initialize_storage()
            self._initialize_retrieval()
            self._initialize_management()
            self.logger.info("All components initialized successfully")
        except Exception as e:
            self.logger.error(f"Component initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize MemorySystem: {str(e)}")

    def _initialize_encoder(self):
        """Initialize encoder component."""
        if self.encoder is None:
            self.encoder = ExperienceEncoder(
                base_url=self.config.llm_base_url,
                api_key=self.config.llm_api_key,
                model=self.config.llm_model
            )
            self.encoder.initialize_llm()
            self.logger.info("Encoder initialized")

    def _initialize_storage(self):
        """Initialize the storage backend."""
        if self.config.storage_backend:
            self.storage = self.config.storage_backend
            self.logger.info("Storage backend configured")

    def _initialize_retrieval(self):
        """Initialize the retrieval context."""
        if self.config.retrieval_strategy:
            self.retrieval_context = RetrievalContext(
                strategy=self.config.retrieval_strategy,
                default_top_k=self.config.default_retrieval_top_k
            )
            self.logger.info("Retrieval strategy configured")

    def _initialize_management(self):
        """Initialize the memory manager."""
        if self.storage and self.config.management_strategy:
            self.memory_manager = MemoryManager(
                storage_backend=self.storage,
                management_strategy=self.config.management_strategy
            )
            self.logger.info("Memory manager configured")

    def add_experience(self, experience: Dict[str, Any]) -> str:
        """Add a single experience to memory.

        Returns:
            The ID of the stored memory unit
        """
        try:
            if not self.encoder or not self.storage:
                raise RuntimeError(
                    "Encoder and storage must be initialized"
                )

            encoded_unit = self.encoder.encode_experience(experience)
            unit_id = self.storage.store(encoded_unit)

            self._log_operation(
                "add_experience",
                {"experience_id": experience.get("id"), "unit_id": unit_id}
            )

            if self.config.on_encode_complete:
                self.config.on_encode_complete(unit_id, encoded_unit)

            self._auto_manage()

            return unit_id
        except Exception as e:
            self.logger.error(f"Failed to add experience: {str(e)}")
            raise RuntimeError(f"Add experience failed: {str(e)}")

    def add_trajectory(
        self,
        trajectory: List[Dict[str, Any]]
    ) -> List[str]:
        """Add a trajectory of experiences to memory.

        Returns:
            List of stored unit IDs
        """
        try:
            if not self.encoder or not self.storage:
                raise RuntimeError(
                    "Encoder and storage must be initialized"
                )

            encoded_units = self.encoder.encode_trajectory(trajectory)
            unit_ids = self.storage.store_batch(encoded_units)

            self._log_operation(
                "add_trajectory",
                {
                    "trajectory_length": len(trajectory),
                    "units_stored": len(unit_ids)
                }
            )

            if self.config.on_encode_complete:
                for unit_id, unit in zip(unit_ids, encoded_units):
                    self.config.on_encode_complete(unit_id, unit)

            self._auto_manage()

            return unit_ids
        except Exception as e:
            self.logger.error(f"Failed to add trajectory: {str(e)}")
            raise RuntimeError(f"Add trajectory failed: {str(e)}")

    def query_memory(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query memory for relevant information.

        Returns:
            List of retrieved memory units
        """
        try:
            if not self.retrieval_context or not self.storage:
                raise RuntimeError(
                    "Retrieval context and storage must be initialized"
                )

            results = self.retrieval_context.retrieve(
                query=query,
                storage_backend=self.storage,
                top_k=top_k,
                filters=filters
            )

            self._log_operation(
                "query_memory",
                {
                    "query": query,
                    "top_k": top_k,
                    "results_count": len(results)
                }
            )

            if self.config.on_retrieve_complete:
                self.config.on_retrieve_complete(query, results)

            return [r.unit for r in results]
        except Exception as e:
            self.logger.error(f"Failed to query memory: {str(e)}")
            raise RuntimeError(f"Query memory failed: {str(e)}")

    def retrieve_by_ids(
        self,
        unit_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Retrieve specific memory units by their IDs.

        Returns:
            List of retrieved memory units
        """
        try:
            if not self.retrieval_context or not self.storage:
                raise RuntimeError(
                    "Retrieval context and storage must be initialized"
                )

            results = self.retrieval_context.retrieve_by_ids(
                unit_ids=unit_ids,
                storage_backend=self.storage
            )

            return [r.unit for r in results]
        except Exception as e:
            self.logger.error(f"Failed to retrieve by IDs: {str(e)}")
            raise RuntimeError(f"Retrieve by IDs failed: {str(e)}")

    def generate_abstraction(
        self,
        unit_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate a high-level abstraction from memory units.

        Returns:
            Generated abstraction
        """
        try:
            if not self.encoder:
                raise RuntimeError("Encoder must be initialized")

            if unit_ids:
                units = self.retrieve_by_ids(unit_ids)
            else:
                units = self.storage.retrieve_all() if self.storage else []

            if not units:
                raise RuntimeError("No units available for abstraction")

            abstraction = self.encoder.generate_abstraction(units)

            self._log_operation(
                "generate_abstraction",
                {"units_used": len(units)}
            )

            return abstraction
        except Exception as e:
            self.logger.error(f"Failed to generate abstraction: {str(e)}")
            raise RuntimeError(f"Generate abstraction failed: {str(e)}")

    def manage_memory(
        self,
        operation: str,
        **kwargs
    ) -> Any:
        """Execute memory management operations.

        Args:
            operation: "prune", "consolidate", "deduplicate", "forget"
            **kwargs: Operation-specific arguments

        Returns:
            Operation result
        """
        try:
            if not self.memory_manager:
                raise RuntimeError("Memory manager must be initialized")

            result = None

            if operation == "prune":
                result = self.memory_manager.prune(
                    criteria=kwargs.get("criteria"))
            elif operation == "consolidate":
                result = self.memory_manager.consolidate(
                    units=kwargs.get("units")
                )
            elif operation == "deduplicate":
                result = self.memory_manager.deduplicate(
                    similarity_threshold=kwargs.get(
                        "similarity_threshold", 0.9)
                )
            elif operation == "forget":
                result = self.memory_manager.apply_forgetting(
                    strategy=kwargs.get("strategy", "lru"),
                    count=kwargs.get("count")
                )
            else:
                raise ValueError(f"Unknown operation: {operation}")

            self._log_operation(
                f"manage_{operation}",
                kwargs
            )

            if self.config.on_manage_complete:
                self.config.on_manage_complete(operation, result)

            return result
        except Exception as e:
            self.logger.error(f"Failed to manage memory: {str(e)}")
            raise RuntimeError(f"Manage memory failed: {str(e)}")

    def get_health_metrics(self) -> Optional[HealthMetrics]:
        """Get current health metrics for the memory system.

        Returns:
            Health metrics or None if manager not initialized
        """
        try:
            if not self.memory_manager:
                return None

            return self.memory_manager.get_health_metrics()
        except Exception as e:
            self.logger.error(f"Failed to get health metrics: {str(e)}")
            return None

    def get_operation_log(self) -> List[Dict[str, Any]]:
        """Get the operation log.

        Returns:
            List of operation logs
        """
        return self.operation_log.copy()

    def clear_operation_log(self):
        """Clear the operation log."""
        self.operation_log.clear()
        self.logger.info("Operation log cleared")

    def _auto_manage(self):
        """Perform automatic memory management if enabled."""
        if not self.config.enable_auto_management:
            return

        if not self.memory_manager:
            return

        try:
            current_count = self.storage.count() if self.storage else 0
            if current_count > self.config.auto_prune_threshold:
                self.logger.info(
                    f"Auto-management: count {current_count} "
                    f"exceeds threshold {self.config.auto_prune_threshold}"
                )
                target_count = max(1, current_count - 100)
                self.manage_memory("prune", criteria={"max_count": target_count})
        except Exception as e:
            self.logger.warning(f"Auto-management failed: {str(e)}")

    def _log_operation(self, operation: str, details: Dict[str, Any]):
        """Log an operation to the operation log."""
        self.operation_log.append({
            "operation": operation,
            "details": details,
            "timestamp": self._get_timestamp()
        })

    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.now(timezone.utc).isoformat() + "Z"
