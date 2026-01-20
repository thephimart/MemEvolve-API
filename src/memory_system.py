from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import os

from components.encode import ExperienceEncoder
from components.store import StorageBackend
from components.retrieve import RetrievalStrategy, RetrievalContext
from components.manage import ManagementStrategy, MemoryManager, HealthMetrics
from utils.config import MemEvolveConfig


@dataclass
class MemorySystemConfig:
    """Configuration for MemorySystem.

    This class holds all configuration options for a MemEvolve memory system,
    including LLM settings, storage preferences, and behavioral parameters.

    Example:
        >>> config = MemorySystemConfig(
        ...     llm_base_url="http://localhost:8080/v1",
        ...     llm_api_key="your-api-key",
        ...     default_retrieval_top_k=10,
        ...     enable_auto_management=True
        ... )
    """

    llm_base_url: str = field(
        default_factory=lambda: os.getenv("MEMEVOLVE_LLM_BASE_URL", ""),
        metadata={"help": "Base URL for LLM API (e.g., OpenAI, vLLM)"}
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("MEMEVOLVE_LLM_API_KEY", ""),
        metadata={"help": "API key for LLM authentication"}
    )
    llm_model: Optional[str] = field(
        default_factory=lambda: os.getenv("MEMEVOLVE_LLM_MODEL") or None,
        metadata={
            "help": "LLM model name (optional, may be inferred from API)"}
    )
    llm_timeout: int = field(
        default_factory=lambda: int(os.getenv("MEMEVOLVE_LLM_TIMEOUT", "600")),
        metadata={"help": "Timeout for LLM requests in seconds"}
    )
    storage_backend: Optional[StorageBackend] = field(
        default=None,
        metadata={"help": "Storage backend for persisting memories"}
    )
    retrieval_strategy: Optional[RetrievalStrategy] = field(
        default=None,
        metadata={"help": "Strategy for retrieving relevant memories"}
    )
    management_strategy: Optional[ManagementStrategy] = field(
        default=None,
        metadata={"help": "Strategy for managing memory lifecycle"}
    )
    default_retrieval_top_k: int = field(
        default=5,
        metadata={"help": "Default number of memories to retrieve"}
    )
    enable_auto_management: bool = field(
        default=True,
        metadata={"help": "Whether to automatically manage memory lifecycle"}
    )
    auto_prune_threshold: int = field(
        default=1000,
        metadata={"help": "Memory count threshold for auto-pruning"}
    )
    log_level: str = field(
        default="INFO",
        metadata={"help": "Logging level (DEBUG, INFO, WARNING, ERROR)"}
    )
    on_encode_complete: Optional[Callable] = field(
        default=None,
        metadata={"help": "Callback when encoding completes"}
    )
    on_retrieve_complete: Optional[Callable] = field(
        default=None,
        metadata={"help": "Callback when retrieval completes"}
    )
    on_manage_complete: Optional[Callable] = field(
        default=None,
        metadata={"help": "Callback when management completes"}
    )


class MemorySystem:
    """Main memory system integrating encode, store, retrieve, and manage components.

    MemEvolve's MemorySystem provides a unified interface for memory operations,
    automatically handling the four core components: encoding experiences into
    structured memories, storing them efficiently, retrieving relevant memories,
    and managing memory lifecycle.

    Key Features:
    - Automatic component initialization with sensible defaults
    - Flexible configuration through MemorySystemConfig or MemEvolveConfig
    - Batch processing capabilities for improved performance
    - Callback system for monitoring operations
    - Comprehensive logging and error handling

    Basic Usage:
        >>> from memevole import MemorySystem, MemorySystemConfig
        >>>
        >>> # Configure with LLM settings
        >>> config = MemorySystemConfig(
        ...     llm_base_url="http://localhost:8080/v1",
        ...     llm_api_key="your-api-key"
        ... )
        >>>
        >>> # Create memory system
        >>> memory = MemorySystem(config)
        >>>
        >>> # Add an experience
        >>> experience = {
        ...     "action": "debug code",
        ...     "result": "found null pointer exception",
        ...     "timestamp": "2024-01-01T10:00:00Z"
        ... }
        >>> memory.add_experience(experience)
        >>>
        >>> # Query memories
        >>> results = memory.query_memory("debugging techniques", top_k=3)
        >>> print(f"Found {len(results)} relevant memories")

    Advanced Usage:
        >>> # Use custom storage backend
        >>> from components.store import GraphStorageBackend
        >>> config.storage_backend = GraphStorageBackend()
        >>> memory = MemorySystem(config)
        >>>
        >>> # Batch processing for better performance
        >>> experiences = [exp1, exp2, exp3]
        >>> memory.add_trajectory_batch(experiences)
        >>>
        >>> # Custom retrieval with LLM guidance
        >>> from components.retrieve import LLMGuidedRetrievalStrategy
        >>> llm_func = lambda prompt: "your-llm-response"
        >>> config.retrieval_strategy = LLMGuidedRetrievalStrategy(llm_func)
    """

    def __init__(
        self,
        config: Optional[Union[MemorySystemConfig, MemEvolveConfig]] = None,
        encoder: Optional[Any] = None
    ):
        if isinstance(config, MemEvolveConfig):
            # Convert MemEvolveConfig to MemorySystemConfig
            self.config = MemorySystemConfig(
                llm_base_url=config.llm.base_url,
                llm_api_key=config.llm.api_key,
                llm_model=config.llm.model,
                llm_timeout=config.llm.timeout,
                default_retrieval_top_k=config.retrieval.default_top_k,
                enable_auto_management=config.management.enable_auto_management,
                auto_prune_threshold=config.management.auto_prune_threshold,
                log_level=config.logging.level
            )
        else:
            self.config = config or MemorySystemConfig()

        self._provided_encoder = encoder
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
            # Use provided encoder if available, otherwise create default
            if hasattr(self, '_provided_encoder') and self._provided_encoder:
                self.encoder = self._provided_encoder
                self.logger.info("Using provided encoder")
            else:
                self.encoder = ExperienceEncoder(
                    base_url=self.config.llm_base_url,
                    api_key=self.config.llm_api_key,
                    model=self.config.llm_model,
                    timeout=self.config.llm_timeout
                )
                self.encoder.initialize_llm()
                self.logger.info("Encoder initialized")

    def _initialize_storage(self):
        """Initialize the storage backend."""
        if self.config.storage_backend:
            self.storage = self.config.storage_backend
            self.logger.info("Storage backend configured")
        else:
            # Create default storage backend
            from components.store import JSONFileStore
            import tempfile
            import os
            temp_dir = tempfile.gettempdir()
            storage_path = os.path.join(
                temp_dir, f"memory_system_{id(self)}.json")
            self.storage = JSONFileStore(storage_path)
            self.logger.info(
                f"Default storage backend created at {storage_path}")

    def _initialize_retrieval(self):
        """Initialize the retrieval context."""
        if self.config.retrieval_strategy:
            self.retrieval_context = RetrievalContext(
                strategy=self.config.retrieval_strategy,
                default_top_k=self.config.default_retrieval_top_k
            )
            self.logger.info("Retrieval strategy configured")
        else:
            # Skip retrieval initialization for now - will be handled when needed
            self.logger.info(
                "Retrieval initialization skipped - will use on-demand")

    def _initialize_management(self):
        """Initialize the memory manager."""
        if self.storage:
            if self.config.management_strategy:
                self.memory_manager = MemoryManager(
                    storage_backend=self.storage,
                    management_strategy=self.config.management_strategy
                )
                self.logger.info("Memory manager configured")
            else:
                # Create default memory manager
                from components.manage import SimpleManagementStrategy
                strategy = SimpleManagementStrategy()
                self.memory_manager = MemoryManager(
                    storage_backend=self.storage,
                    management_strategy=strategy
                )
                self.logger.info("Default memory manager created")

    def add_experience(self, experience: Dict[str, Any]) -> str:
        """Add a single experience to memory.

        This method processes a raw experience through the encoding pipeline,
        transforms it into a structured memory unit, and stores it for future retrieval.

        Args:
            experience: Dictionary containing experience data. Common fields:
                - "action": What was done (e.g., "searched database")
                - "result": What happened (e.g., "found relevant records")
                - "context": Additional context information
                - "timestamp": When the experience occurred
                - "metadata": Any additional relevant data

        Returns:
            The unique ID of the stored memory unit

        Raises:
            RuntimeError: If encoder or storage components are not initialized

        Example:
            >>> experience = {
            ...     "action": "debug code",
            ...     "result": "found null pointer exception",
            ...     "context": "C++ application",
            ...     "timestamp": "2024-01-01T10:00:00Z"
            ... }
            >>> unit_id = memory.add_experience(experience)
            >>> print(f"Stored memory unit: {unit_id}")
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

    def add_trajectory_batch(
        self,
        trajectory: List[Dict[str, Any]],
        use_parallel: bool = True,
        max_workers: int = 4,
        batch_size: int = 10
    ) -> List[str]:
        """Add a trajectory using optimized batch encoding.

        Args:
            trajectory: List of experience dictionaries
            use_parallel: Whether to use parallel processing
            max_workers: Maximum parallel workers
            batch_size: Size of processing batches

        Returns:
            List of stored unit IDs
        """
        try:
            if not self.encoder or not self.storage:
                raise RuntimeError(
                    "Encoder and storage must be initialized"
                )

            if use_parallel and hasattr(self.encoder, 'encode_trajectory_batch'):
                encoded_units = self.encoder.encode_trajectory_batch(
                    trajectory, max_workers=max_workers, batch_size=batch_size
                )
            else:
                encoded_units = self.encoder.encode_trajectory(trajectory)

            unit_ids = self.storage.store_batch(encoded_units)

            self._log_operation(
                "add_trajectory_batch",
                {
                    "trajectory_length": len(trajectory),
                    "units_stored": len(unit_ids),
                    "parallel_processing": use_parallel,
                    "max_workers": max_workers
                }
            )

            if self.config.on_encode_complete:
                for unit_id, unit in zip(unit_ids, encoded_units):
                    self.config.on_encode_complete(unit_id, unit)

            self._auto_manage()

            return unit_ids
        except Exception as e:
            self.logger.error(f"Failed to add trajectory batch: {str(e)}")
            raise RuntimeError(f"Add trajectory batch failed: {str(e)}")

    def query_memory(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query memory for relevant information.

        This method searches the memory system for memories relevant to the given query,
        using the configured retrieval strategy (keyword, semantic, hybrid, or LLM-guided).

        Args:
            query: Natural language query describing the information needed
            top_k: Maximum number of memories to return. If None, uses config default.
            filters: Optional filters to narrow search results. Supported filters:
                - "types": List of memory types to include (e.g., ["lesson", "skill"])
                - "tags": List of tags to match
                - "date_range": Dict with "start" and "end" date strings
                - Custom filters depending on retrieval strategy

        Returns:
            List of memory units, each containing:
                - "id": Unique memory identifier
                - "type": Memory type (lesson, skill, tool, abstraction)
                - "content": The memory content
                - "tags": Associated tags
                - "metadata": Additional metadata
                - "score": Relevance score (0-1, higher is more relevant)

        Raises:
            RuntimeError: If retrieval context or storage are not initialized

        Examples:
            >>> # Basic query
            >>> results = memory.query_memory("debugging techniques")
            >>> for result in results:
            ...     print(f"{result['type']}: {result['content'][:50]}...")

            >>> # Query with filters
            >>> results = memory.query_memory(
            ...     "Python programming",
            ...     top_k=5,
            ...     filters={"types": ["skill", "tool"]}
            ... )

            >>> # Use with LLM-guided retrieval
            >>> results = memory.query_memory(
            ...     "How do I optimize database queries?",
            ...     filters={"tags": ["database", "performance"]}
            ... )
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
                self.manage_memory("prune", criteria={
                                   "max_count": target_count})
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
