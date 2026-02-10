import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from .components.encode import ExperienceEncoder
from .components.manage import HealthMetrics, ManagementStrategy, MemoryManager
from .components.retrieve import (HybridRetrievalStrategy,
                                  KeywordRetrievalStrategy, RetrievalContext,
                                  RetrievalResult, RetrievalStrategy,
                                  SemanticRetrievalStrategy)
from .components.store import StorageBackend
from .utils.config import ConfigManager, MemEvolveConfig, load_config
from .utils.embeddings import create_embedding_function
from .utils.logging_manager import LoggingManager


class ComponentType(Enum):
    """Enumeration of memory system component types for reconfiguration."""
    ENCODER = "encoder"
    STORAGE = "storage"
    RETRIEVAL = "retrieval"
    MANAGER = "manager"


@dataclass
class MemorySystemConfig:
    """Configuration for MemorySystem.

    This class holds all configuration options for a MemEvolve memory system,
    including LLM settings, storage preferences, and behavioral parameters.

    Example:
        >>> config = MemorySystemConfig(
        ...     memory_base_url="http://localhost:11433",
        ...     memory_api_key="your-api-key",
        ...     default_retrieval_top_k=10,
        ...     enable_auto_management=True
        ... )
    """

    memory_base_url: str = field(
        default="", metadata={
            "help": "Base URL for Memory API (port 11433) - dedicated memory encoding service"})
    memory_api_key: str = field(
        default="",
        metadata={"help": "API key for Memory API authentication"}
    )
    memory_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "Memory API model name (optional, may be inferred from API)"}
    )
    memory_timeout: int = field(
        default=600,
        metadata={"help": "Timeout for Memory API requests in seconds"}
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
        >>> # Configure with Memory API settings
        >>> config = MemorySystemConfig(
        ...     memory_base_url="http://localhost:11433",
        ...     memory_api_key="your-api-key"
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
        >>> from .components.store import GraphStorageBackend
        >>> config.storage_backend = GraphStorageBackend()
        >>> memory = MemorySystem(config)
        >>>
        >>> # Batch processing for better performance
        >>> experiences = [exp1, exp2, exp3]
        >>> memory.add_trajectory_batch(experiences)
        >>>
        >>> # Custom retrieval with LLM guidance
        >>> from .components.retrieve import APIGuidedRetrievalStrategy
from .utils.config import ConfigManager
        >>> llm_func = lambda prompt: "your-llm-response"
        >>> config.retrieval_strategy = APIGuidedRetrievalStrategy(api_func)
    """

    def __init__(
        self,
        config: Optional[Union[MemorySystemConfig, MemEvolveConfig]] = None,
        encoder: Optional[Any] = None,
        evolution_manager: Optional[Any] = None
    ):
        # Store the original config for strategy creation
        self._original_config = config

        if isinstance(config, MemEvolveConfig):
            # Store the full MemEvolveConfig for encoder access
            self._mem_evolve_config = config
            # Create ConfigManager for P0.51 compliance - will use .env by default
            self.config_manager = ConfigManager()
            # Convert MemEvolveConfig to MemorySystemConfig
            self.config = MemorySystemConfig(
                memory_base_url=config.memory.base_url or "",
                memory_api_key=config.memory.api_key or "",
                memory_model=config.memory.model or "",
                memory_timeout=config.memory.timeout,
                default_retrieval_top_k=config.retrieval.default_top_k,
                enable_auto_management=config.management.enable_auto_management,
                auto_prune_threshold=config.management.auto_prune_threshold,
                log_level=config.logging.level
            )
        elif config is None:
            # Load centralized config if no config provided
            centralized_config = load_config()
            self._mem_evolve_config = centralized_config
            # Create ConfigManager for P0.51 compliance - will use .env by default
            self.config_manager = ConfigManager()
            # Convert MemEvolveConfig to MemorySystemConfig
            self.config = MemorySystemConfig(
                memory_base_url=centralized_config.memory.base_url or "",
                memory_api_key=centralized_config.memory.api_key or "",
                memory_model=centralized_config.memory.model or "",
                memory_timeout=centralized_config.memory.timeout,
                default_retrieval_top_k=centralized_config.retrieval.default_top_k,
                enable_auto_management=centralized_config.management.enable_auto_management,
                auto_prune_threshold=centralized_config.management.auto_prune_threshold,
                log_level=centralized_config.logging.level
            )
        else:
            self.config = config
            # Create ConfigManager for MemorySystemConfig case
            self.config_manager = ConfigManager()

        self._provided_encoder = encoder
        self._evolution_manager = evolution_manager
        self._setup_logging()

        self.encoder: Optional[ExperienceEncoder] = None
        self.storage: Optional[StorageBackend] = None
        self.retrieval_context: Optional[RetrievalContext] = None
        self.memory_manager: Optional[MemoryManager] = None

        self.operation_log: List[Dict[str, Any]] = []

        # Bad memory cleanup tracking
        self._memories_since_cleanup: int = 0
        self._cleanup_interval: int = 512  # Cleanup every 512 memories

        self._initialize_components()

    def _setup_logging(self):
        """Configure logging based on config."""
        # Only set basicConfig if logging hasn't been configured yet
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=getattr(logging, self.config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        self.logger = LoggingManager.get_logger(__name__)

        # Legacy file logging removed - now handled by LoggingManager

    def _initialize_components(self):
        """Initialize all components based on configuration."""
        try:
            self._initialize_encoder()
            self._initialize_storage()
            self._initialize_retrieval()
            self._initialize_management()
            self.logger.debug("All components initialized successfully")
            self.logger.info(f"[STORAGE_DEBUG] 🏗️ MemorySystem initialized with storage: {type(self.storage).__name__}")
        except Exception as e:
            self.logger.error(f"Component initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize MemorySystem: {str(e)}")

    def reconfigure_component(
        self,
        component_type: ComponentType,
        new_component: Any,
        validate: bool = True,
        save_state: bool = True
    ) -> bool:
        """Reconfigure a component with hot-swapping capability.

        This method allows dynamic reconfiguration of memory system components
        without restarting the system. It supports safe rollback and state preservation.

        Args:
            component_type: Type of component to reconfigure
            new_component: New component instance to install
            validate: Whether to validate component before installation
            save_state: Whether to save current component state for rollback

        Returns:
            True if reconfiguration successful, False otherwise

        Raises:
            ValueError: If component type or validation fails
            RuntimeError: If reconfiguration fails and rollback unsuccessful
        """
        try:
            # Validate component type
            if not isinstance(component_type, ComponentType):
                raise ValueError(f"Invalid component type: {component_type}")

            # Validate new component if requested
            if validate:
                self._validate_component(component_type, new_component)

            # Save current state for rollback
            saved_state = None
            if save_state:
                saved_state = self._save_component_state(component_type)

            # Store old component for rollback
            old_component = self._get_current_component(component_type)

            # Install new component
            self._install_component(component_type, new_component)

            # Test the new component
            try:
                self._test_component(component_type, new_component)

                # Enhanced logging with component details
                component_name = component_type.value
                if hasattr(new_component, '__class__'):
                    component_class = new_component.__class__.__name__
                else:
                    component_class = str(type(new_component).__name__)

                # Log component type and class for better tracking
                self.logger.info(
                    f"✅ Successfully reconfigured {component_name} → {component_class}")

                # Log additional component details if available
                try:
                    details = []
                    if hasattr(new_component, 'strategy_type'):
                        details.append(f"strategy={new_component.strategy_type}")
                    if hasattr(new_component, 'model'):
                        details.append(f"model={new_component.model}")
                    if hasattr(new_component, 'max_tokens'):
                        details.append(f"max_tokens={new_component.max_tokens}")
                    if hasattr(new_component, 'batch_size'):
                        details.append(f"batch_size={new_component.batch_size}")

                    if details:
                        self.logger.info(f"🔧 {component_name} configuration: {', '.join(details)}")

                except Exception as detail_error:
                    # Don't let detail logging fail the reconfiguration
                    self.logger.debug(f"Could not log component details: {detail_error}")

                return True
            except Exception as test_error:
                self.logger.warning(f"Component test failed, rolling back: {test_error}")
                # Rollback to old component
                if old_component:
                    self._install_component(component_type, old_component)
                    if saved_state:
                        self._restore_component_state(component_type, saved_state)
                raise RuntimeError(
                    f"Component reconfiguration failed and rolled back: {test_error}")

        except Exception as e:
            self.logger.error(f"Component reconfiguration failed: {str(e)}")
            raise

    def _validate_component(self, component_type: ComponentType, component: Any):
        """Validate a component before installation."""
        if component is None:
            raise ValueError(f"Component cannot be None for {component_type.value}")

        # Type-specific validation
        if component_type == ComponentType.ENCODER:
            required_methods = ['encode_experience', 'encode_trajectory']
        elif component_type == ComponentType.STORAGE:
            required_methods = ['store', 'retrieve', 'count']
        elif component_type == ComponentType.RETRIEVAL:
            required_methods = ['retrieve']
        elif component_type == ComponentType.MANAGER:
            required_methods = ['prune', 'consolidate']
        else:
            raise ValueError(f"Unknown component type: {component_type}")

        # Check required methods exist
        for method in required_methods:
            if not hasattr(component, method):
                raise ValueError(f"Component missing required method: {method}")

    def _save_component_state(self, component_type: ComponentType) -> Any:
        """Save current component state for rollback."""
        component = self._get_current_component(component_type)
        if component and hasattr(component, 'save_state'):
            try:
                return component.save_state()
            except Exception as e:
                self.logger.warning(f"Failed to save state for {component_type.value}: {e}")
        return None

    def _restore_component_state(self, component_type: ComponentType, state: Any):
        """Restore component state after rollback."""
        component = self._get_current_component(component_type)
        if component and hasattr(component, 'restore_state') and state:
            try:
                component.restore_state(state)
            except Exception as e:
                self.logger.warning(f"Failed to restore state for {component_type.value}: {e}")

    def _get_current_component(self, component_type: ComponentType) -> Any:
        """Get the currently installed component."""
        if component_type == ComponentType.ENCODER:
            return self.encoder
        elif component_type == ComponentType.STORAGE:
            return self.storage
        elif component_type == ComponentType.RETRIEVAL:
            return self.retrieval_context.strategy if self.retrieval_context else None
        elif component_type == ComponentType.MANAGER:
            return self.memory_manager
        return None

    def _install_component(self, component_type: ComponentType, component: Any):
        """Install a new component."""
        if component_type == ComponentType.ENCODER:
            self.encoder = component
        elif component_type == ComponentType.STORAGE:
            self.storage = component
            # Reinitialize manager if it depends on storage
            if self.memory_manager:
                self.memory_manager.storage_backend = component
        elif component_type == ComponentType.RETRIEVAL:
            if self.retrieval_context:
                self.retrieval_context.strategy = component
            else:
                from .components.retrieve import RetrievalContext
                self.retrieval_context = RetrievalContext(
                    strategy=component,
                    default_top_k=self.config.default_retrieval_top_k
                )
        elif component_type == ComponentType.MANAGER:
            self.memory_manager = component

    def _test_component(self, component_type: ComponentType, component: Any):
        """Test a newly installed component."""
        try:
            if component_type == ComponentType.ENCODER:
                # Test with minimal experience
                test_exp = {"action": "test", "result": "ok"}
                component.encode_experience(test_exp)
            elif component_type == ComponentType.STORAGE:
                # Test basic operations
                component.count()
            elif component_type == ComponentType.RETRIEVAL:
                # Test retrieval interface
                pass  # Retrieval testing requires storage, handled elsewhere
            elif component_type == ComponentType.MANAGER:
                # Test manager interface
                pass  # Manager testing requires storage, handled elsewhere
        except Exception as e:
            raise RuntimeError(f"Component test failed: {e}")

    def _initialize_encoder(self):
        """Initialize encoder component."""
        if self.encoder is None:
            # Use provided encoder if available, otherwise create default
            if hasattr(self, '_provided_encoder') and self._provided_encoder:
                self.encoder = self._provided_encoder
                self.logger.info("Using provided encoder")
            else:
                self.logger.debug(
                    f"Initializing encoder with base_url: {
                        self.config.memory_base_url}")
                # Get encoding strategies with priority: evolution > config
                encoding_strategies = None
                evolution_encoding_strategies = None

                # Priority 1: Check evolution manager for current genotype
                if (self._evolution_manager and
                    hasattr(self._evolution_manager, 'current_genotype') and
                        self._evolution_manager.current_genotype):
                    evolution_encoding_strategies = (
                        self._evolution_manager.current_genotype.encode.encoding_strategies)

                # Priority 2: Use config if no evolution override
                if (not evolution_encoding_strategies and
                    hasattr(self, '_mem_evolve_config') and
                        self._mem_evolve_config):
                    encoding_strategies = self._mem_evolve_config.encoder.encoding_strategies

                self.encoder = ExperienceEncoder(
                    config_manager=self.config_manager,
                    encoding_strategies=encoding_strategies,
                    evolution_encoding_strategies=evolution_encoding_strategies
                )
                self.encoder.initialize_memory_api()
                self.logger.debug("Encoder initialized")

    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension from centralized config with evolution state override."""
        import json
        import os

        # 1. Check evolution_state.json for dynamic override
        if hasattr(self, '_mem_evolve_config') and self._mem_evolve_config:
            data_dir = self._mem_evolve_config.data_dir
        else:
            raise ValueError("MemorySystem requires MemEvolveConfig for embedding dimension")

        evolution_dir = os.path.join(data_dir, 'evolution')
        os.makedirs(evolution_dir, exist_ok=True)
        evolution_state_path = os.path.join(evolution_dir, 'evolution_state.json')
        if os.path.exists(evolution_state_path):
            try:
                with open(evolution_state_path, 'r') as f:
                    evolution_data = json.load(f)
                    # Look for current genotype embedding dimension
                    current_genotype = evolution_data.get('current_genotype', {})
                    if 'embedding_dim' in current_genotype:
                        dim = current_genotype['embedding_dim']
                        self.logger.debug(f"Using embedding dimension from evolution state: {dim}")
                        return dim
            except Exception as e:
                self.logger.warning(f"Failed to read evolution state: {e}")

        # 2. Use centralized config embedding dimension (which already has fallback logic)
        if hasattr(self, '_mem_evolve_config') and self._mem_evolve_config.embedding.dimension:
            dim = self._mem_evolve_config.embedding.dimension
            self.logger.debug(f"Using embedding dimension from config: {dim}")
            return dim

        # 3. Fallback to 768 (should not happen with proper config)
        self.logger.debug("Using default embedding dimension: 768")
        return 768

    def _initialize_storage(self):
        """Initialize the storage backend based on configuration."""
        # Use provided storage backend if available
        if self.config.storage_backend is not None:
            self.storage = self.config.storage_backend
            self.logger.info(f"Using provided storage backend: {type(self.storage).__name__}")
            return

        # Require MemEvolveConfig for storage configuration
        if not hasattr(self, '_mem_evolve_config') or not self._mem_evolve_config:
            raise ValueError("MemorySystem requires MemEvolveConfig for storage initialization")

        # Get storage configuration from centralized config
        backend_type = self._mem_evolve_config.storage.backend_type
        data_dir = self._mem_evolve_config.data_dir
        index_type = self._mem_evolve_config.storage.index_type

        os.makedirs(data_dir, exist_ok=True)

        # Create memory subdirectory for all backends
        memory_dir = os.path.join(data_dir, "memory")
        os.makedirs(memory_dir, exist_ok=True)

        # Instantiate the appropriate backend
        if backend_type == 'vector':
            from .components.store import VectorStore
            from .utils.embeddings import create_embedding_function

            index_file = os.path.join(memory_dir, "vector")

            # Use embedding configuration from MemEvolveConfig
            embedding_function = create_embedding_function(
                provider="openai",
                base_url=self._mem_evolve_config.embedding.base_url,
                api_key=self._mem_evolve_config.embedding.api_key
            )

            embedding_dim = self._get_embedding_dimension()

            self.storage = VectorStore(
                index_file=index_file,
                embedding_function=embedding_function,
                embedding_dim=embedding_dim,
                index_type=index_type
            )
            self.logger.debug(
                f"Initialized vector storage backend at {index_file} with index type: {index_type}, embedding dim: {embedding_dim}")
        elif backend_type == 'graph':
            from .components.store import GraphStorageBackend

            # Use centralized Neo4j configuration
            neo4j_config = getattr(self._mem_evolve_config, 'neo4j', None)
            if neo4j_config and neo4j_config.disabled:
                # Neo4j is disabled via config, use JSON fallback
                self.logger.info("Neo4j disabled via config. Using JSON fallback.")
                from .components.store import JSONFileStore
                storage_path = os.path.join(memory_dir, "memory_system.json")
                self.storage = JSONFileStore(storage_path)
                self.logger.info("Initialized JSON storage backend (Neo4j disabled)")
            elif neo4j_config:
                neo4j_uri = neo4j_config.uri
                neo4j_user = neo4j_config.user
                neo4j_password = neo4j_config.password
                self.storage = GraphStorageBackend(
                    uri=neo4j_uri,
                    user=neo4j_user,
                    password=neo4j_password
                )
                self.logger.info("Initialized graph storage backend")
            else:
                raise RuntimeError(
                    "Graph storage backend requires Neo4j configuration. "
                    "Ensure MemEvolveConfig is passed to MemorySystem."
                )
        else:  # json (default)
            from .components.store import JSONFileStore
            storage_path = os.path.join(memory_dir, "memory.json")
            self.storage = JSONFileStore(storage_path)
            self.logger.info(f"Initialized JSON storage backend at {storage_path}")

        self.logger.debug(f"Storage backend created: {backend_type}")

    def _initialize_retrieval(self):
        """Initialize the retrieval context."""
        if hasattr(self.config, 'retrieval_strategy') and self.config.retrieval_strategy:
            self.retrieval_context = RetrievalContext(
                strategy=self.config.retrieval_strategy,
                default_top_k=self.config.default_retrieval_top_k
            )
            self.logger.info("Retrieval strategy configured")
        else:
            # Try to create strategy from MemEvolveConfig strategy_type
            strategy = None
            if isinstance(self._original_config, MemEvolveConfig):
                strategy_type = self._original_config.retrieval.strategy_type
                try:
                    strategy = self._create_strategy_from_type(strategy_type)
                    self.logger.debug(f"Created {strategy_type} retrieval strategy")
                except Exception as e:
                    self.logger.warning(f"Failed to create {strategy_type} strategy: {e}")

            if strategy:
                self.retrieval_context = RetrievalContext(
                    strategy=strategy,
                    default_top_k=self.config.default_retrieval_top_k
                )
                self.logger.debug("Retrieval context initialized from config")
            else:
                # Skip retrieval initialization for now - will be handled when needed
                self.logger.info(
                    "Retrieval initialization skipped - will use on-demand")

    def _create_strategy_from_type(self, strategy_type: str) -> RetrievalStrategy:
        """Create a retrieval strategy from strategy type string."""
        if not isinstance(self._original_config, MemEvolveConfig):
            raise ValueError("Cannot create strategy without MemEvolveConfig")

        config = self._original_config  # Type: MemEvolveConfig

        if strategy_type == "keyword":
            return KeywordRetrievalStrategy()
        elif strategy_type == "semantic":
            # Create embedding function for semantic search
            embedding_function = create_embedding_function(
                provider="openai",  # Will fall back to configured embedding endpoint
                base_url=config.embedding.base_url,
                api_key=config.embedding.api_key
            )
            return SemanticRetrievalStrategy(embedding_function=embedding_function)
        elif strategy_type == "hybrid":
            # Create embedding function for hybrid search
            embedding_function = create_embedding_function(
                provider="openai",  # Will fall back to configured embedding endpoint
                base_url=config.embedding.base_url,
                api_key=config.embedding.api_key
            )
            # Use ConfigManager for component access (P0.51 compliance)
            config_manager = self.config_manager

            return HybridRetrievalStrategy(
                embedding_function=embedding_function,
                config_manager=config_manager
            )
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def _initialize_management(self):
        """Initialize the memory manager."""
        if self.storage:
            config_manager = self.config_manager
            self.memory_manager = MemoryManager(
                storage_backend=self.storage,
                config_manager=config_manager
            )
            self.logger.info("Memory manager configured")

        else:
            self.logger.debug("Default memory manager created")

    def add_experience(self, experience: Dict[str, Any]) -> Optional[str]:
        """Add a single experience to memory.

        This method processes a raw experience through the encoding pipeline,
        transforms it into a structured memory unit, and stores it for future retrieval.
        When batch processing is used (large experiences), multiple memory units may be created.

        Args:
            experience: Dictionary containing experience data. Common fields:
                - "action": What was done (e.g., "searched database")
                - "result": What happened (e.g., "found relevant records")
                - "context": Additional context information
                - "timestamp": When the experience occurred
                - "metadata": Any additional relevant data

        Returns:
            The unique ID of the first stored memory unit, or None if no units were created

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

            self.logger.info(f"[STORAGE_DEBUG] 📥 Adding experience: {experience.get('id', 'unknown')}")
            encoded_result = self.encoder.encode_experience(experience)
            self.logger.info(f"[STORAGE_DEBUG] ✅ Encoding completed, result type: {type(encoded_result).__name__}")

            # Handle both single unit and batch processing (list of units)
            if isinstance(encoded_result, list):
                # Filter out bad memories BEFORE storage
                valid_results = []
                rejected_count = 0

                for result in encoded_result:
                    encoding_method = result.get("metadata", {}).get("encoding_method")
                    if encoding_method not in ["fallback_chunk", "chunk_error"]:
                        valid_results.append(result)
                    else:
                        rejected_count += 1
                        self.logger.warning(
                            f"Rejecting fallback chunk: {
                                result.get(
                                    'content', '')[
                                    :50]}...")

                if rejected_count > 0:
                    self.logger.info(
                        f"Rejected {rejected_count} bad memory chunks, keeping {
                            len(valid_results)}")

                self.logger.info(f"[STORAGE_DEBUG] 📦 Storing {len(valid_results)} valid memory units from batch processing")
                unit_ids = self.storage.store_batch(valid_results)
                self.logger.info(f"[STORAGE_DEBUG] ✅ Batch storage completed, received {len(unit_ids)} unit IDs: {unit_ids[:3]}{'...' if len(unit_ids) > 3 else ''}")

                self._log_operation(
                    "add_experience",
                    {"experience_id": experience.get("id"), "unit_count": len(unit_ids)}
                )

                if self.config.on_encode_complete:
                    for unit_id, encoded_unit in zip(unit_ids, encoded_result):
                        self.config.on_encode_complete(unit_id, encoded_unit)

                self._auto_manage()

                return unit_ids[0] if unit_ids else None

            self.logger.info(f"[STORAGE_DEBUG] 📦 Storing single memory unit")
            unit_id = self.storage.store(encoded_result)
            self.logger.info(f"[STORAGE_DEBUG] ✅ Single storage completed, unit ID: {unit_id}")

            self._log_operation(
                "add_experience",
                {"experience_id": experience.get("id"), "unit_id": unit_id}
            )

            if self.config.on_encode_complete:
                self.config.on_encode_complete(unit_id, encoded_result)

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

            # Use config default when top_k is None
            if top_k is None:
                top_k = getattr(
                    self._mem_evolve_config,
                    'retrieval',
                    {}).get(
                    'default_top_k',
                    5) if hasattr(
                    self,
                    '_mem_evolve_config') else 5

            self.logger.info(f"[STORAGE_DEBUG] 🔍 Querying memory: '{query}' (top_k={top_k})")
            results = self.retrieval_context.retrieve(
                query=query,
                storage_backend=self.storage,
                top_k=top_k,
                filters=filters
            )
            self.logger.info(f"[STORAGE_DEBUG] 📊 Retrieval completed: found {len(results)} results")

            # Detailed memory retrieval logging
            self._log_memory_retrieval(query, results, top_k or 0, filters)

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

            # Include score in returned memory dictionaries
            return [
                {
                    **r.unit,
                    'score': r.score,
                    'retrieval_metadata': r.metadata or {}
                }
                for r in results
            ]
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

            # Include score in returned memory dictionaries (score may be 1.0 for
            # direct ID retrieval)
            return [
                {
                    **r.unit,
                    'score': r.score,
                    'retrieval_metadata': r.metadata or {}
                }
                for r in results
            ]
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

            # Periodic bad memory cleanup
            self._memories_since_cleanup += 1
            if self._memories_since_cleanup >= self._cleanup_interval:
                self._cleanup_bad_memories()
                self._memories_since_cleanup = 0
        except Exception as e:
            self.logger.warning(f"Auto-management failed: {str(e)}")

    def _cleanup_bad_memories_at_startup(self):
        """Perform bad memory cleanup at system startup."""
        try:
            self.logger.info("Performing startup bad memory cleanup...")
            removed_count = self._cleanup_bad_memories()
            self.logger.info(f"Startup cleanup complete: removed {removed_count} bad memories")
        except Exception as e:
            self.logger.warning(f"Startup bad memory cleanup failed: {str(e)}")

    def _cleanup_bad_memories(self) -> int:
        """Detect and remove bad memories (fallback chunks, errors).

        Returns:
            Number of bad memories removed
        """
        if not self.storage:
            return 0

        try:
            all_units = self.storage.retrieve_all()
            bad_memory_ids = []

            for unit in all_units:
                unit_id = unit.get("id", "")
                metadata = unit.get("metadata", {})
                encoding_method = metadata.get("encoding_method", "")
                content = unit.get("content", "")
                tags = unit.get("tags", [])

                # Detect bad memories
                is_bad = False

                # 1. Fallback chunk errors
                if encoding_method in ["fallback_chunk", "chunk_error"]:
                    is_bad = True
                    self.logger.debug(f"Bad memory detected (fallback): {unit_id}")

                # 2. Content indicates processing error
                if "Chunk" in content and "processing" in content.lower():
                    is_bad = True
                    self.logger.debug(f"Bad memory detected (chunk processing): {unit_id}")

                # 3. Error tags
                if any(tag in ["fallback", "chunk_error", "processing_error"] for tag in tags):
                    is_bad = True
                    self.logger.debug(f"Bad memory detected (error tags): {unit_id}")

                # 4. Empty or near-empty content
                if len(content.strip()) < 20:
                    is_bad = True
                    self.logger.debug(f"Bad memory detected (empty content): {unit_id}")

                if is_bad:
                    bad_memory_ids.append(unit_id)

            # Remove bad memories
            removed_count = 0
            for unit_id in bad_memory_ids:
                try:
                    self.storage.delete(unit_id)
                    removed_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete bad memory {unit_id}: {e}")

            if removed_count > 0:
                self.logger.info(
                    f"Bad memory cleanup: removed {removed_count} bad memories "
                    f"({len(bad_memory_ids)} detected)"
                )

            return removed_count

        except Exception as e:
            self.logger.error(f"Bad memory cleanup failed: {str(e)}")
            return 0

    def _log_operation(self, operation: str, details: Dict[str, Any]):
        """Log an operation to operation log."""
        # Check if operation logging is enabled (always enabled with simplified config)
        if (self._mem_evolve_config and
            hasattr(self._mem_evolve_config, 'logging') and
                getattr(self._mem_evolve_config.logging, 'enable', True)):
            self.operation_log.append({
                "operation": operation,
                "details": details,
                "timestamp": self._get_timestamp()
            })

    def _log_memory_retrieval(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ):
        """Log detailed memory retrieval information."""
        import logging
        # Use self.logger already configured in __init__

        # Get retrieval strategy information
        strategy_info = "unknown"
        if self.retrieval_context and hasattr(self.retrieval_context, 'strategy'):
            strategy = getattr(self.retrieval_context, 'strategy', None)
            if strategy:
                strategy_info = type(strategy).__name__

        # Log retrieval summary
        self.logger.info(
            f"Memory retrieval: query='{query[:100]}{'...' if len(query) > 100 else ''}', "
            f"strategy={strategy_info}, requested={top_k}, found={len(results)}"
        )

        # Log detailed results
        if results:
            self.logger.info(f"Top {min(len(results), 3)} retrieved memories:")
            for i, result in enumerate(results[:3]):  # Log top 3 results
                unit_content = result.unit.get('content', '') if result.unit else ''
                content_preview = unit_content[:200] + ('...' if len(unit_content) > 200 else '')

                self.logger.info(
                    f"  #{i + 1}: id={result.unit_id}, score={result.score:.3f}, "
                    f"content='{content_preview}'"
                )

                # Log metadata if available
                if result.metadata:
                    metadata_str = ", ".join(f"{k}={v}" for k, v in result.metadata.items())
                    self.logger.info(f"    metadata: {metadata_str}")

        # Log retrieval metrics
        if results:
            scores = [r.score for r in results]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)

            self.logger.info(
                f"Retrieval metrics: avg_score={avg_score:.3f}, "
                f"max_score={max_score:.3f}, min_score={min_score:.3f}"
            )

        # Log filters if applied
        if filters:
            filters_str = ", ".join(f"{k}={v}" for k, v in filters.items())
            self.logger.info(f"Applied filters: {filters_str}")

    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.now(timezone.utc).isoformat() + "Z"
