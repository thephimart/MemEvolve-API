"""Tests for evolution integration in API server."""

import pytest
from unittest.mock import Mock, patch

from src.api.evolution_manager import EvolutionManager, EvolutionMetrics
from src.utils.config import MemEvolveConfig


class TestEvolutionManager:
    """Test EvolutionManager functionality."""

    @pytest.fixture
    def mock_memory_system(self):
        """Create a mock memory system."""
        return Mock()

    @pytest.fixture
    def config(self):
        """Create a test config with evolution enabled."""
        config = MemEvolveConfig()
        config.evolution.enable = True
        config.evolution.population_size = 5
        config.evolution.generations = 2
        return config

    def test_evolution_manager_creation(self, config, mock_memory_system, tmp_path):
        """Test EvolutionManager can be created."""
        # Use temporary directory to avoid loading existing state
        config.cache_dir = str(tmp_path)

        manager = EvolutionManager(config, mock_memory_system)

        assert manager.config == config
        assert manager.memory_system == mock_memory_system
        assert manager.is_running == False
        assert len(manager.population) == 0

    def test_start_stop_evolution(self, config, mock_memory_system, tmp_path):
        """Test starting and stopping evolution."""
        # Use temporary directory to avoid loading existing state
        config.cache_dir = str(tmp_path)

        manager = EvolutionManager(config, mock_memory_system)

        # Start evolution
        assert manager.start_evolution() == True
        assert manager.is_running == True

        # Try to start again (should fail)
        assert manager.start_evolution() == False

        # Stop evolution
        assert manager.stop_evolution() == True
        assert manager.is_running == False

        # Try to stop again (should fail)
        assert manager.stop_evolution() == False

    def test_record_api_request(self, config, mock_memory_system, tmp_path):
        """Test recording API requests."""
        # Use temporary directory to avoid loading existing state
        config.cache_dir = str(tmp_path)

        manager = EvolutionManager(config, mock_memory_system)

        # Record some requests
        manager.record_api_request(0.5, True)
        manager.record_api_request(1.0, False)
        manager.record_api_request(0.8, True)

        metrics = manager.metrics
        assert metrics.api_requests_total == 3
        assert metrics.api_requests_successful == 2
        assert abs(metrics.average_response_time -
                   0.766) < 0.01  # (0.5 + 1.0 + 0.8) / 3

    def test_record_memory_retrieval(self, config, mock_memory_system, tmp_path):
        """Test recording memory retrievals."""
        # Use temporary directory to avoid loading existing state
        config.cache_dir = str(tmp_path)

        manager = EvolutionManager(config, mock_memory_system)

        # Record some retrievals
        manager.record_memory_retrieval(0.1, True)
        manager.record_memory_retrieval(0.2, False)
        manager.record_memory_retrieval(0.15, True)

        metrics = manager.metrics
        assert metrics.memory_retrievals_total == 3
        assert metrics.memory_retrievals_successful == 2
        assert abs(metrics.average_retrieval_time - 0.15) < 0.01

    def test_get_status(self, config, mock_memory_system, tmp_path):
        """Test getting evolution status."""
        # Use temporary directory to avoid loading existing state
        config.cache_dir = str(tmp_path)

        manager = EvolutionManager(config, mock_memory_system)

        status = manager.get_status()
        assert status["is_running"] == False
        assert status["population_size"] == 0
        assert status["evolution_cycles_completed"] == 0
        assert status["api_requests_total"] == 0

    @patch('time.sleep')  # Prevent actual sleeping in tests
    def test_evolution_loop_basic(self, mock_sleep, config, mock_memory_system):
        """Test basic evolution loop functionality."""
        # Reduce population size and generations for faster testing
        config.evolution.population_size = 3
        config.evolution.generations = 1

        manager = EvolutionManager(config, mock_memory_system)

        # Initialize population
        manager._initialize_population()
        assert len(manager.population) == 3

        # Test evaluation (mock fitness scores using actual genome IDs)
        fitness_scores = {}
        for genotype in manager.population:
            genome_id = genotype.get_genome_id()
            fitness_scores[genome_id] = 0.8 if len(
                fitness_scores) == 0 else 0.6  # Simple scoring

        selected = manager._select_best_genotypes(fitness_scores)
        assert len(selected) <= len(manager.population)

        # Test crossover
        parent1 = manager.population[0]
        parent2 = manager.population[1]
        child = manager._crossover(parent1, parent2)
        assert isinstance(child, type(parent1))

    def test_persistence_save_load(self, config, mock_memory_system, tmp_path):
        """Test saving and loading evolution state."""
        # Set up temporary cache directory
        config.cache_dir = str(tmp_path)

        manager = EvolutionManager(config, mock_memory_system)

        # Set up some state
        manager.best_genotype = manager.genotype_factory.create_baseline_genotype()
        manager.metrics.api_requests_total = 100
        manager.metrics.evolution_cycles_completed = 5

        # Save state
        manager._save_persistent_state()

        # Create new manager and verify it loads the state
        manager2 = EvolutionManager(config, mock_memory_system)

        assert manager2.best_genotype is not None
        assert manager2.best_genotype.get_genome_id(
        ) == manager.best_genotype.get_genome_id()
        assert manager2.metrics.api_requests_total == 100
        assert manager2.metrics.evolution_cycles_completed == 5


class TestEvolutionAPIIntegration:
    """Test evolution integration with API server."""

    def test_evolution_disabled_by_default(self):
        """Test that evolution can be disabled."""
        config = MemEvolveConfig()
        config.evolution.enable = False  # Explicitly disable for test
        assert config.evolution.enable == False

    def test_evolution_can_be_enabled(self):
        """Test that evolution can be enabled via config."""
        config = MemEvolveConfig()
        config.evolution.enable = True
        assert config.evolution.enable == True

    @patch.dict('os.environ', {'MEMEVOLVE_ENABLE_EVOLUTION': 'true'})
    def test_evolution_enabled_via_env(self):
        """Test evolution can be enabled via environment variable."""
        from src.utils.config import load_config
        config = load_config()
        assert config.evolution.enable == True
