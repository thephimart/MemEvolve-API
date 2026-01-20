"""Tests for configuration management system."""

from utils.config import (
    LLMConfig,
    StorageConfig,
    RetrievalConfig,
    ManagementConfig,
    EncoderConfig,
    EvolutionConfig,
    LoggingConfig,
    MemEvolveConfig,
    ConfigManager,
    load_config,
    save_config
)
from unittest.mock import patch
import pytest
import json
import tempfile
from pathlib import Path
import sys
import os
sys.path.insert(0, 'src')


sys.path.insert(0, 'src')


class TestLLMConfig:
    """Test LLM configuration."""

    def test_default_values(self):
        config = LLMConfig()
        # When env vars are loaded, defaults may be overridden
        import os
        assert config.base_url == os.getenv("MEMEVOLVE_LLM_BASE_URL", "")
        assert config.api_key == os.getenv("MEMEVOLVE_LLM_API_KEY", "")
        model_env = os.getenv("MEMEVOLVE_LLM_MODEL")
        if model_env is not None:
            assert config.model == model_env
        else:
            assert config.model is None
        auto_resolve_env = os.getenv("MEMEVOLVE_LLM_AUTO_RESOLVE_MODELS")
        if auto_resolve_env is not None:
            expected_auto_resolve = auto_resolve_env.lower() in ("true", "1", "yes", "on")
            assert config.auto_resolve_models == expected_auto_resolve
        else:
            assert config.auto_resolve_models is True
        timeout_env = os.getenv("MEMEVOLVE_LLM_TIMEOUT")
        if timeout_env:
            try:
                expected_timeout = int(timeout_env)
                assert config.timeout == expected_timeout
            except ValueError:
                assert config.timeout == 120
        else:
            assert config.timeout == 120
        max_retries_env = os.getenv("MEMEVOLVE_LLM_MAX_RETRIES")
        if max_retries_env:
            try:
                expected_max_retries = int(max_retries_env)
                assert config.max_retries == expected_max_retries
            except ValueError:
                assert config.max_retries == 3
        else:
            assert config.max_retries == 3

    def test_custom_values(self):
        # Note: Environment variables override constructor arguments
        # This test verifies that non-environment-controlled fields work
        import os
        config = LLMConfig(timeout=60, max_retries=5)

        # Environment-controlled fields use environment values
        assert config.base_url == os.getenv("MEMEVOLVE_LLM_BASE_URL")
        assert config.api_key == os.getenv("MEMEVOLVE_LLM_API_KEY", "")

        # Non-environment fields work as expected
        assert config.timeout == 60
        assert config.max_retries == 5


class TestStorageConfig:
    """Test storage configuration."""

    def test_default_values(self):
        import os
        config = StorageConfig()
        assert config.backend_type == "json"
        assert config.path == os.getenv(
            "MEMEVOLVE_STORAGE_PATH", "./data/memory")
        assert config.vector_dim == 768
        assert config.index_type == "flat"

    def test_custom_values(self):
        # Note: Environment variables override constructor arguments
        # This test verifies that non-environment-controlled fields work
        import os
        config = StorageConfig(
            backend_type="vector",
            vector_dim=1024,
            index_type="hnsw"
        )

        # Environment-controlled fields use environment values
        assert config.path == os.getenv(
            "MEMEVOLVE_STORAGE_PATH", "./data/memory")

        # Non-environment fields work as expected
        assert config.backend_type == "vector"
        assert config.vector_dim == 1024
        assert config.index_type == "hnsw"


class TestRetrievalConfig:
    """Test retrieval configuration."""

    def test_default_values(self):
        config = RetrievalConfig()
        assert config.strategy_type == "hybrid"
        assert config.default_top_k == 5
        assert config.semantic_weight == 0.7
        assert config.keyword_weight == 0.3
        assert config.enable_caching is True
        assert config.cache_size == 1000

    def test_custom_values(self):
        config = RetrievalConfig(
            strategy_type="semantic",
            default_top_k=10,
            semantic_weight=0.9,
            keyword_weight=0.1,
            enable_caching=False,
            cache_size=500
        )
        assert config.strategy_type == "semantic"
        assert config.default_top_k == 10
        assert config.semantic_weight == 0.9
        assert config.keyword_weight == 0.1
        assert config.enable_caching is False
        assert config.cache_size == 500


class TestManagementConfig:
    """Test management configuration."""

    def test_default_values(self):
        config = ManagementConfig()
        assert config.enable_auto_management is True
        assert config.auto_prune_threshold == 1000
        assert config.auto_consolidate_interval == 100
        assert config.deduplicate_threshold == 0.9
        assert config.forgetting_strategy == "lru"
        assert config.max_memory_age_days == 365

    def test_custom_values(self):
        config = ManagementConfig(
            enable_auto_management=False,
            auto_prune_threshold=500,
            auto_consolidate_interval=50,
            deduplicate_threshold=0.8,
            forgetting_strategy="random",
            max_memory_age_days=180
        )
        assert config.enable_auto_management is False
        assert config.auto_prune_threshold == 500
        assert config.auto_consolidate_interval == 50
        assert config.deduplicate_threshold == 0.8
        assert config.forgetting_strategy == "random"
        assert config.max_memory_age_days == 180


class TestEncoderConfig:
    """Test encoder configuration."""

    def test_default_values(self):
        config = EncoderConfig()
        assert config.encoding_strategies == ["lesson", "skill"]
        assert config.enable_abstraction is True
        assert config.abstraction_threshold == 10
        assert config.enable_tool_extraction is True

    def test_custom_values(self):
        config = EncoderConfig(
            encoding_strategies=["lesson", "skill", "tool"],
            enable_abstraction=False,
            abstraction_threshold=20,
            enable_tool_extraction=False
        )
        assert config.encoding_strategies == ["lesson", "skill", "tool"]
        assert config.enable_abstraction is False
        assert config.abstraction_threshold == 20
        assert config.enable_tool_extraction is False


class TestEvolutionConfig:
    """Test evolution configuration."""

    def test_default_values(self):
        config = EvolutionConfig()
        assert config.population_size == 10
        assert config.generations == 20
        assert config.mutation_rate == 0.1
        assert config.crossover_rate == 0.5
        assert config.selection_method == "pareto"
        assert config.tournament_size == 3

    def test_custom_values(self):
        config = EvolutionConfig(
            population_size=20,
            generations=50,
            mutation_rate=0.2,
            crossover_rate=0.7,
            selection_method="tournament",
            tournament_size=5
        )
        assert config.population_size == 20
        assert config.generations == 50
        assert config.mutation_rate == 0.2
        assert config.crossover_rate == 0.7
        assert config.selection_method == "tournament"
        assert config.tournament_size == 5

    def test_validation_valid_config(self):
        """Test that valid configuration passes validation."""
        config = EvolutionConfig(
            population_size=20,
            generations=50,
            mutation_rate=0.2,
            crossover_rate=0.7,
            selection_method="pareto",
            tournament_size=5
        )
        # Should not raise any exception
        assert config.population_size == 20

    def test_validation_invalid_population_size(self):
        """Test validation of population_size."""
        with pytest.raises(ValueError, match="population_size must be an integer >= 3"):
            EvolutionConfig(population_size=2)

        with pytest.raises(ValueError, match="population_size should not exceed 1000"):
            EvolutionConfig(population_size=2000)

    def test_validation_invalid_generations(self):
        """Test validation of generations."""
        with pytest.raises(ValueError, match="generations must be an integer >= 1"):
            EvolutionConfig(generations=0)

        with pytest.raises(ValueError, match="generations should not exceed 1000"):
            EvolutionConfig(generations=2000)

    def test_validation_invalid_mutation_rate(self):
        """Test validation of mutation_rate."""
        with pytest.raises(ValueError, match="mutation_rate must be a float between 0.0 and 1.0"):
            EvolutionConfig(mutation_rate=-0.1)

        with pytest.raises(ValueError, match="mutation_rate must be a float between 0.0 and 1.0"):
            EvolutionConfig(mutation_rate=1.5)

    def test_validation_invalid_crossover_rate(self):
        """Test validation of crossover_rate."""
        with pytest.raises(ValueError, match="crossover_rate must be a float between 0.0 and 1.0"):
            EvolutionConfig(crossover_rate=-0.1)

        with pytest.raises(ValueError, match="crossover_rate must be a float between 0.0 and 1.0"):
            EvolutionConfig(crossover_rate=1.5)

    def test_validation_invalid_selection_method(self):
        """Test validation of selection_method."""
        with pytest.raises(ValueError, match="selection_method must be one of"):
            EvolutionConfig(selection_method="invalid_method")

    def test_validation_invalid_tournament_size(self):
        """Test validation of tournament_size."""
        with pytest.raises(ValueError, match="tournament_size must be an integer >= 2"):
            EvolutionConfig(tournament_size=1)

        with pytest.raises(ValueError, match="tournament_size.*cannot exceed population_size"):
            EvolutionConfig(population_size=5, tournament_size=10)

    def test_validation_tournament_vs_population(self):
        """Test validation that tournament_size <= population_size."""
        with pytest.raises(ValueError, match="population_size.*must be >= tournament_size"):
            EvolutionConfig(population_size=3, tournament_size=5)

    @patch.dict('os.environ', {'MEMEVOLVE_EVOLUTION_POPULATION_SIZE': '2'}, clear=True)
    def test_validation_from_env_vars(self):
        """Test that validation also works when loading from environment variables."""
        with pytest.raises(ValueError, match="population_size must be an integer >= 3"):
            load_config()


class TestLoggingConfig:
    """Test logging configuration."""

    def test_default_values(self):
        config = LoggingConfig()
        level_env = os.getenv("MEMEVOLVE_LOG_LEVEL")
        if level_env is not None:
            assert config.level == level_env
        else:
            assert config.level == "INFO"

        format_env = os.getenv("MEMEVOLVE_LOGGING_FORMAT")
        if format_env is not None:
            assert config.format == format_env
        else:
            assert config.format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        log_file_env = os.getenv("MEMEVOLVE_LOGGING_LOG_FILE")
        if log_file_env is not None:
            assert config.log_file == log_file_env
        else:
            assert config.log_file is None

        enable_op_env = os.getenv("MEMEVOLVE_LOGGING_ENABLE_OPERATION_LOG")
        if enable_op_env is not None:
            assert config.enable_operation_log == (
                enable_op_env.lower() in ("true", "1", "yes", "on"))
        else:
            assert config.enable_operation_log is True

        max_log_env = os.getenv("MEMEVOLVE_LOGGING_MAX_LOG_SIZE_MB")
        if max_log_env is not None:
            assert config.max_log_size_mb == int(max_log_env)
        else:
            assert config.max_log_size_mb == 100

    def test_custom_values(self):
        config = LoggingConfig(
            level="DEBUG",
            format="%(message)s",
            log_file="app.log",
            enable_operation_log=False,
            max_log_size_mb=200
        )
        assert config.level == "DEBUG"
        assert config.format == "%(message)s"
        assert config.log_file == "app.log"
        assert config.enable_operation_log is False
        assert config.max_log_size_mb == 200


class TestMemEvolveConfig:
    """Test main MemEvolve configuration."""

    def test_default_values(self):
        config = MemEvolveConfig()
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.storage, StorageConfig)
        assert isinstance(config.retrieval, RetrievalConfig)
        assert isinstance(config.management, ManagementConfig)
        assert isinstance(config.encoder, EncoderConfig)
        assert isinstance(config.evolution, EvolutionConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert config.project_name == "memevolve"
        assert config.project_root == "."
        assert config.data_dir == "./data"
        assert config.cache_dir == "./cache"


class TestConfigManager:
    """Test configuration manager."""

    def test_initialization_no_file(self):
        manager = ConfigManager()
        assert manager.config_path is None
        assert isinstance(manager.config, MemEvolveConfig)

    def test_initialization_with_invalid_path(self):
        manager = ConfigManager("/nonexistent/path.yaml")
        assert manager.config_path == "/nonexistent/path.yaml"
        assert isinstance(manager.config, MemEvolveConfig)

    def test_update_config(self):
        manager = ConfigManager()
        manager.update(
            **{"llm.base_url": "http://updated:8080/v1"},
            **{"retrieval.default_top_k": 15},
            **{"management.enable_auto_management": False}
        )

        assert manager.config.llm.base_url == "http://updated:8080/v1"
        assert manager.config.retrieval.default_top_k == 15
        assert manager.config.management.enable_auto_management is False

    def test_get_config_value(self):
        import os
        manager = ConfigManager()
        base_url_env = os.getenv("MEMEVOLVE_LLM_BASE_URL")
        upstream_env = os.getenv("MEMEVOLVE_UPSTREAM_BASE_URL")
        expected_base_url = base_url_env or upstream_env or ""
        assert manager.get("llm.base_url") == expected_base_url
        assert manager.get("retrieval.default_top_k") == 5
        assert manager.get("nonexistent.key", "default") == "default"
        assert manager.get("nonexistent.key") is None

    def test_validate_config(self):
        manager = ConfigManager()
        assert manager.validate() is True

        manager.config.retrieval.semantic_weight = 1.5
        assert manager.validate() is False

    def test_to_dict(self):
        manager = ConfigManager()
        config_dict = manager.to_dict()
        assert "llm" in config_dict
        assert "storage" in config_dict
        assert "retrieval" in config_dict
        assert "management" in config_dict
        assert "encoder" in config_dict
        assert "evolution" in config_dict
        assert "logging" in config_dict


class TestLoadConfig:
    """Test load_config function."""

    def test_load_default_config(self):
        config = load_config()
        assert isinstance(config, MemEvolveConfig)
        assert isinstance(config.llm, LLMConfig)


class TestArchitecturePresets:
    """Test architecture preset configurations."""

    def test_agentkb_preset(self):
        config = ConfigManager.get_architecture_config("agentkb")
        assert config["storage"]["backend_type"] == "json"
        assert config["retrieval"]["strategy_type"] == "keyword"
        assert config["management"]["enable_auto_management"] is False
        assert config["encoder"]["encoding_strategies"] == ["lesson"]

    def test_lightweight_preset(self):
        config = ConfigManager.get_architecture_config("lightweight")
        assert config["storage"]["backend_type"] == "json"
        assert config["retrieval"]["strategy_type"] == "hybrid"
        assert config["management"]["enable_auto_management"] is True
        assert config["management"]["auto_prune_threshold"] == 500

    def test_riva_preset(self):
        config = ConfigManager.get_architecture_config("riva")
        assert config["storage"]["backend_type"] == "vector"
        assert config["retrieval"]["semantic_weight"] == 0.8
        assert config["retrieval"]["keyword_weight"] == 0.2
        assert config["encoder"]["encoding_strategies"] == [
            "lesson", "skill", "tool"]

    def test_cerebra_preset(self):
        config = ConfigManager.get_architecture_config("cerebra")
        assert config["storage"]["backend_type"] == "vector"
        assert config["retrieval"]["semantic_weight"] == 0.9
        assert config["retrieval"]["default_top_k"] == 10
        assert config["management"]["auto_prune_threshold"] == 2000
        assert config["encoder"]["enable_abstraction"] is True
        assert config["encoder"]["abstraction_threshold"] == 5

    def test_unknown_architecture(self):
        config = ConfigManager.get_architecture_config("unknown")
        assert config == {}
