"""Tests for configuration management system."""

import sys
sys.path.insert(0, 'src')

from pathlib import Path
import tempfile
import json
import pytest

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
import sys
sys.path.insert(0, 'src')


class TestLLMConfig:
    """Test LLM configuration."""

    def test_default_values(self):
        config = LLMConfig()
        assert config.base_url == "http://localhost:11434/v1"
        assert config.api_key == "dummy-key"
        assert config.model is None
        assert config.timeout == 120
        assert config.max_retries == 3

    def test_custom_values(self):
        config = LLMConfig(
            base_url="http://custom:8080/v1",
            api_key="test-key",
            model="test-model",
            timeout=60,
            max_retries=5
        )
        assert config.base_url == "http://custom:8080/v1"
        assert config.api_key == "test-key"
        assert config.model == "test-model"
        assert config.timeout == 60
        assert config.max_retries == 5


class TestStorageConfig:
    """Test storage configuration."""

    def test_default_values(self):
        config = StorageConfig()
        assert config.backend_type == "json"
        assert config.path == "./data/memory"
        assert config.vector_dim == 768
        assert config.index_type == "flat"

    def test_custom_values(self):
        config = StorageConfig(
            backend_type="vector",
            path="/custom/path",
            vector_dim=1024,
            index_type="hnsw"
        )
        assert config.backend_type == "vector"
        assert config.path == "/custom/path"
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


class TestLoggingConfig:
    """Test logging configuration."""

    def test_default_values(self):
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert config.log_file is None
        assert config.enable_operation_log is True
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

    def test_load_from_yaml_file(self):
        config_data = {
            "llm": {"base_url": "http://test:8080/v1", "model": "test-model"},
            "retrieval": {"default_top_k": 10}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            manager = ConfigManager(temp_path)
            assert manager.config.llm.base_url == "http://test:8080/v1"
            assert manager.config.llm.model == "test-model"
            assert manager.config.retrieval.default_top_k == 10
        finally:
            Path(temp_path).unlink()

    def test_load_from_json_file(self):
        config_data = {
            "llm": {"base_url": "http://json:8080/v1", "model": "json-model"},
            "storage": {"backend_type": "vector"}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            manager = ConfigManager(temp_path)
            assert manager.config.llm.base_url == "http://json:8080/v1"
            assert manager.config.llm.model == "json-model"
            assert manager.config.storage.backend_type == "vector"
        finally:
            Path(temp_path).unlink()

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
        manager = ConfigManager()
        assert manager.get("llm.base_url") == "http://localhost:11434/v1"
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

    def test_save_to_yaml(self):
        manager = ConfigManager()
        manager.update(**{"llm.base_url": "http://saved:8080/v1"})

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            manager.save_to_file(temp_path)
            assert Path(temp_path).exists()

            loaded_manager = ConfigManager(temp_path)
            assert loaded_manager.config.llm.base_url == "http://saved:8080/v1"
        finally:
            Path(temp_path).unlink()

    def test_save_to_json(self):
        manager = ConfigManager()
        manager.update(**{"storage.backend_type": "vector"})

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            manager.save_to_file(temp_path)
            assert Path(temp_path).exists()

            loaded_manager = ConfigManager(temp_path)
            assert loaded_manager.config.storage.backend_type == "vector"
        finally:
            Path(temp_path).unlink()


class TestLoadConfig:
    """Test load_config function."""

    def test_load_default_config(self):
        config = load_config()
        assert isinstance(config, MemEvolveConfig)
        assert isinstance(config.llm, LLMConfig)

    def test_load_from_file(self):
        config_data = {"llm": {"base_url": "http://loaded:8080/v1"}}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.llm.base_url == "http://loaded:8080/v1"
        finally:
            Path(temp_path).unlink()


class TestSaveConfig:
    """Test save_config function."""

    def test_save_config(self):
        config = MemEvolveConfig()
        config.llm.base_url = "http://save:8080/v1"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            save_config(config, temp_path)
            assert Path(temp_path).exists()

            loaded_config = load_config(temp_path)
            assert loaded_config.llm.base_url == "http://save:8080/v1"
        finally:
            Path(temp_path).unlink()


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
