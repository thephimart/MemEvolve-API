"""
Test fixtures for MemEvolve testing.

This module provides reusable test fixtures and helper functions for testing
memory systems and components.
"""

from utils.mock_generators import (
    MemoryUnitGenerator,
    ExperienceGenerator,
    generate_test_scenario
)
from utils.config import MemEvolveConfig
from components.manage import SimpleManagementStrategy, MemoryManager
from components.retrieve import SemanticRetrievalStrategy
from components.store import JSONFileStore
from components.encode import ExperienceEncoder
from memory_system import MemorySystem
import numpy as np
import pytest
from pathlib import Path
import tempfile
import sys
import warnings
# Suppress FAISS SWIG deprecation warnings globally for all tests
warnings.filterwarnings(
    "ignore", message=".*SwigPyPacked.*", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore", message=".*SwigPyObject.*", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore", message=".*swigvarlink.*", category=DeprecationWarning)


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_memory_units():
    """Generate a set of sample memory units for testing."""
    # Use real encoding by default, with fallback to mock
    generator = MemoryUnitGenerator(seed=42, use_real_encoding=True)
    return generator.generate_units(count=10)


@pytest.fixture
def large_memory_units():
    """Generate a large set of memory units for performance testing."""
    generator = MemoryUnitGenerator(seed=42)
    return generator.generate_units(count=100)


@pytest.fixture
def diverse_memory_units():
    """Generate memory units with diverse types and categories."""
    generator = MemoryUnitGenerator(seed=42)
    return generator.generate_units(
        count=20,
        unit_types=["lesson", "skill", "tool", "abstraction"],
        categories=["programming", "ai", "data", "engineering", "science"]
    )


@pytest.fixture
def sample_experience():
    """Generate a sample experience for testing."""
    generator = ExperienceGenerator(seed=42)
    return generator.generate_experience(size="medium")


@pytest.fixture
def test_scenario():
    """Generate a basic test scenario."""
    return generate_test_scenario("basic")


@pytest.fixture
def basic_memory_config(temp_dir):
    """Create a basic memory system configuration."""
    config = MemEvolveConfig()
    config.storage.path = str(temp_dir / "memory.json")
    config.storage.backend_type = "json"
    config.encoder.encoding_strategies = ["lesson", "skill"]
    config.retrieval.strategy_type = "semantic"
    config.retrieval.default_top_k = 5
    return config


@pytest.fixture
def memory_system_config(temp_dir):
    """Create a complete memory system configuration."""
    config = MemEvolveConfig()
    config.storage.path = str(temp_dir / "test_memory.json")
    config.storage.backend_type = "json"
    config.encoder.encoding_strategies = [
        "lesson", "skill", "tool", "abstraction"]
    config.retrieval.strategy_type = "semantic"
    config.retrieval.default_top_k = 10
    config.retrieval.semantic_weight = 0.7
    config.management.enable_auto_management = True
    config.management.auto_prune_threshold = 1000
    return config


@pytest.fixture
def json_store(temp_dir):
    """Create a JSON file store for testing."""
    store_path = temp_dir / "test_store.json"
    return JSONFileStore(str(store_path))


@pytest.fixture
def populated_json_store(json_store, sample_memory_units):
    """Create a JSON store populated with sample data."""
    for unit in sample_memory_units:
        json_store.store(unit)
    return json_store


class MockExperienceEncoder:
    """Experience encoder for testing - uses real encoding when available, mock otherwise."""

    def __init__(self):
        self.metrics_collector = None
        self.real_encoder = None
        self.real_embedding_fn = None

        # Try to initialize real components
        try:
            from components.encode import ExperienceEncoder
            from utils.embeddings import create_embedding_function

            self.real_encoder = ExperienceEncoder()
            self.real_encoder.initialize_llm()
            self.real_embedding_fn = create_embedding_function("openai")
        except Exception:
            # Will fall back to mock encoding
            pass

    def initialize_llm(self):
        """Initialize LLM if real encoder is available."""
        if self.real_encoder:
            self.real_encoder.initialize_llm()

    def encode_experience(self, experience):
        """Encode experience using real LLM when available, mock otherwise."""
        if self.real_encoder and self.real_embedding_fn:
            try:
                # Use real encoding
                unit = self.real_encoder.encode_experience(experience)
                # Add real embedding
                embedding = self.real_embedding_fn(unit.get("content", ""))
                unit["embedding"] = embedding.tolist() if hasattr(
                    embedding, 'tolist') else embedding
                unit["metadata"]["encoding_method"] = "real"
                return unit
            except Exception:
                # Fall back to mock
                pass

        # Mock encoding fallback
        return {
            "id": experience.get("id", f"encoded_{hash(str(experience))}"),
            "type": experience.get("type", "lesson"),
            "content": experience.get("content", ""),
            "tags": experience.get("tags", []),
            "metadata": {
                **experience.get("metadata", {}),
                "encoded_at": "2024-01-01T00:00:00Z",
                "encoding_method": "mock"
            },
            "embedding": [0.1, 0.2, 0.3] * 10  # Mock embedding
        }

    def get_metrics(self):
        """Mock metrics."""
        return {
            "total_encodings": 0,
            "successful_encodings": 0,
            "failed_encodings": 0,
            "success_rate": 0.0,
            "average_encoding_time": 0.0
        }


@pytest.fixture
def experience_encoder():
    """Create a mock experience encoder for testing."""
    return MockExperienceEncoder()


@pytest.fixture
def semantic_retriever(json_store):
    """Create a semantic retrieval strategy."""
    def mock_embedding_function(text: str) -> np.ndarray:
        """Mock embedding function for testing."""
        # Return a fixed-size random vector for testing
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(384).astype(np.float32)

    return SemanticRetrievalStrategy(
        embedding_function=mock_embedding_function,
        similarity_threshold=0.7
    )


@pytest.fixture
def simple_memory_manager(json_store):
    """Create a simple memory manager."""
    strategy = SimpleManagementStrategy()
    return MemoryManager(storage_backend=json_store, management_strategy=strategy)


@pytest.fixture
def basic_memory_system(memory_system_config, experience_encoder):
    """Create a basic memory system instance."""
    return MemorySystem(memory_system_config, encoder=experience_encoder)


@pytest.fixture
def populated_memory_system(basic_memory_system, sample_memory_units):
    """Create a memory system populated with sample data."""
    for unit in sample_memory_units:
        basic_memory_system.add_experience(unit)
    return basic_memory_system


@pytest.fixture
def large_memory_system(memory_system_config, large_memory_units):
    """Create a memory system with a large dataset for performance testing."""
    system = MemorySystem(memory_system_config)
    for unit in large_memory_units:
        system.add_experience(unit)
    return system


@pytest.fixture
def programming_focused_units():
    """Generate units focused on programming topics."""
    generator = MemoryUnitGenerator(seed=42)
    return generator.generate_units(
        count=15,
        categories=["programming"],
        unit_types=["lesson", "skill", "tool"]
    )


@pytest.fixture
def ai_focused_units():
    """Generate units focused on AI topics."""
    generator = MemoryUnitGenerator(seed=42)
    return generator.generate_units(
        count=12,
        categories=["ai"],
        unit_types=["lesson", "skill", "abstraction"]
    )


@pytest.fixture
def mixed_category_units():
    """Generate units from multiple categories."""
    generator = MemoryUnitGenerator(seed=42)
    return generator.generate_units(
        count=25,
        categories=["programming", "ai", "data"],
        unit_types=["lesson", "skill", "tool"]
    )


@pytest.fixture
def edge_case_units():
    """Generate units with edge cases for testing."""
    units = []

    # Normal unit
    generator = MemoryUnitGenerator(seed=42)
    units.append(generator.generate_unit())

    # Unit with very long content
    long_content_unit = generator.generate_unit()
    long_content_unit["content"] = "Very long content. " * 500  # ~10KB content
    units.append(long_content_unit)

    # Unit with many tags
    many_tags_unit = generator.generate_unit()
    many_tags_unit["tags"] = [f"tag_{i}" for i in range(20)]
    units.append(many_tags_unit)

    # Unit with special characters
    special_chars_unit = generator.generate_unit()
    special_chars_unit["content"] = "Content with émojis 🚀 and spëcial chärs"
    special_chars_unit["tags"] = ["tëst", "spëcial"]
    units.append(special_chars_unit)

    # Unit with empty content
    empty_content_unit = generator.generate_unit()
    empty_content_unit["content"] = ""
    units.append(empty_content_unit)

    # Unit with minimal metadata
    minimal_unit = generator.generate_unit()
    minimal_unit["metadata"] = {"created_at": "2024-01-01T00:00:00Z"}
    units.append(minimal_unit)

    return units


@pytest.fixture
def performance_test_units():
    """Generate units optimized for performance testing."""
    # Create units with consistent structure for fair comparison
    generator = MemoryUnitGenerator(seed=12345)

    units = []
    for i in range(50):
        unit = generator.generate_unit(
            unit_type="lesson",
            category="programming",
            custom_fields={
                "performance_id": i,
                "content_length": 100 + (i * 10)  # Increasing content length
            }
        )
        # Override content to have predictable length
        unit["content"] = f"Performance test content {i}. " * (10 + i)
        units.append(unit)

    return units


@pytest.fixture
def retrieval_test_setup(json_store, diverse_memory_units):
    """Set up a test environment for retrieval testing."""
    # Populate store
    for unit in diverse_memory_units:
        json_store.store(unit)

    # Create retriever
    def mock_embedding_function(text: str) -> np.ndarray:
        """Mock embedding function for testing."""
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(384).astype(np.float32)

    retriever = SemanticRetrievalStrategy(
        embedding_function=mock_embedding_function,
        similarity_threshold=0.5
    )

    return {
        "store": json_store,
        "retriever": retriever,
        "units": diverse_memory_units,
        "unit_count": len(diverse_memory_units)
    }


@pytest.fixture
def encoding_test_setup(experience_encoder, sample_memory_units):
    """Set up a test environment for encoding testing."""
    return {
        "encoder": experience_encoder,
        "test_units": sample_memory_units,
        "expected_strategies": ["lesson", "skill"]
    }


@pytest.fixture
def management_test_setup(simple_memory_manager, json_store, sample_memory_units):
    """Set up a test environment for memory management testing."""
    # Populate store first
    for unit in sample_memory_units:
        json_store.store(unit)

    return {
        "manager": simple_memory_manager,
        "store": json_store,
        "initial_units": sample_memory_units,
        "initial_count": len(sample_memory_units)
    }


@pytest.fixture
def integration_test_setup(memory_system_config, diverse_memory_units):
    """Set up a complete integration test environment."""
    system = MemorySystem(memory_system_config)

    # Add diverse units
    for unit in diverse_memory_units:
        system.add_experience(unit)

    return {
        "system": system,
        "test_units": diverse_memory_units,
        "config": memory_system_config
    }


# Utility functions for tests
def assert_memory_unit_structure(unit):
    """Assert that a memory unit has the expected structure."""
    required_fields = ["id", "type", "content", "tags", "metadata"]
    for field in required_fields:
        assert field in unit, f"Missing required field: {field}"

    assert isinstance(unit["tags"], list), "Tags should be a list"
    assert isinstance(unit["metadata"], dict), "Metadata should be a dict"
    assert "created_at" in unit["metadata"], "Metadata should have created_at"


def assert_memory_units_unique(units):
    """Assert that memory units have unique IDs."""
    ids = [unit["id"] for unit in units]
    assert len(ids) == len(set(ids)), "Unit IDs are not unique"


def assert_memory_system_health(system):
    """Assert that a memory system is in a healthy state."""
    health = system.get_health_metrics()
    assert health is not None, "Health metrics should be available"
    assert health.total_units >= 0, "Total units should be non-negative"
    assert health.total_size_bytes >= 0, "Total size should be non-negative"


def count_units_by_type(units):
    """Count units by type."""
    type_counts = {}
    for unit in units:
        unit_type = unit.get("type", "unknown")
        type_counts[unit_type] = type_counts.get(unit_type, 0) + 1
    return type_counts


def count_units_by_category(units):
    """Count units by category."""
    category_counts = {}
    for unit in units:
        category = unit.get("metadata", {}).get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    return category_counts
