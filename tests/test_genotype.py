import pytest
from memevolve.evolution.genotype import (
    MemoryGenotype,
    EncodeConfig,
    StoreConfig,
    RetrieveConfig,
    ManageConfig,
    GenotypeFactory
)
import sys

# sys.path.insert(0, 'src')  # No longer needed with package structure


def test_genotype_creation():
    """Test genotype creation."""
    genotype = MemoryGenotype()

    assert isinstance(genotype.encode, EncodeConfig)
    assert isinstance(genotype.store, StoreConfig)
    assert isinstance(genotype.retrieve, RetrieveConfig)
    assert isinstance(genotype.manage, ManageConfig)
    assert isinstance(genotype.metadata, dict)


def test_genome_id():
    """Test genome ID generation."""
    genotype1 = MemoryGenotype()
    genotype2 = MemoryGenotype()

    id1 = genotype1.get_genome_id()
    id2 = genotype2.get_genome_id()

    assert len(id1) == 8
    assert len(id2) == 8
    assert id1 == id2

    genotype3 = MemoryGenotype(encode=EncodeConfig(temperature=0.5))
    id3 = genotype3.get_genome_id()
    assert id3 != id1


def test_factory_baseline():
    """Test baseline genotype factory."""
    genotype = GenotypeFactory.create_baseline_genotype()

    assert isinstance(genotype, MemoryGenotype)
    assert "lesson" in genotype.encode.encoding_strategies
    assert genotype.store.backend_type == "json"


def test_factory_agentkb():
    """Test AgentKB genotype factory."""
    genotype = GenotypeFactory.create_agentkb_genotype()

    assert genotype.metadata["architecture"] == "agentkb"
    assert "lesson" in genotype.encode.encoding_strategies
    assert not genotype.encode.enable_abstractions


def test_factory_lightweight():
    """Test Lightweight genotype factory."""
    genotype = GenotypeFactory.create_lightweight_genotype()

    assert genotype.metadata["architecture"] == "lightweight"
    assert genotype.encode.batch_size == 20
    assert genotype.manage.auto_prune_threshold == 500


def test_factory_riva():
    """Test Riva genotype factory."""
    genotype = GenotypeFactory.create_riva_genotype()

    assert genotype.metadata["architecture"] == "riva"
    assert genotype.store.backend_type == "vector"
    assert genotype.retrieve.strategy_type == "hybrid"


def test_factory_cerebra():
    """Test Cerebra genotype factory."""
    genotype = GenotypeFactory.create_cerebra_genotype()

    assert genotype.metadata["architecture"] == "cerebra"
    assert "tool" in genotype.encode.encoding_strategies
    assert genotype.store.backend_type == "vector"


def test_genotype_serialization():
    """Test genotype JSON serialization/deserialization."""
    original = MemoryGenotype(
        encode=EncodeConfig(temperature=0.5),
        store=StoreConfig(backend_type="json")
    )

    json_str = original.to_json()

    restored = MemoryGenotype.from_json(json_str)

    assert restored.encode.temperature == 0.5
    assert restored.store.backend_type == "json"


def test_genotype_mutation():
    """Test genotype mutation."""
    original = GenotypeFactory.create_baseline_genotype()

    mutated = GenotypeFactory.mutate_genotype(original, mutate_probability=0.0)

    assert mutated.encode.temperature == original.encode.temperature
    assert mutated is not original
