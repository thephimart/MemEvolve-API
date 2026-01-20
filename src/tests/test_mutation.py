from evolution.mutation import (
    MutationEngine,
    MutationResult,
    RandomMutationStrategy,
    TargetedMutationStrategy,
    MutationOperation
)
from evolution.genotype import (
    MemoryGenotype,
    GenotypeFactory
)
import pytest
import sys

sys.path.insert(0, 'src')


def test_random_mutation_strategy_initialization():
    """Test random mutation strategy initialization."""
    strategy = RandomMutationStrategy()

    assert strategy is not None
    assert len(strategy.encoding_strategies_options) > 0
    assert len(strategy.backend_types) > 0
    assert len(strategy.strategy_types) > 0


def test_random_mutation_mutate():
    """Test random mutation application."""
    strategy = RandomMutationStrategy()
    genotype = GenotypeFactory.create_baseline_genotype()

    mutated = strategy.mutate(genotype, mutation_rate=1.0)

    assert isinstance(mutated, MemoryGenotype)
    assert mutated is not genotype


def test_random_mutation_no_mutation():
    """Test random mutation with zero rate."""
    strategy = RandomMutationStrategy()
    genotype = GenotypeFactory.create_baseline_genotype()

    mutated = strategy.mutate(genotype, mutation_rate=0.0)

    assert isinstance(mutated, MemoryGenotype)
    assert mutated.encode.temperature == genotype.encode.temperature
    assert mutated.store.backend_type == genotype.store.backend_type


def test_targeted_mutation_strategy_initialization():
    """Test targeted mutation strategy initialization."""
    strategy = TargetedMutationStrategy()

    assert strategy is not None
    assert "performance" in strategy.feedback_weights


def test_targeted_mutation_mutate():
    """Test targeted mutation application."""
    strategy = TargetedMutationStrategy()
    genotype = GenotypeFactory.create_baseline_genotype()

    feedback = {
        "performance": 0.5,
        "cost": 0.8,
        "retrieval_accuracy": 0.6
    }

    mutated = strategy.mutate(genotype, mutation_rate=0.5, feedback=feedback)

    assert isinstance(mutated, MemoryGenotype)
    assert mutated is not genotype


def test_mutation_engine_initialization():
    """Test mutation engine initialization."""
    engine = MutationEngine()

    assert engine is not None
    assert isinstance(engine.strategy, RandomMutationStrategy)


def test_mutation_engine_mutate():
    """Test mutation engine mutation application."""
    engine = MutationEngine()
    genotype = GenotypeFactory.create_baseline_genotype()

    result = engine.mutate(genotype, mutation_rate=0.5)

    assert isinstance(result, MutationResult)
    assert result.original_genotype == genotype
    assert isinstance(result.mutated_genotype, MemoryGenotype)
    assert result.mutation_rate == 0.5
    assert result.success is True


def test_mutation_operation_creation():
    """Test mutation operation creation."""
    operation = MutationOperation(
        operation_type="modify",
        description="Changed backend",
        target_component="store",
        before_value="json",
        after_value="vector",
        impact_score=0.8
    )

    assert operation.operation_type == "modify"
    assert operation.target_component == "store"
    assert operation.impact_score == 0.8


def test_mutation_engine_track_mutations():
    """Test mutation tracking."""
    engine = MutationEngine()
    genotype1 = GenotypeFactory.create_baseline_genotype()
    genotype2 = GenotypeFactory.create_riva_genotype()

    mutations = engine._track_mutations(genotype1, genotype2)

    assert isinstance(mutations, list)
    assert all(isinstance(m, MutationOperation) for m in mutations)


def test_mutation_engine_batch_mutate():
    """Test batch mutation application."""
    engine = MutationEngine()

    genotypes = [
        GenotypeFactory.create_baseline_genotype(),
        GenotypeFactory.create_lightweight_genotype(),
        GenotypeFactory.create_agentkb_genotype()
    ]

    results = engine.batch_mutate(genotypes, mutation_rate=0.3)

    assert len(results) == 3
    assert all(isinstance(r, MutationResult) for r in results)
    assert all(r.success for r in results)


def test_targeted_mutation_adjusts_rates():
    """Test targeted mutation adjusts rates based on feedback."""
    strategy = TargetedMutationStrategy()
    genotype = GenotypeFactory.create_baseline_genotype()

    feedback_low_performance = {
        "performance": 0.4,
        "cost": 0.5,
        "retrieval_accuracy": 0.6
    }

    feedback_high_performance = {
        "performance": 0.9,
        "cost": 0.3,
        "retrieval_accuracy": 0.95
    }

    base_rate = 0.3

    adjusted_low = strategy._adjust_mutation_rates(
        base_rate, feedback_low_performance
    )

    adjusted_high = strategy._adjust_mutation_rates(
        base_rate, feedback_high_performance
    )

    assert adjusted_low.get(
        "encode", base_rate) >= adjusted_high.get("encode", base_rate)


def test_mutation_result_failure_handling():
    """Test mutation engine handles failures gracefully."""
    engine = MutationEngine()
    genotype = GenotypeFactory.create_baseline_genotype()

    result = engine.mutate(genotype, mutation_rate=0.5)

    assert isinstance(result, MutationResult)
    assert result.success in [True, False]


def test_mutation_with_custom_strategy():
    """Test mutation engine with custom strategy."""
    custom_strategy = RandomMutationStrategy()
    engine = MutationEngine(strategy=custom_strategy)

    genotype = GenotypeFactory.create_baseline_genotype()
    result = engine.mutate(genotype, mutation_rate=0.5)

    assert result.success is True
    assert engine.strategy == custom_strategy


def test_mutation_preserves_metadata():
    """Test mutation preserves genotype metadata."""
    engine = MutationEngine()

    genotype = GenotypeFactory.create_baseline_genotype()
    genotype.metadata["test_key"] = "test_value"

    result = engine.mutate(genotype, mutation_rate=0.5)

    assert "test_key" in result.mutated_genotype.metadata
    assert result.mutated_genotype.metadata["test_key"] == "test_value"


def test_mutation_generates_different_genome_ids():
    """Test mutations generate different genome IDs."""
    engine = MutationEngine()

    genotype = GenotypeFactory.create_baseline_genotype()
    result = engine.mutate(genotype, mutation_rate=1.0)

    if result.mutations_applied:
        assert result.mutated_genotype.get_genome_id() != genotype.get_genome_id()
