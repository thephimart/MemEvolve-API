import pytest
from memevolve.evolution.selection import (
    FitnessMetrics,
    EvaluationResult,
    ParetoSelector
)
from memevolve.evolution.genotype import (
    MemoryGenotype,
    GenotypeFactory
)
import sys

# sys.path.insert(0, 'src')  # No longer needed with package structure


def test_fitness_metrics_creation():
    """Test fitness metrics dataclass."""
    metrics = FitnessMetrics(
        performance=0.8,
        cost=0.3,
        retrieval_accuracy=0.9,
        storage_efficiency=0.7,
        response_time=1.2,
        memory_size_mb=50.0
    )

    assert metrics.performance == 0.8
    assert metrics.cost == 0.3
    assert metrics.retrieval_accuracy == 0.9
    assert metrics.storage_efficiency == 0.7

    score = metrics.calculate_fitness_score()

    expected_score = (
        0.4 * 0.8 + (-0.3) * 0.3 +
        0.2 * 0.9 + 0.1 * 0.7
    )

    assert abs(score - expected_score) < 0.01


def test_fitness_metrics_default_weights():
    """Test fitness score calculation with default weights."""
    metrics = FitnessMetrics(
        performance=0.8,
        cost=0.2,
        retrieval_accuracy=0.9,
        storage_efficiency=0.5
    )

    score = metrics.calculate_fitness_score()

    expected_score = (
        0.4 * 0.8 + (-0.3) * 0.2 +
        0.2 * 0.9 + 0.1 * 0.5
    )

    assert abs(score - expected_score) < 0.01


def test_evaluation_result_creation():
    """Test evaluation result dataclass."""
    genotype = MemoryGenotype()
    metrics = FitnessMetrics(performance=0.8, cost=0.3)
    fitness_score = metrics.calculate_fitness_score()

    result = EvaluationResult(
        genotype=genotype,
        fitness_metrics=metrics,
        fitness_score=fitness_score
    )

    assert isinstance(result.genotype, MemoryGenotype)
    assert result.fitness_score == fitness_score
    assert result.is_pareto_optimal is False
    assert result.dominated_by is None
    assert result.dominates_list == []


def test_evaluation_result_domination():
    """Test Pareto domination logic."""
    genotype1 = MemoryGenotype()
    genotype2 = MemoryGenotype()

    metrics1 = FitnessMetrics(performance=0.8, cost=0.3, retrieval_accuracy=0.8,
                              storage_efficiency=0.8, response_time=1.0,
                              memory_size_mb=10.0)
    metrics2 = FitnessMetrics(performance=0.6, cost=0.4, retrieval_accuracy=0.6,
                              storage_efficiency=0.6, response_time=1.5,
                              memory_size_mb=15.0)

    result1 = EvaluationResult(
        genotype=genotype1,
        fitness_metrics=metrics1,
        fitness_score=metrics1.calculate_fitness_score()
    )

    result2 = EvaluationResult(
        genotype=genotype2,
        fitness_metrics=metrics2,
        fitness_score=metrics2.calculate_fitness_score()
    )

    assert result2.dominates(result1) is False
    assert result1.dominates(result2) is True


def test_pareto_selector_initialization():
    """Test Pareto selector initialization."""
    selector = ParetoSelector()

    assert selector.performance_weight == 0.4
    assert selector.cost_weight == 0.3


def test_pareto_selector_custom_weights():
    """Test Pareto selector with custom weights."""
    selector = ParetoSelector(
        performance_weight=0.6,
        cost_weight=0.1
    )

    assert selector.performance_weight == 0.6
    assert selector.cost_weight == 0.1


def test_pareto_selector_backend_costs():
    """Test cost calculation for all backend types."""
    backends = ["json", "vector", "graph"]
    selector = ParetoSelector()

    costs = []
    for backend in backends:
        genotype = MemoryGenotype()
        genotype.store.backend_type = backend
        cost = selector._get_backend_cost(backend)
        costs.append(cost)

    assert costs == [1.0, 1.5, 2.0]


def test_pareto_selector_strategy_costs():
    """Test cost calculation for all strategy types."""
    strategies = ["keyword", "semantic", "hybrid"]
    selector = ParetoSelector()

    costs = []
    for strategy in strategies:
        genotype = MemoryGenotype()
        genotype.retrieve.strategy_type = strategy
        cost = selector._get_strategy_cost(strategy)
        costs.append(cost)

    assert costs == [1.0, 1.3, 1.5]


def test_pareto_selector_all_backends_strategies():
    """Test cost calculation for all backend x strategy combinations."""
    backends = ["json", "vector"]
    strategies = ["keyword", "semantic"]
    selector = ParetoSelector()

    count = 0
    for backend in backends:
        for strategy in strategies:
            genotype = MemoryGenotype()
            genotype.store.backend_type = backend
            genotype.retrieve.strategy_type = strategy
            cost = selector._get_backend_cost(backend)
            strategy_cost = selector._get_strategy_cost(strategy)
            total_cost = 1.0 * cost * strategy_cost
            assert total_cost > 0
            count += 1

    assert count == 4


def test_pareto_selector_evaluate():
    """Test Pareto selector evaluation."""
    genotypes = [
        GenotypeFactory.create_baseline_genotype(),
        GenotypeFactory.create_lightweight_genotype(),
        GenotypeFactory.create_agentkb_genotype()
    ]

    performance_data = {
        genotypes[0].get_genome_id(): 0.7,
        genotypes[1].get_genome_id(): 0.85,
        genotypes[2].get_genome_id(): 0.8
    }

    cost_data = {
        genotypes[0].get_genome_id(): 1.0,
        genotypes[1].get_genome_id(): 0.8,
        genotypes[2].get_genome_id(): 1.2
    }

    selector = ParetoSelector()
    results = selector.evaluate(genotypes, performance_data, cost_data)

    assert len(results) == 3
    assert all(isinstance(r, EvaluationResult) for r in results)


def test_pareto_selector_pareto_front():
    """Test Pareto front selection."""
    genotypes = [
        GenotypeFactory.create_baseline_genotype(),
        GenotypeFactory.create_baseline_genotype(),
        GenotypeFactory.create_lightweight_genotype()
    ]

    performance_data = {
        genotypes[0].get_genome_id(): 0.9,
        genotypes[1].get_genome_id(): 0.85,
        genotypes[2].get_genome_id(): 0.8
    }

    selector = ParetoSelector()
    results = selector.evaluate(genotypes, performance_data)
    pareto_front = selector.select_pareto_front(results)

    assert len(pareto_front) == 2
    assert all(r is not None for r in pareto_front)


def test_pareto_selector_select_best():
    """Test Pareto selector best selection."""
    genotypes = [
        GenotypeFactory.create_baseline_genotype(),
        GenotypeFactory.create_lightweight_genotype(),
        GenotypeFactory.create_agentkb_genotype()
    ]

    performance_data = {
        genotypes[0].get_genome_id(): 0.7,
        genotypes[1].get_genome_id(): 0.9,
        genotypes[2].get_genome_id(): 0.6
    }

    selector = ParetoSelector()
    results = selector.evaluate(genotypes, performance_data)

    best = selector.select_best(results)

    assert best.fitness_score == max(r.fitness_score for r in results)
    assert best.is_pareto_optimal is True


def test_pareto_selector_select_top_n():
    """Test Pareto selector top N selection."""
    genotypes = [
        GenotypeFactory.create_baseline_genotype(),
        GenotypeFactory.create_lightweight_genotype(),
        GenotypeFactory.create_agentkb_genotype(),
        GenotypeFactory.create_riva_genotype()
    ]

    performance_data = {
        genotypes[0].get_genome_id(): 0.6,
        genotypes[1].get_genome_id(): 0.9,
        genotypes[2].get_genome_id(): 0.85,
        genotypes[3].get_genome_id(): 0.7
    }

    selector = ParetoSelector()
    results = selector.evaluate(genotypes, performance_data)
    top_n = selector.select_top_n(results, n=2)

    assert len(top_n) == 2
    assert top_n[0].fitness_score >= top_n[1].fitness_score


def test_pareto_selector_empty_results():
    """Test Pareto selector with empty results."""
    selector = ParetoSelector()

    pareto_front = selector.select_pareto_front([])

    assert pareto_front == []

    try:
        selector.select_best([])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_pareto_selector_calculate_cost():
    """Test cost calculation for different genotypes."""
    genotype1 = MemoryGenotype(
        store=MemoryGenotype().store
    )
    genotype2 = MemoryGenotype(
        store=MemoryGenotype().store
    )

    selector = ParetoSelector()

    cost1 = selector._calculate_cost(genotype1, 1.0)
    cost2 = selector._calculate_cost(genotype2, 1.5)

    assert cost1 < cost2


def test_pareto_selector_storage_efficiency():
    """Test storage efficiency calculation."""
    genotype1 = MemoryGenotype(
        store=MemoryGenotype().store
    )
    genotype2 = MemoryGenotype(
        store=MemoryGenotype().store
    )

    genotype1.manage.deduplicate_enabled = True
    genotype1.store.enable_persistence = True

    genotype2.manage.deduplicate_enabled = False
    genotype2.store.enable_persistence = False

    selector = ParetoSelector()

    eff1 = selector._calculate_storage_efficiency(genotype1)
    eff2 = selector._calculate_storage_efficiency(genotype2)

    assert eff1 > eff2
