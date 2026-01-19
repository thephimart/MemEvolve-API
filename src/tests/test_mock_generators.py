import sys
import pytest

sys.path.insert(0, 'src')

from utils.mock_generators import (
    MemoryUnitGenerator,
    ExperienceGenerator,
    ScenarioGenerator,
    generate_test_units,
    generate_test_experience,
    generate_test_scenario
)


def test_memory_unit_generator_initialization():
    """Test memory unit generator initialization."""
    generator = MemoryUnitGenerator(seed=42)
    assert generator is not None
    assert generator.random is not None


def test_memory_unit_generator_generate_unit():
    """Test generating a single memory unit."""
    generator = MemoryUnitGenerator(seed=42)

    unit = generator.generate_unit()

    assert "id" in unit
    assert "type" in unit
    assert "content" in unit
    assert "tags" in unit
    assert "metadata" in unit
    assert unit["type"] in ["lesson", "skill", "tool", "abstraction"]
    assert len(unit["content"]) > 0
    assert len(unit["tags"]) > 0


def test_memory_unit_generator_generate_unit_with_type():
    """Test generating a unit with specific type."""
    generator = MemoryUnitGenerator(seed=42)

    unit = generator.generate_unit(unit_type="skill")

    assert unit["type"] == "skill"
    # Content should contain skill-related terms
    content_lower = unit["content"].lower()
    assert any(term in content_lower for term in ["proficiency", "practice", "mastering", "developing"])


def test_memory_unit_generator_generate_unit_with_category():
    """Test generating a unit with specific category."""
    generator = MemoryUnitGenerator(seed=42)

    unit = generator.generate_unit(category="programming")

    assert unit["metadata"]["category"] == "programming"
    # Should have programming as a tag
    assert "programming" in unit["tags"]


def test_memory_unit_generator_generate_units():
    """Test generating multiple units."""
    generator = MemoryUnitGenerator(seed=42)

    units = generator.generate_units(count=5)

    assert len(units) == 5
    for unit in units:
        assert "id" in unit
        assert "type" in unit
        assert "content" in unit


def test_memory_unit_generator_generate_units_with_constraints():
    """Test generating units with type and category constraints."""
    generator = MemoryUnitGenerator(seed=42)

    units = generator.generate_units(
        count=10,
        unit_types=["lesson", "skill"],
        categories=["programming", "ai"]
    )

    assert len(units) == 10
    for unit in units:
        assert unit["type"] in ["lesson", "skill"]
        assert unit["metadata"]["category"] in ["programming", "ai"]


def test_experience_generator_initialization():
    """Test experience generator initialization."""
    generator = ExperienceGenerator()
    assert generator is not None
    assert generator.unit_generator is not None


def test_experience_generator_generate_experience():
    """Test generating a single experience."""
    generator = ExperienceGenerator()

    experience = generator.generate_experience()

    assert "id" in experience
    assert "type" in experience
    assert "title" in experience
    assert "description" in experience
    assert "units" in experience
    assert "metadata" in experience
    assert experience["type"] == "experience"
    assert len(experience["units"]) > 0


def test_experience_generator_generate_experience_with_size():
    """Test generating experience with specific size."""
    generator = ExperienceGenerator()

    small_exp = generator.generate_experience(size="small")
    medium_exp = generator.generate_experience(size="medium")
    large_exp = generator.generate_experience(size="large")

    assert 1 <= len(small_exp["units"]) <= 3
    assert 3 <= len(medium_exp["units"]) <= 8
    assert 8 <= len(large_exp["units"]) <= 15


def test_experience_generator_generate_experience_batch():
    """Test generating a batch of experiences."""
    generator = ExperienceGenerator()

    experiences = generator.generate_experience_batch(count=3)

    assert len(experiences) == 3
    for exp in experiences:
        assert "id" in exp
        assert "units" in exp
        assert len(exp["units"]) > 0


def test_test_scenario_generator_initialization():
    """Test scenario generator initialization."""
    generator = ScenarioGenerator()
    assert generator is not None
    assert generator.unit_generator is not None
    assert generator.experience_generator is not None


def test_test_scenario_generator_basic_scenario():
    """Test generating a basic scenario."""
    generator = ScenarioGenerator(seed=42)

    scenario = generator.generate_scenario("basic")

    assert scenario["name"] == "basic_test_scenario"
    assert "units" in scenario
    assert "expected_outcomes" in scenario
    assert len(scenario["units"]) == 10
    assert scenario["expected_outcomes"]["total_units"] == 10


def test_test_scenario_generator_complex_scenario():
    """Test generating a complex scenario."""
    generator = ScenarioGenerator(seed=42)

    scenario = generator.generate_scenario("complex")

    assert scenario["name"] == "complex_test_scenario"
    assert "experiences" in scenario
    assert "units" in scenario
    assert len(scenario["experiences"]) == 3
    assert len(scenario["units"]) > 10


def test_test_scenario_generator_performance_scenario():
    """Test generating a performance scenario."""
    generator = ScenarioGenerator(seed=42)

    scenario = generator.generate_scenario("performance")

    assert scenario["name"] == "performance_test_scenario"
    assert len(scenario["units"]) == 100
    assert scenario["expected_outcomes"]["total_units"] == 100


def test_test_scenario_generator_edge_case_scenario():
    """Test generating an edge case scenario."""
    generator = ScenarioGenerator(seed=42)

    scenario = generator.generate_scenario("edge_case")

    assert scenario["name"] == "edge_case_test_scenario"
    assert len(scenario["units"]) == 4
    assert "edge_cases" in scenario["expected_outcomes"]


def test_convenience_functions():
    """Test convenience functions."""
    # Test generate_test_units
    units = generate_test_units(count=5)
    assert len(units) == 5
    for unit in units:
        assert "id" in unit
        assert "type" in unit

    # Test generate_test_experience
    experience = generate_test_experience()
    assert "id" in experience
    assert "units" in experience

    # Test generate_test_scenario
    scenario = generate_test_scenario("basic")
    assert "name" in scenario
    assert "units" in scenario


def test_memory_unit_generator_deterministic_with_seed():
    """Test that generator produces deterministic results with seed."""
    gen1 = MemoryUnitGenerator(seed=123)
    gen2 = MemoryUnitGenerator(seed=123)

    unit1 = gen1.generate_unit()
    unit2 = gen2.generate_unit()

    # Should produce identical results with same seed
    assert unit1["id"] == unit2["id"]
    assert unit1["type"] == unit2["type"]
    assert unit1["content"] == unit2["content"]


def test_memory_unit_generator_different_results_without_seed():
    """Test that generator produces different results without seed."""
    gen1 = MemoryUnitGenerator()
    gen2 = MemoryUnitGenerator()

    unit1 = gen1.generate_unit()
    unit2 = gen2.generate_unit()

    # Should be very unlikely to be identical without seed
    assert unit1["id"] != unit2["id"]


def test_experience_generator_with_custom_unit_generator():
    """Test experience generator with custom unit generator."""
    custom_generator = MemoryUnitGenerator(seed=999)
    experience_gen = ExperienceGenerator(custom_generator)

    experience = experience_gen.generate_experience()

    assert len(experience["units"]) > 0
    # All units should be deterministic due to seeded generator
    unit_ids = [unit["id"] for unit in experience["units"]]
    assert len(set(unit_ids)) == len(unit_ids)  # All IDs unique
