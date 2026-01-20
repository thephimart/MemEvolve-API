"""
Data generators for MemEvolve testing.

This module provides utilities for generating test data. By default it uses
real LLM encoding and embeddings when available, falling back to mock data.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
import random
import uuid

from .logging import get_logger


class MemoryUnitGenerator:
    """Generator for memory unit data."""

    def __init__(self, seed: Optional[int] = None, use_real_encoding: bool = True):
        self.logger = get_logger("memory_unit_generator")
        self.random = random.Random(seed)
        self.use_real_encoding = use_real_encoding

        # Try to initialize real components if requested
        self.real_generator = None
        if use_real_encoding:
            try:
                from .real_data_generator import RealMemoryUnitGenerator
                self.real_generator = RealMemoryUnitGenerator(seed=seed)
                self.logger.info(
                    "Using real encoding for test data generation")
            except Exception as e:
                self.logger.warning(
                    f"Could not initialize real generator: {e}, falling back to mock data")
                self.use_real_encoding = False

        # Predefined content templates
        self.content_templates = {
            "lesson": [
                "Understanding {topic} is fundamental to {field}. Key concepts include {concepts}.",
                "The principles of {topic} involve {processes}. This enables {benefits}.",
                "{topic} provides a framework for {applications} in {field}.",
                "Learning {topic} requires understanding {prerequisites} and leads to {outcomes}."
            ],
            "skill": [
                "Developing proficiency in {topic} involves practicing {techniques} and mastering {tools}.",
                "The skill of {topic} encompasses {aspects} and requires {experience}.",
                "Mastering {topic} enables {capabilities} through {methods}.",
                "{topic} skills include {competencies} and lead to {achievements}."
            ],
            "tool": [
                "{tool_name} is a {tool_type} that facilitates {functions} in {domain}.",
                "Using {tool_name} enables {efficiencies} when working with {artifacts}.",
                "{tool_name} provides {features} for {use_cases} in {field}.",
                "The {tool_name} tool supports {workflows} and integrates with {ecosystem}."
            ],
            "abstraction": [
                "{concept} represents an abstraction of {concrete_examples} that enables {generalizations}.",
                "The abstraction {concept} captures {patterns} across {domains}.",
                "{concept} provides a unified view of {variations} through {principles}.",
                "Abstracting {concrete_examples} leads to {concept} which enables {applications}."
            ]
        }

        # Topic categories
        self.topic_categories = {
            "programming": ["Python", "JavaScript", "algorithms", "data structures", "OOP", "functional programming"],
            "ai": ["machine learning", "neural networks", "natural language processing", "computer vision"],
            "data": ["statistics", "databases", "data analysis", "visualization", "big data"],
            "engineering": ["software architecture", "testing", "DevOps", "security", "performance"],
            "science": ["mathematics", "physics", "biology", "chemistry", "research methods"]
        }

    def generate_unit(
        self,
        unit_type: Optional[str] = None,
        category: Optional[str] = None,
        custom_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a single memory unit.

        Args:
            unit_type: Type of unit ("lesson", "skill", "tool", "abstraction")
            category: Content category ("programming", "ai", etc.)
            custom_fields: Additional fields to include

        Returns:
            Generated memory unit
        """
        # Try real encoding first if available
        if self.use_real_encoding and self.real_generator:
            try:
                unit = self.real_generator.generate_unit(
                    category=category, custom_fields=custom_fields)
                # Override type if specified
                if unit_type:
                    unit["type"] = unit_type
                return unit
            except Exception as e:
                self.logger.warning(
                    f"Real encoding failed: {e}, using mock generation")

        # Fallback to mock generation
        # Select random type and category if not specified
        if unit_type is None:
            unit_type = self.random.choice(list(self.content_templates.keys()))

        if category is None:
            category = self.random.choice(list(self.topic_categories.keys()))

        # Generate basic unit
        random_id = self.random.randint(10000000, 99999999)
        unit_id = f"{unit_type}_{random_id:08x}"

        unit = {
            "id": unit_id,
            "type": unit_type,
            "content": self._generate_content(unit_type, category),
            "tags": self._generate_tags(category),
            "metadata": {
                "created_at": self._generate_timestamp(),
                "category": category,
                "confidence": self.random.uniform(0.7, 1.0),
                "source": "mock_generated"
            }
        }

        # Add custom fields
        if custom_fields:
            unit.update(custom_fields)

        return unit

    def generate_units(
        self,
        count: int,
        unit_types: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate multiple memory units.

        Args:
            count: Number of units to generate
            unit_types: List of allowed unit types
            categories: List of allowed categories
            **kwargs: Additional arguments for generate_unit

        Returns:
            List of generated memory units
        """
        units = []

        for _ in range(count):
            unit_type = None
            if unit_types:
                unit_type = self.random.choice(unit_types)

            category = None
            if categories:
                category = self.random.choice(categories)

            unit = self.generate_unit(
                unit_type=unit_type, category=category, **kwargs)
            units.append(unit)

        return units

    def _generate_content(self, unit_type: str, category: str) -> str:
        """Generate content for a unit."""
        templates = self.content_templates.get(
            unit_type, ["Generic content about {topic}."])
        template = self.random.choice(templates)

        # Fill in template variables
        topics = self.topic_categories.get(category, ["general topics"])
        topic = self.random.choice(topics)

        # Generate related concepts
        concepts = self.random.sample(topics, min(3, len(topics)))
        processes = ["analysis", "design", "implementation", "testing"]
        benefits = ["efficiency", "scalability",
                    "maintainability", "reliability"]
        applications = ["problem solving", "system design", "optimization"]
        prerequisites = ["basic knowledge",
                         "foundation skills", "prior experience"]
        outcomes = ["expertise", "proficiency", "competency"]

        techniques = ["practice", "experimentation", "study", "application"]
        tools = ["frameworks", "libraries", "platforms", "methodologies"]
        aspects = ["theory", "practice", "implementation", "optimization"]
        experience = ["training", "hands-on work", "study", "mentoring"]

        capabilities = ["automation", "analysis", "creation", "optimization"]
        methods = ["algorithms", "frameworks", "tools", "processes"]
        competencies = ["analysis", "design", "implementation", "debugging"]
        achievements = ["projects", "certifications",
                        "recognition", "advancement"]

        tool_types = ["framework", "library", "platform", "utility"]
        functions = ["development", "analysis", "deployment", "monitoring"]
        domain = self.random.choice(
            ["software development", "data science", "research", "production"])
        tool_name = f"{topic}Tool_{self.random.randint(1, 100)}"
        efficiencies = ["speed", "accuracy", "scalability", "reliability"]
        artifacts = ["code", "data", "models", "applications"]
        features = ["APIs", "interfaces", "integrations", "automation"]
        use_cases = ["development", "testing", "deployment", "monitoring"]
        field = self.random.choice(
            ["technology", "science", "engineering", "business"])
        workflows = ["development", "deployment", "maintenance", "scaling"]
        ecosystem = ["existing tools", "platforms", "services", "communities"]

        concept = f"{topic}Concept_{self.random.randint(1, 100)}"
        concrete_examples = ["specific instances",
                             "real-world cases", "practical examples"]
        generalizations = ["patterns", "principles", "frameworks", "models"]
        patterns = ["behaviors", "structures", "processes", "relationships"]
        domains = ["fields", "areas", "disciplines", "industries"]
        variations = ["instances", "cases", "examples", "scenarios"]
        principles = ["laws", "rules", "guidelines", "best practices"]
        applications2 = ["solutions", "systems", "products", "services"]

        # Format the template
        try:
            content = template.format(
                topic=topic, concepts=", ".join(concepts), processes=self.random.choice(processes),
                benefits=self.random.choice(benefits), applications=self.random.choice(applications),
                prerequisites=self.random.choice(prerequisites), outcomes=self.random.choice(outcomes),
                techniques=self.random.choice(techniques), tools=self.random.choice(tools),
                aspects=self.random.choice(aspects), experience=self.random.choice(experience),
                capabilities=self.random.choice(capabilities), methods=self.random.choice(methods),
                competencies=self.random.choice(competencies), achievements=self.random.choice(achievements),
                tool_name=tool_name, tool_type=self.random.choice(tool_types), functions=self.random.choice(functions),
                domain=domain, efficiencies=self.random.choice(efficiencies), artifacts=self.random.choice(artifacts),
                features=self.random.choice(features), use_cases=self.random.choice(use_cases), field=field,
                workflows=self.random.choice(workflows), ecosystem=self.random.choice(ecosystem),
                concept=concept, concrete_examples=self.random.choice(
                    concrete_examples),
                generalizations=self.random.choice(generalizations), patterns=self.random.choice(patterns),
                domains=self.random.choice(domains), variations=self.random.choice(variations),
                principles=self.random.choice(principles), applications2=self.random.choice(applications2)
            )
        except (KeyError, ValueError):
            content = f"Content about {topic} in the context of {unit_type}."

        return content

    def _generate_tags(self, category: str) -> List[str]:
        """Generate tags for a unit."""
        base_tags = [category]
        topics = self.topic_categories.get(category, [])

        # Add 1-3 random topics as tags
        num_tags = self.random.randint(1, 3)
        additional_tags = self.random.sample(
            topics, min(num_tags, len(topics)))

        # Add some generic tags
        generic_tags = ["learning", "knowledge", "expertise", "proficiency"]
        if self.random.random() < 0.3:
            additional_tags.append(self.random.choice(generic_tags))

        return base_tags + additional_tags

    def _generate_timestamp(self) -> str:
        """Generate a random timestamp within the last 30 days."""
        now = datetime.now(timezone.utc)
        days_ago = self.random.randint(0, 30)
        hours_ago = self.random.randint(0, 23)
        minutes_ago = self.random.randint(0, 59)

        timestamp = now - timedelta(days=days_ago,
                                    hours=hours_ago, minutes=minutes_ago)
        return timestamp.isoformat() + "Z"

    def _generate_source(self) -> str:
        """Generate a source identifier."""
        sources = ["user_input", "documentation",
                   "tutorial", "experience", "research", "practice"]
        return self.random.choice(sources)


class ExperienceGenerator:
    """Generator for complete experiences that can be added to memory systems."""

    def __init__(self, unit_generator: Optional[MemoryUnitGenerator] = None, seed: Optional[int] = None, use_real_encoding: bool = True):
        self.logger = get_logger("experience_generator")
        self.unit_generator = unit_generator or MemoryUnitGenerator(
            seed, use_real_encoding=use_real_encoding)

    def generate_experience(
        self,
        experience_type: str = "mixed",
        size: str = "small",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a complete experience.

        Args:
            experience_type: Type of experience ("lesson", "skill", "tool", "abstraction", "mixed")
            size: Size of experience ("small", "medium", "large")
            **kwargs: Additional arguments

        Returns:
            Experience dictionary
        """
        size_configs = {
            "small": {"units": (1, 3), "complexity": "simple"},
            "medium": {"units": (3, 8), "complexity": "moderate"},
            "large": {"units": (8, 15), "complexity": "complex"}
        }

        config = size_configs.get(size, size_configs["small"])
        num_units = self.unit_generator.random.randint(*config["units"])

        # Determine unit types
        if experience_type == "mixed":
            unit_types = list(self.unit_generator.content_templates.keys())
        else:
            unit_types = [experience_type] * num_units

        # Generate units
        units = []
        for i in range(num_units):
            unit_type = unit_types[i % len(
                unit_types)] if experience_type == "mixed" else experience_type
            unit = self.unit_generator.generate_unit(
                unit_type=unit_type, **kwargs)
            units.append(unit)

        # Create experience
        experience = {
            "id": f"exp_{uuid.uuid4().hex[:12]}",
            "type": "experience",
            "title": self._generate_experience_title(experience_type, size),
            "description": self._generate_experience_description(experience_type, size),
            "units": units,
            "metadata": {
                "created_at": self.unit_generator._generate_timestamp(),
                "experience_type": experience_type,
                "size": size,
                "num_units": num_units,
                "categories": list(set(u.get("metadata", {}).get("category") for u in units))
            }
        }

        return experience

    def generate_experience_batch(
        self,
        count: int,
        experience_types: Optional[List[str]] = None,
        sizes: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate a batch of experiences.

        Args:
            count: Number of experiences to generate
            experience_types: List of experience types to use
            sizes: List of sizes to use
            **kwargs: Additional arguments

        Returns:
            List of generated experiences
        """
        experiences = []

        for _ in range(count):
            exp_type = self.unit_generator.random.choice(
                experience_types) if experience_types else "mixed"
            size = self.unit_generator.random.choice(
                sizes) if sizes else "small"

            experience = self.generate_experience(
                experience_type=exp_type,
                size=size,
                **kwargs
            )
            experiences.append(experience)

        return experiences

    def _generate_experience_title(self, exp_type: str, size: str) -> str:
        """Generate a title for the experience."""
        titles = {
            "lesson": ["Learning Session", "Study Experience", "Knowledge Acquisition"],
            "skill": ["Skill Development", "Practice Session", "Proficiency Building"],
            "tool": ["Tool Exploration", "Technology Adoption", "Framework Learning"],
            "abstraction": ["Conceptual Understanding", "Theory Development", "Pattern Recognition"],
            "mixed": ["Comprehensive Learning", "Multi-faceted Experience", "Diverse Knowledge"]
        }

        type_titles = titles.get(exp_type, titles["mixed"])
        base_title = self.unit_generator.random.choice(type_titles)

        size_modifiers = {
            "small": ["Brief", "Introductory", "Foundational"],
            "medium": ["Comprehensive", "In-depth", "Detailed"],
            "large": ["Extensive", "Complete", "Thorough"]
        }

        modifier = self.unit_generator.random.choice(
            size_modifiers.get(size, ["General"]))
        return f"{modifier} {base_title}"

    def _generate_experience_description(self, exp_type: str, size: str) -> str:
        """Generate a description for the experience."""
        descriptions = [
            "An experience focused on {focus} through {method}.",
            "Comprehensive exploration of {focus} involving {method}.",
            "Practical engagement with {focus} using {method}."
        ]

        focus_map = {
            "lesson": "conceptual learning",
            "skill": "skill acquisition",
            "tool": "tool mastery",
            "abstraction": "abstract thinking",
            "mixed": "diverse learning"
        }

        method_map = {
            "small": "targeted practice",
            "medium": "structured learning",
            "large": "intensive study"
        }

        focus = focus_map.get(exp_type, "learning")
        method = method_map.get(size, "systematic approach")

        template = self.unit_generator.random.choice(descriptions)
        return template.format(focus=focus, method=method)


class ScenarioGenerator:
    """Generator for complete test scenarios."""

    def __init__(self, seed: Optional[int] = None, use_real_encoding: bool = True):
        self.logger = get_logger("test_scenario_generator")
        self.unit_generator = MemoryUnitGenerator(
            seed, use_real_encoding=use_real_encoding)
        self.experience_generator = ExperienceGenerator(
            self.unit_generator, seed, use_real_encoding=use_real_encoding)

    def generate_scenario(
        self,
        scenario_type: str = "basic",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a complete test scenario.

        Args:
            scenario_type: Type of scenario ("basic", "complex", "performance", "edge_case")
            **kwargs: Additional arguments

        Returns:
            Complete test scenario
        """
        scenarios = {
            "basic": self._generate_basic_scenario,
            "complex": self._generate_complex_scenario,
            "performance": self._generate_performance_scenario,
            "edge_case": self._generate_edge_case_scenario
        }

        generator = scenarios.get(scenario_type, self._generate_basic_scenario)
        return generator(**kwargs)

    def _generate_basic_scenario(self, **kwargs) -> Dict[str, Any]:
        """Generate a basic test scenario."""
        units = self.unit_generator.generate_units(
            count=10,
            unit_types=["lesson", "skill"],
            categories=["programming", "ai"]
        )

        return {
            "name": "basic_test_scenario",
            "description": "Basic scenario with mixed lesson and skill units",
            "units": units,
            "expected_outcomes": {
                "total_units": 10,
                "unit_types": {"lesson": 5, "skill": 5},
                "categories": ["programming", "ai"]
            }
        }

    def _generate_complex_scenario(self, **kwargs) -> Dict[str, Any]:
        """Generate a complex test scenario."""
        experiences = self.experience_generator.generate_experience_batch(
            count=3,
            experience_types=["mixed"],
            sizes=["medium", "large"]
        )

        # Flatten units from experiences
        units = []
        for exp in experiences:
            units.extend(exp["units"])

        return {
            "name": "complex_test_scenario",
            "description": "Complex scenario with multiple experiences and diverse content",
            "experiences": experiences,
            "units": units,
            "expected_outcomes": {
                "total_units": len(units),
                "total_experiences": 3,
                "experience_sizes": ["medium", "large"]
            }
        }

    def _generate_performance_scenario(self, **kwargs) -> Dict[str, Any]:
        """Generate a performance testing scenario."""
        units = self.unit_generator.generate_units(
            count=100,
            unit_types=["lesson", "skill", "tool"],
            categories=["programming", "ai", "data", "engineering"]
        )

        return {
            "name": "performance_test_scenario",
            "description": "Large-scale scenario for performance testing",
            "units": units,
            "expected_outcomes": {
                "total_units": 100,
                # Roughly equal
                "unit_types": {"lesson": 34, "skill": 33, "tool": 33},
                "categories": ["programming", "ai", "data", "engineering"]
            }
        }

    def _generate_edge_case_scenario(self, **kwargs) -> Dict[str, Any]:
        """Generate an edge case testing scenario."""
        # Generate units with extreme values
        units = []

        # Very long content
        long_content_unit = self.unit_generator.generate_unit()
        long_content_unit["content"] = "Very long content. " * 1000
        units.append(long_content_unit)

        # Unit with many tags
        many_tags_unit = self.unit_generator.generate_unit()
        many_tags_unit["tags"] = [f"tag_{i}" for i in range(50)]
        units.append(many_tags_unit)

        # Unit with special characters
        special_chars_unit = self.unit_generator.generate_unit()
        special_chars_unit["content"] = "Content with special chars: àáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"
        units.append(special_chars_unit)

        # Empty content unit
        empty_content_unit = self.unit_generator.generate_unit()
        empty_content_unit["content"] = ""
        units.append(empty_content_unit)

        return {
            "name": "edge_case_test_scenario",
            "description": "Scenario with edge cases for robustness testing",
            "units": units,
            "expected_outcomes": {
                "total_units": 4,
                "edge_cases": ["long_content", "many_tags", "special_chars", "empty_content"]
            }
        }


# Convenience functions
def generate_test_units(count: int = 10, use_real_encoding: bool = True, **kwargs) -> List[Dict[str, Any]]:
    """Convenience function to generate test memory units."""
    generator = MemoryUnitGenerator(use_real_encoding=use_real_encoding)
    return generator.generate_units(count, **kwargs)


def generate_test_experience(use_real_encoding: bool = True, **kwargs) -> Dict[str, Any]:
    """Convenience function to generate a test experience."""
    generator = ExperienceGenerator(use_real_encoding=use_real_encoding)
    return generator.generate_experience(**kwargs)


def generate_test_scenario(scenario_type: str = "basic", use_real_encoding: bool = True, **kwargs) -> Dict[str, Any]:
    """Convenience function to generate a test scenario."""
    generator = ScenarioGenerator(use_real_encoding=use_real_encoding)
    return generator.generate_scenario(scenario_type, **kwargs)
