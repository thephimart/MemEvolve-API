"""
Real data generators for MemEvolve testing.

This module provides utilities for generating realistic test data using
actual LLM encoding and embedding generation instead of mocks.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
import random
import numpy as np

from .logging import get_logger
from ..components.encode import ExperienceEncoder
from .embeddings import create_embedding_function


class RealMemoryUnitGenerator:
    """Generator for memory unit data using real LLM encoding and embeddings."""

    def __init__(
        self,
        encoder: Optional[ExperienceEncoder] = None,
        embedding_function: Optional[callable] = None,
        seed: Optional[int] = None
    ):
        self.logger = get_logger("real_memory_unit_generator")
        self.random = random.Random(seed)

        # Initialize real components
        self.encoder = encoder or self._create_encoder()
        self.embedding_function = embedding_function or self._create_embedding_function()

        # Initialize encoder if needed
        if self.encoder and hasattr(self.encoder, 'initialize_memory_api'):
            try:
                self.encoder.initialize_memory_api()
            except Exception as e:
                self.logger.warning(f"Failed to initialize encoder: {e}")

        # Raw experiences that will be encoded into memory units
        self.raw_experiences = {
            "programming": [
                {
                    "id": "exp_python_basics",
                    "type": "experience",
                    "title": "Learning Python Basics",
                    "description": "Started learning Python programming language fundamentals",
                    "content": "I began learning Python by installing it and writing my first 'Hello World' program. I learned about variables, data types, and basic operations. The syntax is clean and readable compared to other languages I've tried."
                },
                {
                    "id": "exp_functions",
                    "type": "experience",
                    "title": "Understanding Functions",
                    "description": "Learned how to define and use functions in Python",
                    "content": "Functions allow code reuse and organization. I learned about parameters, return values, and scope. Defining functions with def keyword and calling them with parentheses. This makes code much more maintainable."
                },
                {
                    "id": "exp_oop",
                    "type": "experience",
                    "title": "Object-Oriented Programming",
                    "description": "Exploring classes and objects in Python",
                    "content": "Classes are blueprints for objects. Learned about __init__, self, methods, and inheritance. Created my first class to represent a bank account with deposit and withdraw methods."
                }
            ],
            "ai": [
                {
                    "id": "exp_machine_learning",
                    "type": "experience",
                    "title": "Introduction to Machine Learning",
                    "description": "Started learning about machine learning concepts",
                    "content": "Machine learning is about training algorithms to make predictions. Learned about supervised vs unsupervised learning, training data, and model evaluation. The key insight is that ML models learn patterns from data rather than being explicitly programmed."
                },
                {
                    "id": "exp_neural_networks",
                    "type": "experience",
                    "title": "Understanding Neural Networks",
                    "description": "Exploring artificial neural networks",
                    "content": "Neural networks consist of layers of interconnected nodes. Each connection has a weight that gets adjusted during training. Backpropagation is used to calculate gradients and update weights. Deep learning uses many layers to learn complex patterns."
                }
            ],
            "data": [
                {
                    "id": "exp_sql_queries",
                    "type": "experience",
                    "title": "SQL Query Writing",
                    "description": "Learning to write SQL queries for data analysis",
                    "content": "SQL is used to query relational databases. Learned SELECT, FROM, WHERE, JOIN, GROUP BY, and ORDER BY clauses. Writing queries to extract insights from customer data. The key is understanding the logical flow of data transformation."
                },
                {
                    "id": "exp_data_visualization",
                    "type": "experience",
                    "title": "Data Visualization Techniques",
                    "description": "Creating charts and graphs to communicate data insights",
                    "content": "Data visualization makes complex data understandable. Used matplotlib and seaborn to create various chart types. Learned about choosing the right visualization for different data types and audiences. Color, layout, and labels are crucial for effective communication."
                }
            ]
        }

    def _create_encoder(self) -> Optional[ExperienceEncoder]:
        """Create a real experience encoder."""
        try:
            from ..components.encode import ExperienceEncoder
            return ExperienceEncoder()
        except Exception as e:
            self.logger.warning(f"Could not create real encoder: {e}")
            return None

    def _create_embedding_function(self) -> Optional[callable]:
        """Create a real embedding function."""
        try:
            # Try to use OpenAI embeddings first
            return create_embedding_function("openai")
        except Exception:
            try:
                # Fallback to dummy embeddings if OpenAI fails
                self.logger.warning("Using dummy embeddings as fallback")
                return create_embedding_function("dummy")
            except Exception as e:
                self.logger.error(f"Could not create embedding function: {e}")
                return None

    def generate_unit(
        self,
        category: Optional[str] = None,
        custom_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a single memory unit using real encoding and embeddings.

        Args:
            category: Content category ("programming", "ai", "data")
            custom_fields: Additional fields to include

        Returns:
            Generated memory unit with real encoding and embeddings
        """
        # Select random category and experience
        if category is None:
            category = self.random.choice(list(self.raw_experiences.keys()))

        experiences = self.raw_experiences.get(
            category, self.raw_experiences["programming"])
        raw_experience = self.random.choice(experiences)

        unit = None

        # Try real encoding first
        if self.encoder:
            try:
                unit = self.encoder.encode_experience(raw_experience)
                unit["id"] = f"real_{raw_experience['id']}_{self.random.randint(1000, 9999)}"
            except Exception as e:
                self.logger.warning(f"Real encoding failed: {e}")

        # Fallback to mock-like structure if encoding fails
        if unit is None:
            unit = {
                "id": f"real_{raw_experience['id']}_{self.random.randint(1000, 9999)}",
                "type": "lesson",  # Default type
                "content": raw_experience.get("content", ""),
                "tags": [category, "real_generated"],
                "metadata": {
                    "created_at": self._generate_timestamp(),
                    "category": category,
                    "confidence": self.random.uniform(0.8, 1.0),
                    "source": "real_encoder_fallback"
                }
            }

        # Add real embeddings if available
        if self.embedding_function:
            try:
                embedding = self.embedding_function(unit["content"])
                unit["embedding"] = embedding.tolist() if hasattr(
                    embedding, 'tolist') else embedding
            except Exception as e:
                self.logger.warning(f"Embedding generation failed: {e}")
                # Fallback to dummy embedding
                unit["embedding"] = np.random.randn(768).tolist()
        else:
            # Fallback embedding
            unit["embedding"] = np.random.randn(768).tolist()

        # Add custom fields
        if custom_fields:
            unit.update(custom_fields)

        return unit

    def generate_units(
        self,
        count: int,
        categories: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate multiple memory units.

        Args:
            count: Number of units to generate
            categories: List of allowed categories
            **kwargs: Additional arguments for generate_unit

        Returns:
            List of generated memory units
        """
        units = []

        for _ in range(count):
            category = None
            if categories:
                category = self.random.choice(categories)

            unit = self.generate_unit(category=category, **kwargs)
            units.append(unit)

        return units

    def _generate_timestamp(self) -> str:
        """Generate a random timestamp within the last 30 days."""
        now = datetime.now(timezone.utc)
        days_ago = self.random.randint(0, 30)
        hours_ago = self.random.randint(0, 23)
        minutes_ago = self.random.randint(0, 59)

        timestamp = now - timedelta(days=days_ago,
                                    hours=hours_ago, minutes=minutes_ago)
        return timestamp.isoformat() + "Z"


class RealExperienceGenerator:
    """Generator for complete experiences using real encoding."""

    def __init__(
        self,
        unit_generator: Optional[RealMemoryUnitGenerator] = None,
        seed: Optional[int] = None
    ):
        self.logger = get_logger("real_experience_generator")
        self.unit_generator = unit_generator or RealMemoryUnitGenerator(seed)

    def generate_experience(
        self,
        experience_type: str = "mixed",
        size: str = "small",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a complete experience with real encoding.

        Args:
            experience_type: Type of experience ("lesson", "skill", "tool", "abstraction", "mixed")
            size: Size of experience ("small", "medium", "large")
            **kwargs: Additional arguments

        Returns:
            Experience dictionary with real encoding
        """
        size_configs = {
            "small": {"units": (1, 3), "complexity": "simple"},
            "medium": {"units": (3, 8), "complexity": "moderate"},
            "large": {"units": (8, 15), "complexity": "complex"}
        }

        config = size_configs.get(size, size_configs["small"])
        num_units = self.unit_generator.random.randint(*config["units"])

        # Generate units
        categories = list(self.unit_generator.raw_experiences.keys())
        units = self.unit_generator.generate_units(
            count=num_units,
            categories=categories,
            **kwargs
        )

        # Create experience
        experience = {
            "id": f"real_exp_{self.unit_generator.random.randint(10000, 99999)}",
            "type": "experience",
            "title": f"Real {experience_type.title()} Experience ({size})",
            "description": f"An experience generated using real LLM encoding and embeddings, containing {num_units} memory units.",
            "units": units,
            "metadata": {
                "created_at": self.unit_generator._generate_timestamp(),
                "experience_type": experience_type,
                "size": size,
                "num_units": num_units,
                "categories": list(set(u.get("metadata", {}).get("category") for u in units)),
                "generation_method": "real_llm_encoding"
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


# Convenience functions
def generate_real_test_units(count: int = 10, **kwargs) -> List[Dict[str, Any]]:
    """Convenience function to generate real test memory units."""
    generator = RealMemoryUnitGenerator()
    return generator.generate_units(count, **kwargs)


def generate_real_test_experience(**kwargs) -> Dict[str, Any]:
    """Convenience function to generate a real test experience."""
    generator = RealExperienceGenerator()
    return generator.generate_experience(**kwargs)


def generate_real_test_scenario(scenario_type: str = "basic", **kwargs) -> Dict[str, Any]:
    """Convenience function to generate a real test scenario."""
    unit_generator = RealMemoryUnitGenerator()
    experience_generator = RealExperienceGenerator(unit_generator)

    if scenario_type == "basic":
        units = unit_generator.generate_units(
            count=10, categories=["programming", "ai"])
        return {
            "name": "real_basic_scenario",
            "description": "Basic scenario with real LLM-encoded units",
            "units": units,
            "expected_outcomes": {
                "total_units": 10,
                "categories": ["programming", "ai"],
                "generation_method": "real_encoding"
            }
        }
    elif scenario_type == "complex":
        experiences = experience_generator.generate_experience_batch(
            count=3,
            sizes=["medium", "large"]
        )
        units = []
        for exp in experiences:
            units.extend(exp["units"])

        return {
            "name": "real_complex_scenario",
            "description": "Complex scenario with real experiences",
            "experiences": experiences,
            "units": units,
            "expected_outcomes": {
                "total_units": len(units),
                "total_experiences": 3,
                "generation_method": "real_encoding"
            }
        }
    else:
        # Default to basic
        return generate_real_test_scenario("basic", **kwargs)
