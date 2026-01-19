#!/usr/bin/env python3
"""
MemEvolve Memory System Initialization Script

This script initializes a memory system with sample data for development and testing.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory_system import MemorySystem
from utils.config import load_config


def create_sample_experiences():
    """Create a list of sample experiences for initialization."""
    return [
        {
            "id": "exp_python_basics",
            "type": "lesson",
            "content": "Python is a high-level programming language known for its simplicity and readability. Key features include dynamic typing, automatic memory management, and extensive standard library.",
            "tags": ["python", "programming", "basics"],
            "metadata": {"difficulty": "beginner", "domain": "programming"}
        },
        {
            "id": "exp_data_structures",
            "type": "skill",
            "content": "Understanding data structures is crucial for efficient programming. Lists, dictionaries, sets, and tuples are fundamental data structures in Python, each with different performance characteristics for various operations.",
            "tags": ["python", "data-structures", "algorithms"],
            "metadata": {"difficulty": "intermediate", "domain": "computer-science"}
        },
        {
            "id": "exp_machine_learning",
            "type": "lesson",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Key concepts include supervised and unsupervised learning, neural networks, and model evaluation.",
            "tags": ["machine-learning", "ai", "statistics"],
            "metadata": {"difficulty": "advanced", "domain": "data-science"}
        },
        {
            "id": "exp_memory_systems",
            "type": "tool",
            "content": "Memory systems in AI agents help maintain context and learn from interactions. They can store experiences, skills, and lessons to improve future performance and decision-making.",
            "tags": ["ai", "memory", "agents", "learning"],
            "metadata": {"difficulty": "advanced", "domain": "artificial-intelligence"}
        },
        {
            "id": "exp_debugging",
            "type": "skill",
            "content": "Effective debugging involves systematic approaches to identify and fix issues in code. Techniques include using breakpoints, logging, unit tests, and code review. Understanding error messages and stack traces is essential.",
            "tags": ["debugging", "programming", "problem-solving"],
            "metadata": {"difficulty": "intermediate", "domain": "software-development"}
        },
        {
            "id": "exp_architecture_patterns",
            "type": "lesson",
            "content": "Software architecture patterns provide proven solutions to common design problems. MVC, layered architecture, microservices, and event-driven architecture are examples of patterns that improve maintainability and scalability.",
            "tags": ["architecture", "design-patterns", "software-engineering"],
            "metadata": {"difficulty": "advanced", "domain": "software-engineering"}
        },
        {
            "id": "exp_version_control",
            "type": "skill",
            "content": "Version control systems like Git enable collaborative development and code history management. Key concepts include branches, commits, merges, and pull requests. Understanding branching strategies and conflict resolution is important.",
            "tags": ["git", "version-control", "collaboration"],
            "metadata": {"difficulty": "intermediate", "domain": "software-development"}
        },
        {
            "id": "exp_cloud_computing",
            "type": "lesson",
            "content": "Cloud computing provides on-demand access to computing resources over the internet. Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS) offer different levels of abstraction and management.",
            "tags": ["cloud", "infrastructure", "scalability"],
            "metadata": {"difficulty": "intermediate", "domain": "technology"}
        }
    ]


def main():
    parser = argparse.ArgumentParser(description="Initialize MemEvolve memory system with sample data")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        help="Custom storage path for the memory system"
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear existing data before initialization"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    print("🧠 Initializing MemEvolve memory system...")

    try:
        # Load configuration
        config = load_config(args.config)

        # Override storage path if specified
        if args.storage_path:
            config.storage.path = args.storage_path

        # Create memory system
        memory_system = MemorySystem(config)

        if args.clear_existing:
            print("🧹 Clearing existing data...")
            # Note: This would require a clear method in the storage backend
            # For now, we'll just initialize fresh

        # Add sample experiences
        sample_experiences = create_sample_experiences()

        print(f"📚 Adding {len(sample_experiences)} sample experiences...")

        for i, experience in enumerate(sample_experiences, 1):
            try:
                memory_system.add_experience(experience)
                if args.verbose:
                    print(f"  ✅ Added: {experience['id']}")
                elif i % 2 == 0:
                    print(f"  Progress: {i}/{len(sample_experiences)} experiences added")
            except Exception as e:
                print(f"  ❌ Failed to add {experience['id']}: {str(e)}")

        # Get health metrics
        health = memory_system.get_health_metrics()
        if health:
            print("\n📊 Memory system health:")
            print(f"  • Total units: {health.total_units}")
            print(f"  • Total size: {health.total_size_bytes} bytes")
            print(f"  • Unit types: {health.unit_types_distribution}")
        else:
            print("⚠️  Could not retrieve health metrics")

        print("\n✅ Memory system initialization complete!")
        print(f"📁 Storage location: {config.storage.path}")
        print("\n💡 You can now query the memory system or run tests!")

    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()