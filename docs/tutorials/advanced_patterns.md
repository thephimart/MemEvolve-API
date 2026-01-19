# Advanced Memory Patterns with MemEvolve

This tutorial covers advanced patterns and techniques for building sophisticated memory-augmented applications with MemEvolve.

## Prerequisites

- Completed [Quick Start Tutorial](quick_start.md)
- Basic understanding of MemEvolve components
- Familiarity with Python async/await patterns

## 1. Custom Retrieval Strategies

MemEvolve supports multiple retrieval strategies. Let's explore advanced usage patterns.

### LLM-Guided Retrieval with Custom Prompts

```python
from components.retrieve import LLMGuidedRetrievalStrategy, SemanticRetrievalStrategy

def custom_llm_call(prompt: str) -> str:
    """Custom LLM function with domain-specific guidance."""
    # Add domain context to prompts
    enhanced_prompt = f"""
    You are an expert software engineering assistant with deep knowledge of:
    - System design and architecture
    - Debugging methodologies
    - Performance optimization
    - Code quality and best practices

    {prompt}
    """
    # Call your LLM API here
    return call_your_llm_api(enhanced_prompt)

# Create hybrid retrieval with LLM guidance
base_strategy = SemanticRetrievalStrategy()
llm_guided_strategy = LLMGuidedRetrievalStrategy(
    llm_client_callable=custom_llm_call,
    base_strategy=base_strategy,
    reasoning_temperature=0.1,  # Lower temperature for more focused reasoning
    max_reasoning_tokens=200
)

config = MemorySystemConfig(
    llm_base_url="http://localhost:8080/v1",
    llm_api_key="your-key",
    retrieval_strategy=llm_guided_strategy
)

memory = MemorySystem(config)
```

### Multi-Stage Retrieval Pipeline

```python
from components.retrieve import HybridRetrievalStrategy, KeywordRetrievalStrategy
import asyncio

class MultiStageRetrievalStrategy(RetrievalStrategy):
    """Multi-stage retrieval combining multiple strategies."""

    def __init__(self, strategies_and_weights):
        self.strategies = strategies_and_weights  # List of (strategy, weight) tuples

    def retrieve(self, query, storage_backend, top_k=5, filters=None):
        all_results = []

        # Stage 1: Gather candidates from all strategies
        for strategy, weight in self.strategies:
            results = strategy.retrieve(query, storage_backend, top_k=top_k*2, filters=filters)
            # Add strategy weight to results
            for result in results:
                result['strategy_weight'] = weight
            all_results.extend(results)

        # Stage 2: Rerank and combine results
        combined_results = self._combine_and_rerank(all_results, top_k)
        return combined_results

    def _combine_and_rerank(self, all_results, top_k):
        """Combine results from multiple strategies with intelligent reranking."""
        # Group by memory ID
        memory_groups = {}
        for result in all_results:
            mem_id = result['id']
            if mem_id not in memory_groups:
                memory_groups[mem_id] = {
                    'memory': result,
                    'scores': [],
                    'strategies': []
                }

            memory_groups[mem_id]['scores'].append(result.get('score', 0))
            memory_groups[mem_id]['strategies'].append(result.get('strategy_weight', 1))

        # Calculate combined scores
        ranked_results = []
        for mem_id, group in memory_groups.items():
            # Weighted combination of scores
            combined_score = sum(
                score * weight
                for score, weight in zip(group['scores'], group['strategies'])
            ) / len(group['scores'])

            result = group['memory'].copy()
            result['score'] = combined_score
            result['strategy_count'] = len(group['strategies'])
            ranked_results.append(result)

        # Sort by combined score and return top_k
        ranked_results.sort(key=lambda x: x['score'], reverse=True)
        return ranked_results[:top_k]

# Usage
multi_stage = MultiStageRetrievalStrategy([
    (KeywordRetrievalStrategy(), 0.3),
    (SemanticRetrievalStrategy(), 0.7)
])

config.retrieval_strategy = multi_stage
memory = MemorySystem(config)
```

## 2. Memory Lifecycle Management

Advanced memory management strategies for different use cases.

### Time-Based Memory Decay

```python
from components.manage import ManagementStrategy
from datetime import datetime, timedelta
import math

class TimeDecayManagementStrategy(ManagementStrategy):
    """Management strategy that decays memory importance over time."""

    def __init__(self, half_life_days=30, min_importance=0.1):
        self.half_life_days = half_life_days
        self.min_importance = min_importance

    def manage(self, storage_backend, context=None):
        """Apply time-based decay to memory importance."""
        all_memories = storage_backend.retrieve_all()
        updated_count = 0

        for memory in all_memories:
            age_days = self._calculate_age_days(memory)
            decay_factor = self._calculate_decay_factor(age_days)

            # Update memory metadata with decay information
            metadata = memory.get('metadata', {})
            metadata['decay_factor'] = decay_factor
            metadata['last_decay_update'] = datetime.now().isoformat()

            # Update importance score if it exists
            if 'importance' in metadata:
                original_importance = metadata['original_importance'] = metadata['original_importance'] or metadata['importance']
                metadata['importance'] = max(
                    original_importance * decay_factor,
                    self.min_importance
                )

            # Update the memory
            storage_backend.update(memory['id'], {
                'metadata': metadata
            })
            updated_count += 1

        return {
            'memories_processed': updated_count,
            'decay_applied': True,
            'half_life_days': self.half_life_days
        }

    def _calculate_age_days(self, memory):
        """Calculate how many days old a memory is."""
        created_at = memory.get('metadata', {}).get('created_at')
        if not created_at:
            return 0

        try:
            created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            age = datetime.now(created_dt.tzinfo) - created_dt
            return age.days
        except:
            return 0

    def _calculate_decay_factor(self, age_days):
        """Calculate exponential decay factor."""
        if age_days <= 0:
            return 1.0

        # Exponential decay: factor = e^(-ln(2) * age / half_life)
        return math.exp(-math.log(2) * age_days / self.half_life_days)

# Usage
decay_strategy = TimeDecayManagementStrategy(half_life_days=60)
config.management_strategy = decay_strategy
memory = MemorySystem(config)
```

### Semantic Clustering for Memory Organization

```python
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class SemanticClusteringStrategy(ManagementStrategy):
    """Organize memories into semantic clusters."""

    def __init__(self, num_clusters=10, embedding_function=None):
        self.num_clusters = num_clusters
        self.embedding_function = embedding_function
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    def manage(self, storage_backend, context=None):
        """Cluster memories by semantic similarity."""
        all_memories = storage_backend.retrieve_all()

        if len(all_memories) < self.num_clusters:
            return {"error": "Not enough memories for clustering"}

        # Extract content for clustering
        contents = [mem.get('content', '') for mem in all_memories]

        # Generate embeddings or use TF-IDF
        if self.embedding_function:
            embeddings = np.array([self.embedding_function(content) for content in contents])
        else:
            # Fallback to TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(contents)
            embeddings = tfidf_matrix.toarray()

        # Perform clustering
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        # Update memories with cluster information
        updated_count = 0
        for memory, cluster_id in zip(all_memories, clusters):
            metadata = memory.get('metadata', {})
            metadata['cluster_id'] = int(cluster_id)
            metadata['cluster_center_distance'] = float(
                np.linalg.norm(embeddings[updated_count] - kmeans.cluster_centers_[cluster_id])
            )

            storage_backend.update(memory['id'], {'metadata': metadata})
            updated_count += 1

        return {
            'memories_clustered': updated_count,
            'num_clusters': self.num_clusters,
            'cluster_sizes': np.bincount(clusters).tolist()
        }

# Usage
clustering_strategy = SemanticClusteringStrategy(num_clusters=8)
config.management_strategy = clustering_strategy
memory = MemorySystem(config)
```

## 3. Custom Encoding Strategies

Building specialized encoders for domain-specific memory processing.

### Multi-Modal Experience Encoder

```python
from components.encode.encoder import ExperienceEncoder
from typing import Dict, Any, List
import base64

class MultiModalExperienceEncoder(ExperienceEncoder):
    """Encoder that handles text, images, and code experiences."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supported_modalities = ['text', 'image', 'code', 'data']

    def encode_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Encode multi-modal experiences."""
        modality = self._detect_modality(experience)

        if modality == 'image':
            return self._encode_image_experience(experience)
        elif modality == 'code':
            return self._encode_code_experience(experience)
        elif modality == 'data':
            return self._encode_data_experience(experience)
        else:
            # Default text encoding
            return super().encode_experience(experience)

    def _detect_modality(self, experience: Dict[str, Any]) -> str:
        """Detect the primary modality of the experience."""
        content = experience.get('content', '')
        action = experience.get('action', '')

        # Check for image indicators
        if any(keyword in content.lower() or keyword in action.lower()
               for keyword in ['image', 'photo', 'screenshot', 'diagram']):
            return 'image'

        # Check for code indicators
        if any(keyword in content.lower() or keyword in action.lower()
               for keyword in ['code', 'function', 'class', 'algorithm', 'script']):
            return 'code'

        # Check for data indicators
        if any(keyword in content.lower() or keyword in action.lower()
               for keyword in ['data', 'dataset', 'analysis', 'statistics', 'query']):
            return 'data'

        return 'text'

    def _encode_image_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Encode image-related experiences."""
        prompt = f"""
        Analyze this image-related experience and extract structured knowledge:

        Experience: {experience}

        Focus on:
        - Visual patterns or features learned
        - Image processing techniques used
        - Computer vision insights gained
        - Visual debugging approaches

        Format as JSON with type, content, and metadata.
        """

        return self._call_llm_and_parse(prompt)

    def _encode_code_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Encode programming/code experiences."""
        prompt = f"""
        Analyze this coding experience and extract algorithmic knowledge:

        Experience: {experience}

        Identify:
        - Programming patterns or techniques
        - Algorithm implementations
        - Code optimization approaches
        - Debugging methodologies
        - Best practices learned

        Classify as: lesson, skill, tool, or abstraction.
        """

        return self._call_llm_and_parse(prompt)

    def _encode_data_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Encode data analysis experiences."""
        prompt = f"""
        Analyze this data-related experience:

        Experience: {experience}

        Extract:
        - Data analysis methodologies
        - Statistical insights
        - Data processing techniques
        - Visualization approaches
        - Data quality lessons

        Focus on reusable data science knowledge.
        """

        return self._call_llm_and_parse(prompt)

# Usage
multimodal_encoder = MultiModalExperienceEncoder(
    base_url="http://localhost:8080/v1",
    api_key="your-key"
)

config.encoder = multimodal_encoder
memory = MemorySystem(config)
```

## 4. Memory-Augmented Reasoning Chains

Building complex reasoning workflows that leverage memory throughout the process.

### Memory-Guided Problem Solving

```python
class MemoryAugmentedSolver:
    """Solver that uses memory to guide problem-solving processes."""

    def __init__(self, memory_system):
        self.memory = memory_system

    async def solve_problem(self, problem_description: str) -> Dict[str, Any]:
        """Solve a problem using memory-augmented reasoning."""
        # Phase 1: Gather relevant memories
        relevant_memories = self.memory.query_memory(
            f"similar problems to: {problem_description}",
            top_k=10
        )

        # Phase 2: Analyze problem in context of past experiences
        analysis_prompt = self._build_analysis_prompt(problem_description, relevant_memories)
        analysis = await self._call_llm(analysis_prompt)

        # Phase 3: Generate solution approach
        solution_approach = self._extract_solution_approach(analysis)

        # Phase 4: Execute solution with memory feedback
        solution_result = await self._execute_with_memory_feedback(
            solution_approach,
            problem_description
        )

        # Phase 5: Store the solution experience
        self._store_solution_experience(problem_description, solution_result)

        return {
            'problem': problem_description,
            'analysis': analysis,
            'approach': solution_approach,
            'result': solution_result,
            'memories_used': len(relevant_memories)
        }

    def _build_analysis_prompt(self, problem, memories):
        """Build analysis prompt using relevant memories."""
        memory_context = "\n".join([
            f"- {mem['type']}: {mem['content'][:100]}..."
            for mem in memories[:5]  # Top 5 memories
        ])

        return f"""
        Analyze this problem in the context of past experiences:

        PROBLEM: {problem}

        RELEVANT PAST EXPERIENCES:
        {memory_context}

        Provide:
        1. Problem decomposition
        2. Relevant solution patterns from past experiences
        3. Potential challenges and how they were overcome before
        4. Recommended approach with justification
        """

    async def _execute_with_memory_feedback(self, approach, problem):
        """Execute solution while consulting memory for guidance."""
        steps = self._parse_approach_into_steps(approach)

        results = []
        for step in steps:
            # Query memory for step-specific guidance
            step_guidance = self.memory.query_memory(
                f"how to {step['description']}",
                top_k=3
            )

            # Execute step with guidance
            step_result = await self._execute_step_with_guidance(step, step_guidance)
            results.append(step_result)

            # Check if we need to adjust approach based on memory
            if self._should_adjust_approach(step_result, step_guidance):
                adjustment = self._get_adjustment_from_memory(step_result)
                # Apply adjustment...

        return results

    def _store_solution_experience(self, problem, result):
        """Store the solution as a new memory."""
        experience = {
            'action': 'solve_problem',
            'description': problem,
            'approach': str(result.get('approach', '')),
            'result': 'success' if result.get('success') else 'failure',
            'lessons_learned': result.get('insights', []),
            'timestamp': datetime.now().isoformat()
        }

        self.memory.add_experience(experience)

# Usage
solver = MemoryAugmentedSolver(memory)
result = await solver.solve_problem(
    "How do I optimize a slow database query that's causing timeouts?"
)
print(f"Solution completed using {result['memories_used']} relevant memories")
```

## 5. Performance Optimization Techniques

### Batch Processing for High-Throughput Applications

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

class OptimizedMemorySystem(MemorySystem):
    """Memory system optimized for high-throughput applications."""

    def __init__(self, *args, batch_size=50, num_workers=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    async def add_experiences_batch_async(self, experiences: List[Dict[str, Any]]) -> List[str]:
        """Add experiences in optimized batches asynchronously."""
        if not experiences:
            return []

        # Split into batches
        batches = [
            experiences[i:i + self.batch_size]
            for i in range(0, len(experiences), self.batch_size)
        ]

        # Process batches concurrently
        tasks = [
            asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._process_batch,
                batch
            )
            for batch in batches
        ]

        # Gather results
        batch_results = await asyncio.gather(*tasks)
        return [item for batch in batch_results for item in batch]

    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[str]:
        """Process a single batch of experiences."""
        try:
            return self.add_trajectory_batch(
                batch,
                use_parallel=True,
                max_workers=min(self.num_workers, len(batch))
            )
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return []

    async def query_memory_optimized(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """Query multiple queries concurrently."""
        tasks = [
            asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda q: self.query_memory(q, top_k=top_k),
                query
            )
            for query in queries
        ]

        return await asyncio.gather(*tasks)

# Usage
optimized_memory = OptimizedMemorySystem(config, batch_size=100, num_workers=8)

# Add large batch of experiences efficiently
experiences = [...]  # Large list of experiences
start_time = time.time()
memory_ids = await optimized_memory.add_experiences_batch_async(experiences)
end_time = time.time()

print(f"Added {len(memory_ids)} experiences in {end_time - start_time:.2f} seconds")

# Query multiple topics concurrently
queries = [
    "database optimization techniques",
    "debugging memory leaks",
    "API design patterns",
    "machine learning deployment"
]

start_time = time.time()
all_results = await optimized_memory.query_memory_optimized(queries, top_k=3)
end_time = time.time()

print(f"Processed {len(queries)} queries in {end_time - start_time:.2f} seconds")
for query, results in zip(queries, all_results):
    print(f"{query}: {len(results)} results")
```

This tutorial demonstrates advanced patterns for building sophisticated memory-augmented applications. The techniques shown here can be combined and extended based on your specific use case requirements.