from typing import Dict, List, Any, Optional, Callable
from .base import RetrievalStrategy, RetrievalResult
import json
import logging


class LLMGuidedRetrievalStrategy(RetrievalStrategy):
    """LLM-guided retrieval strategy that uses LLM reasoning to improve retrieval."""

    def __init__(
        self,
        llm_client_callable: Callable[[str], str],
        base_strategy: RetrievalStrategy,
        reasoning_temperature: float = 0.3,
        max_reasoning_tokens: int = 256
    ):
        """Initialize LLM-guided retrieval strategy.

        Args:
            llm_client_callable: Function that takes a prompt string and returns LLM response
            base_strategy: Underlying retrieval strategy to use (e.g., semantic, hybrid)
            reasoning_temperature: Temperature for LLM reasoning calls
            max_reasoning_tokens: Maximum tokens for reasoning responses
        """
        self.llm_call = llm_client_callable
        self.base_strategy = base_strategy
        self.reasoning_temperature = reasoning_temperature
        self.max_reasoning_tokens = max_reasoning_tokens
        self.logger = logging.getLogger(__name__)

    def retrieve(
        self,
        query: str,
        storage_backend,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve memory units with LLM-guided reasoning."""
        # Step 1: Use LLM to analyze query and generate retrieval guidance
        guidance = self._get_retrieval_guidance(query, storage_backend)

        # Step 2: Apply base strategy with enhanced parameters
        enhanced_filters = self._enhance_filters(filters, guidance)
        expanded_top_k = min(top_k * 2, 50)  # Get more candidates for reranking

        candidates = self.base_strategy.retrieve(
            query=guidance.get("enhanced_query", query),
            storage_backend=storage_backend,
            top_k=expanded_top_k,
            filters=enhanced_filters
        )

        # Step 3: Use LLM to rerank and select best results
        if candidates and len(candidates) > top_k:
            final_results = self._llm_rerank(query, candidates, top_k)
        else:
            final_results = candidates[:top_k]

        return final_results

    def _get_retrieval_guidance(self, query: str, storage_backend) -> Dict[str, Any]:
        """Use LLM to generate retrieval guidance for the query."""
        # Get sample memory units to inform the LLM
        sample_units = storage_backend.retrieve_all()[:5]  # Sample a few units

        prompt = f"""Analyze this query and provide guidance for retrieving relevant memory units:

Query: {query}

Sample memory units in the system:
{json.dumps([{"type": u.get("type"), "content": u.get("content", "")[:100] + "...",
              "tags": u.get("tags", [])} for u in sample_units], indent=2)}

Provide retrieval guidance in JSON format:
{{
  "enhanced_query": "improved search query considering context",
  "preferred_types": ["lesson", "skill", "tool", "abstraction"],
  "focus_areas": ["specific topics or aspects to prioritize"],
  "exclusion_criteria": ["what to avoid or filter out"]
}}

Keep the response concise and focused."""

        try:
            response = self.llm_call(prompt)
            guidance = json.loads(response.strip())
            self.logger.debug(f"LLM retrieval guidance: {guidance}")
            return guidance
        except Exception as e:
            self.logger.warning(f"Failed to get LLM guidance: {e}. Using fallback.")
            return {
                "enhanced_query": query,
                "preferred_types": ["lesson", "skill", "tool", "abstraction"],
                "focus_areas": [],
                "exclusion_criteria": []
            }

    def _enhance_filters(self, base_filters: Optional[Dict[str, Any]],
                        guidance: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enhance filters based on LLM guidance."""
        if not base_filters:
            base_filters = {}

        # Add type preferences from guidance
        if "preferred_types" in guidance and guidance["preferred_types"]:
            base_filters["types"] = guidance["preferred_types"]

        return base_filters if base_filters else None

    def _llm_rerank(self, original_query: str, candidates: List[RetrievalResult],
                   top_k: int) -> List[RetrievalResult]:
        """Use LLM to rerank and select the most relevant results."""
        if len(candidates) <= top_k:
            return candidates

        # Prepare candidates for LLM evaluation
        candidate_summaries = []
        for i, candidate in enumerate(candidates):
            candidate_summaries.append({
                "index": i,
                "type": candidate.unit.get("type", "unknown"),
                "content": candidate.unit.get("content", "")[:200] + "...",
                "tags": candidate.unit.get("tags", []),
                "score": candidate.score
            })

        prompt = f"""Given the original query and candidate memory units, select the top {top_k} most relevant ones:

Original Query: {original_query}

Candidate Memory Units:
{json.dumps(candidate_summaries, indent=2)}

Return a JSON array of the indices of the top {top_k} most relevant units, ranked by relevance.
Example: [0, 3, 1, 2, 4]

Consider:
- Relevance to the query
- Actionability and usefulness
- Type appropriateness (tools for technical problems, lessons for general insights, etc.)"""

        try:
            response = self.llm_call(prompt)
            selected_indices = json.loads(response.strip())

            # Validate and filter indices
            valid_indices = [i for i in selected_indices if 0 <= i < len(candidates)][:top_k]

            selected_results = [candidates[i] for i in valid_indices]
            self.logger.debug(f"LLM reranked to {len(selected_results)} results")
            return selected_results

        except Exception as e:
            self.logger.warning(f"LLM reranking failed: {e}. Using original ranking.")
            return candidates[:top_k]

    def retrieve_by_ids(
        self,
        unit_ids: List[str],
        storage_backend
    ) -> List[RetrievalResult]:
        """Retrieve specific memory units by their IDs (delegates to base strategy)."""
        return self.base_strategy.retrieve_by_ids(unit_ids, storage_backend)

    def search(
        self,
        query: str,
        storage_backend,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Search for memory units (same as retrieve for LLM-guided strategy)."""
        return self.retrieve(query, storage_backend, top_k, filters)

    def count_relevant(
        self,
        query: str,
        storage_backend,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count relevant memory units (delegates to base strategy)."""
        return self.base_strategy.count_relevant(query, storage_backend, filters)