import logging
import re
import logging
from typing import Any, Dict, List

from .logging_manager import LoggingManager

logger = LoggingManager.get_logger(__name__)


class ResponseQualityScorer:
    """Independent quality scoring system for AI responses with parity between reasoning and non-reasoning models."""

    def __init__(
            self,
            debug: bool = False,
            min_threshold: float = 0.1,
            bias_correction: bool = True):
        self.debug = debug
        self.min_threshold = min_threshold
        self.bias_correction = bias_correction
        self.model_performance_cache = {
            'direct_models': {'avg_score': 0.72, 'count': 0},
            'reasoning_models': {'avg_score': 0.78, 'count': 0}
        }

    def calculate_response_quality(
        self,
        response: Dict[str, Any],
        context: Dict[str, Any],
        query: str
    ) -> float:
        """
        Calculate response quality with parity between reasoning and non-reasoning models.

        Args:
            response: Response dictionary from AI model
            context: Request context including query and metadata
            query: Original user query

        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            # Extract content
            assistant_content = response.get("content", "").strip()
            reasoning_content = response.get("reasoning_content", "").strip()
            has_reasoning = bool(reasoning_content)

            if self.debug:
                logger.info(
                    f"Scoring response: content_length={
                        len(assistant_content)}, " f"reasoning_length={
                        len(reasoning_content)}, has_reasoning={has_reasoning}")

            # Step 1: Base answer quality evaluation (applied to ALL responses)
            base_quality = self._evaluate_answer_content(assistant_content, context, query)

            # Step 2: Adaptive scoring based on response type
            if has_reasoning:
                final_quality = self._evaluate_reasoning_response(
                    assistant_content, reasoning_content, base_quality, context, query)
            else:
                final_quality = self._evaluate_direct_response(
                    assistant_content, base_quality, context, query)

            # Step 3: Apply parity adjustments to prevent model-type bias
            normalized_quality = self._apply_parity_adjustment(final_quality, has_reasoning)

            if self.debug:
                logger.info(f"Quality scoring: base={base_quality:.3f}, "
                            f"final={final_quality:.3f}, normalized={normalized_quality:.3f}, "
                            f"has_reasoning={has_reasoning}")

            return max(0.0, min(1.0, normalized_quality))

        except Exception as e:
            logger.error(f"Error calculating response quality: {e}")
            return 0.5  # Neutral fallback

    def _evaluate_answer_content(
        self,
        content: str,
        context: Dict[str, Any],
        query: str
    ) -> float:
        """Core answer quality evaluation - applied equally to all responses."""

        if not content:
            return 0.0

        answer_factors = {
            'accuracy_completeness': self._assess_answer_completeness(content, query),
            'actionability': self._assess_practical_value(content),
            'clarity': self._assess_explanation_clarity(content),
            'technical_depth': self._assess_domain_expertise(content),
            'context_relevance': self._assess_query_alignment(content, query),
            'semantic_density': self._calculate_semantic_density(content)
        }

        # Weighted combination
        base_score = (
            answer_factors['accuracy_completeness'] * 0.25 +
            answer_factors['actionability'] * 0.20 +
            answer_factors['clarity'] * 0.20 +
            answer_factors['technical_depth'] * 0.15 +
            answer_factors['context_relevance'] * 0.15 +
            answer_factors['semantic_density'] * 0.05
        )

        return max(0.0, min(1.0, base_score))

    def _evaluate_reasoning_response(
        self,
        answer_content: str,
        reasoning_content: str,
        base_quality: float,
        context: Dict[str, Any],
        query: str
    ) -> float:
        """Evaluate response with reasoning content."""

        reasoning_factors = {
            'problem_understanding': self._check_query_analysis(reasoning_content, query),
            'step_by_step_logic': self._detect_reasoning_structure(reasoning_content),
            'consideration_of_alternatives': self._check_alternative_scenarios(reasoning_content),
            'error_identification': self._detect_error_analysis(reasoning_content),
            'meta_cognition': self._detect_about_reasoning(reasoning_content)
        }

        reasoning_score = sum(reasoning_factors.values()) / len(reasoning_factors) * 0.6

        # Reasoning-answer consistency (capped)
        consistency_score = self._evaluate_reasoning_answer_consistency(
            reasoning_content, answer_content)

        # Combine: 70% answer, 20% reasoning, 10% consistency
        reasoning_quality_score = (
            base_quality * 0.7 +           # Core answer quality
            reasoning_score * 0.2 +           # Quality of reasoning process
            consistency_score * 0.1            # Alignment between reasoning and answer
        )

        return reasoning_quality_score

    def _evaluate_direct_response(
        self,
        answer_content: str,
        base_quality: float,
        context: Dict[str, Any],
        query: str
    ) -> float:
        """Evaluate direct response without reasoning."""

        # Direct responses get full weight on answer quality
        # Plus small bonus for conciseness and directness
        conciseness_bonus = self._calculate_conciseness_bonus(answer_content)
        directness_bonus = self._calculate_directness_bonus(answer_content, query)

        enhanced_score = base_quality + conciseness_bonus + directness_bonus

        # Cap to maintain parity with reasoning responses
        return min(enhanced_score, 0.95)

    def _apply_parity_adjustment(
        self,
        raw_score: float,
        has_reasoning: bool
    ) -> float:
        """Apply adjustments to prevent model-type bias."""

        # Update performance tracking
        model_type = 'reasoning_models' if has_reasoning else 'direct_models'
        stats = self.model_performance_cache[model_type]
        stats['count'] += 1

        # Update rolling average
        alpha = 0.1  # Learning rate
        stats['avg_score'] = (alpha * raw_score + (1 - alpha) * stats['avg_score'])

        # Calculate bias adjustment factor
        reasoning_avg = self.model_performance_cache['reasoning_models']['avg_score']
        direct_avg = self.model_performance_cache['direct_models']['avg_score']

        if abs(reasoning_avg - direct_avg) < 0.05:
            # Minimal bias - no adjustment needed
            return raw_score

        if has_reasoning:
            # Reasoning models tend to score higher, apply slight reduction
            bias_factor = 0.95  # 5% reduction
        else:
            # Direct models tend to score lower, apply slight boost
            bias_factor = 1.05  # 5% boost

        return raw_score * bias_factor

    def _assess_answer_completeness(self, content: str, query: str) -> float:
        """How completely does the answer address the query?"""
        query_entities = self._extract_query_entities(query)
        response_coverage = sum(1 for entity in query_entities if entity.lower() in content.lower())

        # Check for hedging and incomplete responses
        hedging_words = ['maybe', 'perhaps', 'might', 'could', 'possibly', 'depends']
        hedging_penalty = sum(0.05 for word in hedging_words if word in content.lower())

        coverage_score = min(1.0, response_coverage / max(1, len(query_entities)))
        completeness_score = coverage_score * (1.0 - min(0.3, hedging_penalty))

        return completeness_score

    def _assess_practical_value(self, content: str) -> float:
        """Assess how actionable and practical the answer is."""
        actionable_indicators = [
            'step', 'method', 'procedure', 'implementation', 'code', 'example',
            'solution', 'approach', 'technique', 'process', 'algorithm'
        ]

        actionable_count = sum(
            1 for indicator in actionable_indicators if indicator in content.lower())
        word_count = len(content.split())

        if word_count == 0:
            return 0.0

        actionable_density = actionable_count / word_count
        # More generous scoring - base score + density bonus
        base_score = 0.3  # Everyone gets baseline for trying
        density_bonus = min(0.7, actionable_density * 10.0)  # More generous scaling
        return min(1.0, base_score + density_bonus)

    def _assess_explanation_clarity(self, content: str) -> float:
        """Assess how clear and well-structured the explanation is."""
        structure_indicators = {
            'has_examples': self._contains_examples(content),
            'step_by_step': self._has_numbered_steps(content),
            'bullet_points': self._has_bullet_points(content),
            'clear_transitions': self._has_transitional_phrases(content)
        }

        structure_score = sum(structure_indicators.values()) / len(structure_indicators)

        # More generous baseline scoring with structure bonuses
        base_clarity = 0.4  # Everyone gets baseline for clarity
        structure_bonus = structure_score * 0.4

        # Mild length penalties only for extremely long content
        word_count = len(content.split())
        length_penalty = 0.0
        if word_count > 500:  # Increased threshold
            length_penalty = 0.1

        return min(1.0, base_clarity + structure_bonus - length_penalty)

    def _assess_domain_expertise(self, content: str) -> float:
        """Assess technical depth and domain knowledge."""
        technical_indicators = [
            'algorithm', 'function', 'method', 'parameter', 'variable', 'syntax',
            'protocol', 'architecture', 'implementation', 'optimization', 'complexity'
        ]

        tech_count = sum(1 for indicator in technical_indicators if indicator in content.lower())
        # More generous scoring with baseline
        base_tech = 0.3  # Baseline for any technical content
        tech_bonus = min(0.5, tech_count * 0.2)  # More generous scaling
        return min(1.0, base_tech + tech_bonus)

    def _assess_query_alignment(self, content: str, query: str) -> float:
        """How well does response address specific query aspects?"""
        query_words = set(query.lower().split())
        response_words = set(content.lower().split())

        # Calculate semantic overlap
        overlap = len(query_words & response_words)
        overlap_ratio = overlap / max(1, len(query_words))

        # More generous baseline with overlap bonus
        baseline_alignment = 0.4  # Everyone gets baseline for addressing query
        overlap_bonus = min(0.4, overlap_ratio * 0.8)

        # Mild evasion penalties
        evasion_indicators = ['it depends', 'varies', 'not sure', 'hard to say']
        evasion_penalty = sum(
            0.1 for indicator in evasion_indicators if indicator in content.lower())

        alignment_score = baseline_alignment + overlap_bonus - min(0.2, evasion_penalty)
        return max(0.2, min(1.0, alignment_score))

    def _calculate_semantic_density(self, content: str) -> float:
        """Calculate information density - meaningful concepts per word."""
        if not content:
            return 0.0

        # Extract meaningful terms (nouns, verbs, technical terms)
        words = content.lower().split()
        meaningful_words = self._extract_meaningful_words(words)

        if len(words) == 0:
            return 0.0

        density = len(meaningful_words) / len(words)
        return min(1.0, density)

    def _check_query_analysis(self, reasoning: str, query: str) -> float:
        """Does reasoning show understanding of the query?"""
        query_words = set(query.lower().split())
        reasoning_words = set(reasoning.lower().split())

        # Check for restatement and analysis
        restatement_score = len(query_words & reasoning_words) / max(1, len(query_words))

        # Look for analysis indicators
        analysis_indicators = ['analyze', 'consider', 'examine', 'evaluate', 'break down']
        analysis_bonus = 0.2 if any(indicator in reasoning.lower()
                                    for indicator in analysis_indicators) else 0.0

        return min(1.0, restatement_score * 0.8 + analysis_bonus)

    def _detect_reasoning_structure(self, reasoning: str) -> float:
        """Detect structured, step-by-step reasoning."""
        structure_indicators = {
            'numbered_steps': self._has_numbered_steps(reasoning),
            'causal_language': self._has_causal_connectors(reasoning),
            'conditional_logic': self._has_conditional_statements(reasoning),
            'problem_decomposition': self._has_problem_breakdown(reasoning)
        }

        return sum(structure_indicators.values()) / len(structure_indicators)

    def _check_alternative_scenarios(self, reasoning: str) -> float:
        """Does reasoning consider multiple approaches?"""
        alternative_indicators = [
            'however', 'alternatively', 'another approach', 'different method',
            'could also', 'or we could', 'another option'
        ]

        alt_count = sum(1 for indicator in alternative_indicators if indicator in reasoning.lower())
        return min(1.0, alt_count * 0.3)

    def _detect_error_analysis(self, reasoning: str) -> float:
        """Does reasoning include error identification or debugging?"""
        error_indicators = [
            'error', 'mistake', 'issue', 'problem', 'bug', 'exception',
            'debug', 'troubleshoot', 'diagnose', 'identify', 'incorrect'
        ]

        error_count = sum(1 for indicator in error_indicators if indicator in reasoning.lower())
        return min(1.0, error_count * 0.4)

    def _detect_about_reasoning(self, reasoning: str) -> float:
        """Detect meta-cognition about the reasoning process."""
        meta_indicators = [
            'thinking', 'reasoning', 'analysis', 'consider', 'let me think',
            'step by step', 'first', 'second', 'finally'
        ]

        meta_count = sum(1 for indicator in meta_indicators if indicator in reasoning.lower())
        return min(1.0, meta_count * 0.2)

    def _evaluate_reasoning_answer_consistency(self, reasoning: str, answer: str) -> float:
        """Check if answer logically follows from reasoning."""
        reasoning_words = set(reasoning.lower().split())
        answer_words = set(answer.lower().split())

        # Look for conclusion indicators
        conclusion_words = ['therefore', 'thus', 'so', 'consequently', 'as a result']
        has_conclusion = any(word in reasoning.lower() for word in conclusion_words)

        # Basic consistency check
        overlap_ratio = len(reasoning_words & answer_words) / max(1, len(reasoning_words))

        consistency_score = overlap_ratio * 0.7 + (0.3 if has_conclusion else 0.0)
        return min(1.0, consistency_score)

    def _calculate_conciseness_bonus(self, content: str) -> float:
        """Reward concise, to-the-point answers."""
        word_count = len(content.split())

        if word_count <= 20:
            return 0.05  # Bonus for very concise
        elif word_count <= 50:
            return 0.02  # Small bonus for reasonably concise
        else:
            return 0.0  # No penalty for longer content

    def _calculate_directness_bonus(self, content: str, query: str) -> float:
        """Reward direct addressing of query."""
        # Check if content directly answers without excessive preamble
        direct_indicators = ['the answer is', 'solution is', 'result is']
        has_direct_answer = any(indicator in content.lower() for indicator in direct_indicators)

        return 0.03 if has_direct_answer else 0.0

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract key entities/concepts from query."""
        # Simple noun and keyword extraction
        words = query.lower().split()
        entities = [word for word in words if len(word) > 2 and word.isalpha()]
        return list(set(entities))

    def _extract_meaningful_words(self, words: List[str]) -> List[str]:
        """Extract meaningful content words."""
        meaningful = []
        stop_words = {
            'the',
            'a',
            'an',
            'and',
            'or',
            'but',
            'in',
            'on',
            'at',
            'to',
            'for',
            'of',
            'with',
            'by'}

        for word in words:
            if (len(word) > 2 and word.isalpha() and word.lower() not in stop_words):
                meaningful.append(word)

        return meaningful

    def _contains_examples(self, content: str) -> bool:
        """Check if content includes examples."""
        example_indicators = ['for example', 'such as', 'like', 'instance', 'e.g.']
        return any(indicator in content.lower() for indicator in example_indicators)

    def _has_numbered_steps(self, content: str) -> bool:
        """Check for numbered or bulleted steps."""
        pattern = r'\b\d+[.\)]|\b[0-9]+[.\)]|\b(?:first|second|third|fourth|fifth)'
        return bool(re.search(pattern, content, re.IGNORECASE))

    def _has_bullet_points(self, content: str) -> bool:
        """Check for bullet points or lists."""
        bullet_pattern = r'[•\-\*]\s|^\s*[-\*]\s'
        return bool(re.search(bullet_pattern, content, re.MULTILINE))

    def _has_transitional_phrases(self, content: str) -> bool:
        """Check for logical transitions."""
        transitions = ['however', 'therefore', 'moreover', 'furthermore', 'consequently']
        return any(transition in content.lower() for transition in transitions)

    def _has_causal_connectors(self, content: str) -> bool:
        """Check for causal reasoning connectors."""
        causal_words = ['because', 'since', 'due to', 'as a result', 'leads to', 'causes']
        return any(word in content.lower() for word in causal_words)

    def _has_conditional_statements(self, content: str) -> bool:
        """Check for conditional logic."""
        conditionals = ['if', 'when', 'assuming', 'provided that', 'in case']
        return any(conditional in content.lower() for conditional in conditionals)

    def _has_problem_breakdown(self, content: str) -> bool:
        """Check for problem decomposition."""
        breakdown = [
            'break down',
            'first we need',
            'let me consider',
            'analyze the',
            'identify the']
        return any(phrase in content.lower() for phrase in breakdown)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current model performance statistics."""
        return {
            'reasoning_models': self.model_performance_cache['reasoning_models'],
            'direct_models': self.model_performance_cache['direct_models'],
            'bias_gap': abs(
                self.model_performance_cache['reasoning_models']['avg_score'] -
                self.model_performance_cache['direct_models']['avg_score']
            )
        }

    def reset_performance_cache(self) -> None:
        """Reset performance tracking for testing."""
        self.model_performance_cache = {
            'direct_models': {'avg_score': 0.72, 'count': 0},
            'reasoning_models': {'avg_score': 0.78, 'count': 0}
        }
