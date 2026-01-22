"""
Middleware for integrating memory functionality into API requests.
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import streaming extraction function
from ..utils import extract_final_from_stream

# Configure middleware-specific logging
middleware_enable = os.getenv('MEMEVOLVE_LOG_MIDDLEWARE_ENABLE', 'false').lower() == 'true'
logs_dir = os.getenv('MEMEVOLVE_LOGS_DIR', './logs')
middleware_dir = os.getenv('MEMEVOLVE_LOG_MIDDLEWARE_DIR', logs_dir)

middleware_logger = logging.getLogger("middleware")
middleware_logger.setLevel(logging.INFO)

if middleware_enable:
    os.makedirs(middleware_dir, exist_ok=True)
    middleware_handler = logging.FileHandler(os.path.join(middleware_dir, 'middleware.log'))
    middleware_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    middleware_logger.addHandler(middleware_handler)

logger = middleware_logger


class MemoryMiddleware:
    """Middleware to handle memory integration for API requests."""

    def __init__(
        self,
        memory_system: Optional[Any] = None,
        evolution_manager: Optional[Any] = None
    ):
        self.memory_system = memory_system
        self.evolution_manager = evolution_manager
        self.process_request_count = 0
        self.process_response_count = 0

    async def process_request(
        self, path: str, method: str, body: bytes, headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Process incoming request and add memory context if applicable.

        Args:
            path: Request path
            method: HTTP method
            body: Request body
            headers: Request headers

        Returns:
            Modified request data with memory context
        """
        self.process_request_count += 1

        if not self.memory_system or method != "POST" or not path.endswith("chat/completions"):
            return {"body": body, "headers": headers}

        try:
            # Parse the request body
            request_data = json.loads(body)

            # Extract conversation context
            messages = request_data.get("messages", [])
            if not messages:
                return {"body": body, "headers": headers}

            # Create a query from the conversation
            query = self._extract_conversation_context(messages)

            if query:
                # Retrieve relevant memories with timing
                import time
                start_time = time.time()
                memories = self.memory_system.query_memory(
                    query=query, top_k=5)
                retrieval_time = time.time() - start_time

                # Record retrieval metrics for evolution
                if self.evolution_manager:
                    success = len(memories) > 0
                    self.evolution_manager.record_memory_retrieval(
                        retrieval_time, success)

                if memories:
                    # Add memories to the system prompt or context
                    enhanced_messages = self._inject_memories(
                        messages, memories)
                    request_data["messages"] = enhanced_messages

                    logger.info(
                        f"Injected {len(memories)} memories into request")

            # Return modified request
            return {
                "body": json.dumps(request_data).encode(),
                "headers": headers,
                "original_query": query
            }

        except Exception as e:
            logger.error(f"Error processing request for memory integration: {e}")
            return {"body": body, "headers": headers}

    async def process_response(self, path: str, method: str, request_body: bytes, response_body: bytes, request_context: Dict[str, Any]):
        """
        Process response and encode new experiences into memory.

        Args:
            path: Request path
            method: HTTP method
            request_body: Original request body
            response_body: Response body
            request_context: Context from request processing
        """
        if not self.memory_system or method != "POST" or not path.endswith("chat/completions"):
            return

        self.process_response_count += 1

        try:
            logger.info(f"process_response called, response_body length: {len(response_body)}")
            if len(response_body) == 0:
                logger.info("response_body is empty, skipping")
                return

            # Log first 500 characters for debugging
            logger.info(f"response_body preview: {response_body[:500].decode('utf-8', errors='ignore')}")

            # Parse request and response
            request_data = json.loads(request_body)

            # Check if response looks like JSON (starts with '{' or '[')
            response_str = response_body.decode('utf-8', errors='ignore').strip()
            if not response_str or not (response_str.startswith('{') or response_str.startswith('[')):
                # Check if this is a streaming response that needs extraction
                if response_str.startswith('data: '):
                    logger.info("Detected streaming response in experience processing, extracting final result")
                    extracted = extract_final_from_stream(response_str)
                    if isinstance(extracted, str):
                        response_body = extracted.encode('utf-8')
                    else:
                        response_body = extracted
                    logger.info(f"Extracted final response for encoding, length: {len(response_body)}")
                else:
                    logger.error(f"Response doesn't look like JSON or streaming. Content: {response_str[:500]}")
                    return

            try:
                response_data = json.loads(response_body)
                logger.info(f"Parsed response successfully - has choices: {'choices' in response_data}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse response as JSON: {e}")
                logger.error(f"Response content: {response_body[:1000].decode('utf-8', errors='ignore')}")
                return

            # Extract the conversation
            messages = request_data.get("messages", [])
            logger.info(f"Extracted {len(messages)} messages from request")

            # Handle both streaming and non-streaming response formats
            choice = response_data.get("choices", [{}])[0]
            assistant_response = {}

            # For streaming responses, content might be in delta instead of message
            if "delta" in choice:
                # Streaming format
                delta = choice.get("delta", {})
                assistant_response = {
                    "role": delta.get("role", "assistant"),
                    "content": delta.get("content", ""),
                    "reasoning_content": delta.get("reasoning_content", "")
                }
                logger.info(f"Streaming response - delta keys: {list(delta.keys())}")
            elif "message" in choice:
                # Non-streaming format
                assistant_response = choice.get("message", {})
                logger.info(f"Non-streaming response - message keys: {list(assistant_response.keys())}")
            else:
                logger.warning(f"Unknown response format - choice keys: {list(choice.keys())}")

            logger.info(f"Final assistant response keys: {list(assistant_response.keys())}")
            logger.info(f"Assistant content length: {len(assistant_response.get('content', ''))}")

            if messages and assistant_response:
                # Check if assistant has meaningful content
                assistant_content = assistant_response.get("content", "").strip()
                reasoning_content = assistant_response.get("reasoning_content", "").strip()

                logger.info(f"Processing response - content: {len(assistant_content)}, reasoning: {len(reasoning_content)}")

                if not assistant_content and not reasoning_content:
                    logger.info("Skipping experience creation - no content in response")
                    return

                logger.info(f"Creating experience from {len(messages)} messages")
                logger.info("About to call _create_experience method")
                # Create experience from the interaction
                experience = self._create_experience(
                    messages, assistant_response, request_context)
                logger.info(f"Experience created: {experience.get('type', 'unknown')}, content length: {len(experience.get('content', ''))}")

                # Encode the experience
                logger.info("Adding experience to memory system")
                logger.info(f"Calling memory_system.add_experience()")
                try:
                    unit_id = self.memory_system.add_experience(experience)
                    logger.info(f"Experience added successfully with ID: {unit_id}")
                except Exception as e:
                    logger.error(f"Failed to add experience: {e}")
                    logger.error(f"Error type: {type(e).__name__}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")

                logger.info("Encoded new experience into memory")

            # Calculate and record response quality for evolution
            if self.evolution_manager:
                try:
                    quality_score = self._calculate_response_quality(
                        request_context, response_body, assistant_response)
                    self.evolution_manager.record_response_quality(quality_score)
                    logger.info(f"Recorded response quality score: {quality_score:.3f}")
                except Exception as e:
                    logger.error(f"Failed to calculate response quality: {e}")

        except Exception as e:
            logger.error(f"Error processing response for memory encoding: {e}")

    def _extract_conversation_context(self, messages: List[Dict[str, Any]]) -> str:
        """Extract context from conversation messages for memory retrieval."""
        # Use the last user message as the primary query
        user_messages = [msg for msg in messages if msg.get("role") == "user"]

        if user_messages:
            return user_messages[-1].get("content", "")

        # Fallback to the last message regardless of role
        if messages:
            return messages[-1].get("content", "")

        return ""

    def _inject_memories(self, messages: List[Dict[str, Any]], memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Inject retrieved memories into the conversation."""
        # Find or create system message
        system_message = None
        other_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg
            else:
                other_messages.append(msg)

        # Create memory context
        memory_context = self._format_memories(memories)

        if system_message:
            # Append to existing system message
            system_message["content"] += f"\n\nRelevant past experiences:\n{memory_context}"
        else:
            # Create new system message with memories
            system_message = {
                "role": "system",
                "content": f"You have access to relevant past experiences:\n{memory_context}"
            }

        # Return messages with system message first
        return [system_message] + other_messages

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories for injection into prompts."""
        formatted = []
        for i, memory in enumerate(memories, 1):
            content = memory.get("content", "")
            score = memory.get("score", 0.0)
            formatted.append(f"{i}. {content} (relevance: {score:.2f})")

        return "\n".join(formatted)

    def _calculate_response_quality(
        self,
        request_context: Dict[str, Any],
        response_body: bytes,
        assistant_response: Dict[str, Any]
    ) -> float:
        """
        Calculate response quality score based on multiple dimensions:
        - Relevance to original query
        - Memory utilization effectiveness
        - Response structure and completeness
        - Content quality and coherence

        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            # Extract response content
            assistant_content = assistant_response.get("content", "").strip()
            reasoning_content = assistant_response.get("reasoning_content", "").strip()
            response_text = assistant_content or reasoning_content

            if not response_text:
                return 0.0

            # Get original query for relevance evaluation
            original_query = request_context.get("original_query", "")

            # Get memories that were injected for utilization evaluation
            injected_memories = []
            if original_query and self.memory_system:
                try:
                    injected_memories = self.memory_system.query_memory(
                        query=original_query, top_k=5)
                except Exception as e:
                    logger.warning(f"Could not retrieve memories for quality calculation: {e}")

            # Calculate quality components
            relevance_score = self._calculate_query_relevance_score(original_query, response_text)
            memory_score = self._calculate_memory_utilization_score(injected_memories, response_text)
            structure_score = self._calculate_response_structure_score(response_body)
            coherence_score = self._calculate_response_coherence_score(response_text)

            # Weighted combination
            quality_score = (
                relevance_score * 0.3 +    # How well response addresses the query
                memory_score * 0.3 +       # How effectively memories are utilized
                structure_score * 0.2 +    # Proper response formatting
                coherence_score * 0.2      # Internal response quality
            )

            return max(0.0, min(1.0, quality_score))  # Clamp to 0-1 range

        except Exception as e:
            logger.error(f"Error calculating response quality: {e}")
            return 0.5  # Neutral score on error

    def _calculate_memory_utilization_score(
        self,
        injected_memories: List[Dict[str, Any]],
        response_text: str
    ) -> float:
        """Calculate how well the response utilizes injected memories."""
        if not injected_memories:
            return 0.5  # Neutral score if no memories were injected

        try:
            # Extract keywords from memory contents
            memory_keywords = set()
            for memory in injected_memories:
                content = memory.get("content", "").lower()
                # Simple keyword extraction: split on spaces and punctuation
                words = content.replace(".", " ").replace(",", " ").replace(";", " ").split()
                # Filter to meaningful words (length > 3, not common stop words)
                meaningful_words = [
                    word for word in words
                    if len(word) > 3 and word not in {"that", "this", "with", "from", "have", "were", "what", "when", "where", "which"}
                ]
                memory_keywords.update(meaningful_words[:10])  # Limit per memory

            if not memory_keywords:
                return 0.5

            # Check for keyword matches in response
            response_lower = response_text.lower()
            matched_keywords = 0

            for keyword in memory_keywords:
                if keyword in response_lower:
                    matched_keywords += 1

            # Score based on percentage of keywords used
            utilization_score = matched_keywords / len(memory_keywords)
            return min(utilization_score, 1.0)

        except Exception as e:
            logger.error(f"Error calculating memory utilization: {e}")
            return 0.5

    def _calculate_response_structure_score(self, response_body: bytes) -> float:
        """Calculate response structure quality."""
        try:
            # Try to parse as JSON
            response_text = response_body.decode('utf-8', errors='ignore')

            # Check for basic JSON structure
            if '"choices"' in response_text and '"message"' in response_text:
                # Looks like a valid OpenAI-style response
                try:
                    json.loads(response_text)
                    return 1.0  # Perfect JSON structure
                except json.JSONDecodeError:
                    return 0.7  # Malformed JSON but has expected fields
            else:
                return 0.3  # Missing expected structure

        except Exception as e:
            logger.error(f"Error calculating response structure: {e}")
            return 0.3



    def _calculate_query_relevance_score(self, original_query: str, response_text: str) -> float:
        """Calculate how relevant the response is to the original query."""
        if not original_query:
            return 0.5  # Neutral if no query available

        try:
            # Extract key terms from query
            query_words = set(original_query.lower().split())
            # Remove common stop words
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "what", "how", "why", "when", "where", "which", "who"}
            query_keywords = query_words - stop_words

            if not query_keywords:
                return 0.5  # Neutral if no meaningful keywords

            # Check for keyword presence in response
            response_lower = response_text.lower()
            matched_keywords = sum(1 for keyword in query_keywords if keyword in response_lower)

            # Calculate relevance score
            relevance_ratio = matched_keywords / len(query_keywords)

            # Bonus for comprehensive responses that address multiple query aspects
            if relevance_ratio >= 0.7:
                return min(relevance_ratio * 1.2, 1.0)  # Bonus for high relevance
            elif relevance_ratio >= 0.4:
                return relevance_ratio  # Good relevance
            else:
                return relevance_ratio * 0.5  # Penalty for low relevance

        except Exception as e:
            logger.error(f"Error calculating query relevance: {e}")
            return 0.5

    def _calculate_response_coherence_score(self, response_text: str) -> float:
        """Calculate response coherence and quality metrics."""
        try:
            base_score = 1.0

            # 1. Length analysis (more nuanced than before)
            length = len(response_text)

            # Different expectations based on response patterns
            if "yes" in response_text.lower() and length < 10:
                # Very short affirmative responses are acceptable
                length_score = 1.0
            elif length < 20:
                # Very short responses get penalty unless they're direct answers
                direct_answer_patterns = ["no", "yes", "true", "false", "correct", "incorrect", "error", "success"]
                if any(pattern in response_text.lower() for pattern in direct_answer_patterns):
                    length_score = 0.8  # Acceptable for direct answers
                else:
                    length_score = 0.3  # Penalty for incomplete responses
            elif 20 <= length <= 50:
                # Short but potentially complete responses
                length_score = 0.7
            elif 50 <= length <= 2000:
                # Good length range
                length_score = 1.0
            elif 2000 <= length <= 4000:
                # Long but acceptable
                length_score = 0.8
            else:
                # Too long - potential rambling
                length_score = 0.4

            # 2. Repetition analysis (harsher penalties)
            words = response_text.lower().split()
            repetition_penalty = 0.0

            if len(words) > 10:
                # Check for exact word repetition (3+ times)
                word_counts = {}
                for word in words:
                    if len(word) > 2:  # Ignore very short words
                        word_counts[word] = word_counts.get(word, 0) + 1

                for word, count in word_counts.items():
                    if count >= 5:  # Very excessive repetition
                        repetition_penalty += 0.4
                    elif count >= 3:  # Moderate repetition
                        repetition_penalty += 0.2

                # Check for phrase repetition (2+ consecutive words repeated 2+ times)
                for phrase_length in [2, 3]:
                    if len(words) > phrase_length * 2:
                        phrases = [" ".join(words[i:i+phrase_length]) for i in range(len(words) - phrase_length + 1)]
                        phrase_counts = {}
                        for phrase in phrases:
                            if len(phrase.split()) == phrase_length:  # Ensure complete phrase
                                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

                        for phrase, count in phrase_counts.items():
                            if count >= 3:  # Phrase repeated 3+ times
                                repetition_penalty += 0.3
                            elif count >= 2 and len(phrase.split()) >= 3:  # 3+ word phrase repeated 2+ times
                                repetition_penalty += 0.2

            # Cap repetition penalty
            repetition_penalty = min(repetition_penalty, 0.8)

            # 3. Completeness analysis
            completeness_penalty = 0.0

            # Check for truncation indicators
            truncation_indicators = ["...", "The response", "The answer", "Based on", "According to"]
            if any(response_text.endswith(indicator) for indicator in truncation_indicators):
                completeness_penalty += 0.3

            # Check for incomplete sentences (ends mid-sentence)
            if response_text.endswith(("The", "It", "This", "That", "A", "An", "One", "Some", "Many")):
                completeness_penalty += 0.2

            # Check for abrupt endings
            last_sentences = response_text.split('.')[-1].strip()
            if last_sentences and len(last_sentences.split()) < 3 and not any(punct in last_sentences for punct in ['?', '!', ':']):
                completeness_penalty += 0.1

            # 4. Information density (quality per character)
            info_density = self._calculate_information_density(response_text)

            # Combine scores
            coherence_score = length_score * 0.4 + (1.0 - repetition_penalty) * 0.3 + (1.0 - completeness_penalty) * 0.2 + info_density * 0.1

            return max(0.0, coherence_score)

        except Exception as e:
            logger.error(f"Error calculating response coherence: {e}")
            return 0.5

    def _calculate_information_density(self, response_text: str) -> float:
        """Calculate information density (useful content per character)."""
        try:
            # Simple heuristic: ratio of meaningful words to total words
            words = response_text.split()
            if not words:
                return 0.0

            # Count meaningful words (longer than 3 chars, not common fillers)
            meaningful_words = 0
            filler_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}

            for word in words:
                word_lower = word.lower().strip('.,!?;:')
                if len(word_lower) > 3 and word_lower not in filler_words:
                    meaningful_words += 1

            density = meaningful_words / len(words)
            return min(density * 2.0, 1.0)  # Scale up but cap at 1.0

        except Exception as e:
            logger.error(f"Error calculating information density: {e}")
            return 0.5

    def _create_experience(self, messages: List[Dict[str, Any]], assistant_response: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Create an experience object from the conversation."""
        # Extract key information
        user_query = context.get("original_query", "")
        assistant_content = assistant_response.get("content", "").strip()
        reasoning_content = assistant_response.get("reasoning_content", "").strip()

        # Use reasoning content if main content is empty
        if not assistant_content and reasoning_content:
            assistant_content = reasoning_content

        # Determine experience type based on content
        experience_type = "conversation"
        if any(keyword in user_query.lower() for keyword in ["code", "function", "implement", "write"]):
            experience_type = "tool"
        elif any(keyword in user_query.lower() for keyword in ["explain", "what", "how", "why"]):
            experience_type = "lesson"
        elif any(keyword in user_query.lower() for keyword in ["remember", "recall", "previous"]):
            experience_type = "abstraction"

        return {
            "type": experience_type,
            "content": f"Q: {user_query}\nA: {assistant_content}",
            "context": {
                "timestamp": datetime.now().isoformat(),
                "messages_count": len(messages),
                "query": user_query
            },
            "tags": self._extract_tags(user_query, assistant_content)
        }

    def _extract_tags(self, query: str, response: str) -> List[str]:
        """Extract relevant tags from the interaction."""
        tags = []

        # Simple keyword-based tagging
        if "code" in query.lower() or "function" in query.lower():
            tags.append("coding")
        if "explain" in query.lower() or "how" in query.lower():
            tags.append("explanation")
        if "error" in query.lower() or "bug" in query.lower():
            tags.append("debugging")

        return tags
