"""
Enhanced Middleware with Comprehensive Endpoint Metrics Tracking
=====================================================

This middleware integrates the new EndpointMetricsCollector throughout the request pipeline
to provide detailed token, timing, and business impact analysis for each endpoint.

Key Features:
- Tracks upstream API calls with token counting
- Tracks memory API calls with retrieval metrics
- Tracks embedding API calls with generation metrics
- Provides complete request pipeline analysis
- Generates endpoint-specific performance insights
"""

import json
import logging
import os
import time
try:
    import tiktoken
except ImportError:
    tiktoken = None
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import new metrics collector
from ..utils.endpoint_metrics_collector import get_endpoint_metrics_collector
from ..utils import extract_final_from_stream
from ..utils.quality_scorer import ResponseQualityScorer

# Configure middleware-specific logging using centralized config


def setup_enhanced_middleware_logging(config):
    """Setup enhanced middleware logging using centralized configuration."""
    component_logging = getattr(config, 'component_logging', None)
    if component_logging:
        middleware_enable = getattr(component_logging, 'middleware_enable', False)
    else:
        middleware_enable = False

    logs_dir = getattr(config, 'logs_dir', './logs')
    middleware_logs_dir = os.path.join(logs_dir, 'middleware')

    middleware_logger = logging.getLogger("enhanced_middleware")
    middleware_logger.setLevel(logging.INFO)

    if middleware_enable:
        os.makedirs(middleware_logs_dir, exist_ok=True)
        middleware_handler = logging.FileHandler(os.path.join(middleware_logs_dir, 'enhanced_middleware.log'))
        middleware_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        middleware_logger.addHandler(middleware_handler)

    return middleware_logger


# Default logger (will be reconfigured when config is available)
logger = logging.getLogger("enhanced_middleware")


class EnhancedMemoryMiddleware:
    """Enhanced middleware with comprehensive endpoint metrics tracking."""

    def __init__(
        self,
        memory_system: Optional[Any] = None,
        evolution_manager: Optional[Any] = None,
        config: Any = None  # Config is required
    ):
        if config is None:
            raise ValueError("Config is required for EnhancedMemoryMiddleware")
        self.memory_system = memory_system
        self.evolution_manager = evolution_manager
        self.config = config

        # Setup logging using centralized config
        global logger
        if config:
            logger = setup_enhanced_middleware_logging(config)
            component_logging = getattr(config, 'component_logging', None)
            if component_logging:
                middleware_enable = getattr(component_logging, 'middleware_enable', False)
            else:
                middleware_enable = False
        else:
            middleware_enable = False

        self.middleware_enable = middleware_enable  # Store as instance variable
        self.process_request_count = 0
        self.quality_scorer = ResponseQualityScorer(debug=middleware_enable)
        self.process_response_count = 0

        # Get endpoint metrics collector
        self.metrics_collector = get_endpoint_metrics_collector(config)

        # Auto-evolution checking
        self.auto_evolution_check_interval = 50
        self.last_auto_check = 0

        # Token counting setup
        self._setup_token_counter()

    def _setup_token_counter(self):
        """Setup token counter for all endpoint models."""
        try:
            if not tiktoken:
                logger.warning("tiktoken not available, using character-based token estimation")
                self.upstream_tokenizer = None
                self.embedding_tokenizer = None
                self.memory_tokenizer = None
                self.tokenizer = None
                return

            # Default to GPT-3.5-turbo encoding
            self.upstream_tokenizer = tiktoken.get_encoding("cl100k_base")
            self.embedding_tokenizer = tiktoken.get_encoding("cl100k_base")
            self.memory_tokenizer = tiktoken.get_encoding("cl100k_base")

            # Setup upstream tokenizer
            if (self.config and hasattr(self.config, 'upstream') and
                    self.config.upstream and self.config.upstream.model):
                model_name = self.config.upstream.model.lower()
                if 'gpt-4' in model_name:
                    self.upstream_tokenizer = tiktoken.get_encoding("cl100k_base")
                elif 'gpt-3.5' in model_name:
                    self.upstream_tokenizer = tiktoken.get_encoding("cl100k_base")
                else:
                    # Silently use default tokenizer for local models (expected behavior)
                    logger.debug(f"Using default tokenizer for local model: {model_name}")

            # Setup embedding tokenizer
            if (self.config and hasattr(self.config, 'embedding') and
                    self.config.embedding and self.config.embedding.model):
                model_name = self.config.embedding.model.lower()
                if 'gpt-4' in model_name:
                    self.embedding_tokenizer = tiktoken.get_encoding("cl100k_base")
                elif 'gpt-3.5' in model_name:
                    self.embedding_tokenizer = tiktoken.get_encoding("cl100k_base")
                else:
                    # For non-GPT embedding models, try to get specific tokenizer first
                    try:
                        self.embedding_tokenizer = tiktoken.encoding_for_model(model_name)
                        logger.debug(
                            f"Using specific tokenizer {
                                type(
                                    self.embedding_tokenizer).__name__} for embedding model {model_name}")
                    except Exception:
                        # Fallback to cl100k_base if specific tokenizer fails
                        self.embedding_tokenizer = tiktoken.get_encoding("cl100k_base")
                        logger.debug(
                            f"Using fallback cl100k_base tokenizer for embedding model {model_name}")

            # Setup memory tokenizer for metrics collection
            if (self.config and hasattr(self.config, 'memory') and
                    self.config.memory and self.config.memory.model):
                model_name = self.config.memory.model.lower()
                if 'gpt-4' in model_name:
                    self.memory_tokenizer = tiktoken.get_encoding("cl100k_base")
                elif 'gpt-3.5' in model_name:
                    self.memory_tokenizer = tiktoken.get_encoding("cl100k_base")
                else:
                    # Use default tokenizer for local models
                    self.memory_tokenizer = tiktoken.get_encoding("cl100k_base")
                    logger.debug(f"Using default tokenizer for memory model: {model_name}")

            # Default tokenizer for backwards compatibility
            self.tokenizer = self.upstream_tokenizer

            # Only log tokenizer setup when there are actual issues or specific
            # configurations worth noting
            if self.config:
                has_embedding_model = (self.config.embedding and self.config.embedding.model and
                                       'gpt' not in self.config.embedding.model.lower())
                has_memory_model = (self.config.memory and self.config.memory.model and
                                    'gpt' not in self.config.memory.model.lower())

                if has_embedding_model or has_memory_model:
                    logger.debug(f"Tokenizer setup complete for all endpoints")
        except Exception as e:
            logger.error(f"Error setting up tokenizer: {e}")
            self.tokenizer = None
            self.upstream_tokenizer = None
            self.embedding_tokenizer = None
            self.memory_tokenizer = None

    def _count_tokens(self, text: str, endpoint_type: str = 'upstream') -> int:
        """Count tokens in text using appropriate tokenizer for endpoint type."""
        # For memory and embedding endpoints, these are actual API calls that should be counted
        # For upstream, we count request/response tokens at the HTTP client level
        if endpoint_type in ['memory', 'embedding']:
            logger.warning(
                f"Token counting for {endpoint_type} endpoint - this may not be accurate for API calls")
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4

        # Use upstream tokenizer for HTTP requests (upstream endpoint)
        tokenizer = self.upstream_tokenizer

        if not tokenizer:
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4

        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens for {endpoint_type}: {e}")
            return len(text) // 4

    async def process_request(
        self, path: str, method: str, body: bytes, headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Process incoming request with comprehensive endpoint tracking.
        """
        self.process_request_count += 1

        if not self.memory_system or method != "POST" or not path.endswith("chat/completions"):
            return {"body": body, "headers": headers}

        try:
            # Parse request body
            request_data = json.loads(body)
            messages = request_data.get("messages", [])

            if not messages:
                return {"body": body, "headers": headers}

            # Create query and count tokens
            query = self._extract_conversation_context(messages)
            query_tokens = self._count_tokens(query)

            # Generate request ID and start tracking
            import uuid
            request_id = str(uuid.uuid4())

            # Start request pipeline tracking
            request_metrics = self.metrics_collector.start_request_tracking(
                request_id=request_id,
                query=query,
                query_tokens=query_tokens
            )

            # Start memory retrieval tracking
            memory_call_id = None
            memories = []
            memory_relevance_scores = []
            memory_retrieval_time = 0.0
            memory_query_tokens = 0

            if query and self.memory_system:
                # Track memory retrieval with token counting
                retrieval_start = time.time()

                # Estimate tokens for memory query
                memory_query_tokens = self._count_tokens(query)
                memory_call_id = self.metrics_collector.start_endpoint_call(
                    request_id=request_id,
                    endpoint_type='memory',
                    input_tokens=memory_query_tokens,
                    request_type='memory_retrieval'
                )

                # Use retrieval.default_top_k from centralized config
                retrieval_limit = self._get_retrieval_limit()
                memories = self.memory_system.query_memory(
                    query=query, top_k=retrieval_limit)
                memory_retrieval_time = time.time() - retrieval_start

                # Calculate relevance scores
                memory_relevance_scores = [mem.get('score', 0.5) for mem in memories]

            # End memory call tracking
            if memory_call_id:
                estimated_output_tokens = max(1, memory_query_tokens // 2)  # Rough estimate
                self.metrics_collector.end_endpoint_call(
                    call_id=memory_call_id,
                    success=True,
                    output_tokens=estimated_output_tokens
                )

                # Record retrieval metrics for evolution
                if self.evolution_manager:
                    success = len(memories) > 0
                    self.evolution_manager.record_memory_retrieval(
                        memory_retrieval_time, success, len(memories))

                    self.evolution_manager.record_api_request(memory_retrieval_time, success=True)

                    # Check auto-evolution triggers
                    if self.process_request_count % self.auto_evolution_check_interval == 0:
                        if self.evolution_manager.check_auto_evolution_triggers():
                            logger.info("Auto-evolution triggers met - starting evolution")
                            self.evolution_manager.start_evolution(auto_trigger=True)

            # Log detailed memory retrieval information
            self._log_memory_retrieval_details(query, memories, memory_retrieval_time)

            if memories:
                # Inject memories into system prompt
                enhanced_messages = self._inject_memories(messages, memories)
                request_data["messages"] = enhanced_messages
                request_metrics.memories_injected = len(memories)

                # Get retrieval limit for logging
                retrieval_limit = (
                    self.config.retrieval.default_top_k
                    if self.config and hasattr(self.config, 'retrieval')
                    else 3
                )
                logger.info(
                    f"Injected {
                        len(memories)} memories into request (limit: {retrieval_limit})")

            # Estimate baseline tokens (what would be used without memory)
            baseline_response_estimate = self._estimate_response_tokens(messages)

            # Count tokens in enhanced request
            enhanced_request_tokens = self._count_tokens(json.dumps(request_data))

            # Start upstream call tracking
            upstream_call_id = self.metrics_collector.start_endpoint_call(
                request_id=request_id,
                endpoint_type='upstream',
                input_tokens=enhanced_request_tokens,
                model=getattr(self.config.upstream, 'model', None) if self.config else None,
                temperature=request_data.get('temperature', 0.7),
                request_type='chat_completion'
            )

            # Store tracking info in request context
            return {
                "body": json.dumps(request_data).encode(),
                "headers": headers,
                "original_query": query,
                "request_id": request_id,
                "upstream_call_id": upstream_call_id,
                "baseline_response_estimate": baseline_response_estimate,
                "enhanced_request_tokens": enhanced_request_tokens,
                "query_tokens": query_tokens
            }

        except Exception as e:
            logger.error(f"Error processing request for memory integration: {e}")
            return {"body": body, "headers": headers}

    async def process_response(
        self,
        path: str,
        method: str,
        request_body: bytes,
        response_body: bytes,
        request_context: Dict[str, Any]
    ) -> None:
        """
        Process response with comprehensive endpoint metrics tracking.
        """
        if not self.memory_system or method != "POST" or not path.endswith("chat/completions"):
            return

        self.process_response_count += 1

        try:
            total_request_start = time.time()

            # Extract tracking information
            request_id = request_context.get("request_id")
            upstream_call_id = request_context.get("upstream_call_id")
            baseline_estimate = request_context.get("baseline_response_estimate", 0)
            enhanced_request_tokens = request_context.get("enhanced_request_tokens", 0)
            query_tokens = request_context.get("query_tokens", 0)

            if not request_id or not upstream_call_id:
                logger.warning("Missing request tracking context")
                return

            logger.info(f"process_response called for request {request_id}")

            if len(response_body) == 0:
                logger.info("response_body is empty, skipping")
                return

            # Handle streaming responses
            response_str = response_body.decode('utf-8', errors='ignore').strip()
            if not response_str or not (
                    response_str.startswith('{') or response_str.startswith('[')):
                if response_str.startswith('data: '):
                    logger.info(
                        "Detected streaming response in experience processing, extracting final result")
                    extracted = extract_final_from_stream(response_str)
                    if isinstance(extracted, str):
                        response_body = extracted.encode('utf-8')
                    else:
                        response_body = extracted
                    logger.info(
                        f"Extracted final response for encoding, length: {
                            len(response_body)}")
                else:
                    logger.error(
                        f"Response doesn't look like JSON or streaming. Content: {response_str[:500]}")
                    return

            # Parse response
            try:
                response_data = json.loads(response_body)
                logger.info(
                    f"Parsed response successfully - has choices: {'choices' in response_data}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse response as JSON: {e}")
                return

            # Extract conversation and response
            request_data = json.loads(request_body)
            messages = request_data.get("messages", [])
            logger.info(f"Extracted {len(messages)} messages from request")

            choice = response_data.get("choices", [{}])[0]
            assistant_response = {}

            # Handle streaming response format
            if "delta" in choice:
                delta = choice.get("delta", {})
                assistant_response = {
                    "role": delta.get("role", "assistant"),
                    "content": delta.get("content", ""),
                    "reasoning_content": delta.get("reasoning_content", "")
                }
            else:
                assistant_response = {
                    "role": choice.get("message", {}).get("role", "assistant"),
                    "content": choice.get("message", {}).get("content", "")
                }

            # Calculate response tokens
            response_content = assistant_response.get("content", "")
            response_tokens = self._count_tokens(response_content)

            # End upstream call tracking with response metrics
            self.metrics_collector.end_endpoint_call(
                call_id=upstream_call_id,
                output_tokens=response_tokens,
                success=True
            )

            # Calculate total request time
            total_request_time_ms = (time.time() - total_request_start) * 1000

            # Calculate business value score
            business_value_score = self._calculate_business_value_score(
                baseline_estimate, enhanced_request_tokens, response_tokens
            )

            # Complete request tracking
            self.metrics_collector.end_request_tracking(
                request_id=request_id,
                response_tokens=response_tokens,
                total_time_ms=total_request_time_ms,
                business_value_score=business_value_score
            )

            # Continue with experience encoding (existing logic)
            await self._encode_experience(
                messages, assistant_response, request_context, response_tokens
            )

        except Exception as e:
            logger.error(f"Error in enhanced response processing: {e}")
            import traceback
            traceback.print_exc()

    def _estimate_response_tokens(self, messages: List[Dict]) -> int:
        """Estimate response tokens without memory injection."""
        # Simple heuristic: response typically 2-3x input length
        total_input_tokens = sum(self._count_tokens(msg.get("content", "")) for msg in messages)
        return int(total_input_tokens * 2.5)

    def _calculate_business_value_score(
        self, baseline_estimate: int, enhanced_tokens: int, response_tokens: int
    ) -> float:
        """Calculate business value score (0-1 scale)."""
        # Token efficiency component
        if baseline_estimate > 0:
            token_efficiency = min(
                1.0, max(
                    0.0, (baseline_estimate - enhanced_tokens) / baseline_estimate))
        else:
            token_efficiency = 0.5

        # Response quality component (simplified)
        response_quality = min(1.0, response_tokens / max(baseline_estimate, 100))

        # Weighted combination
        business_value = token_efficiency * 0.7 + response_quality * 0.3
        return min(1.0, max(0.0, business_value))

    async def _encode_experience(
        self, messages: List[Dict], assistant_response: Dict[str, str],
        request_context: Dict[str, Any], response_tokens: int
    ):
        """Encode new experience with enhanced tracking."""
        try:
            # Extract tracking info
            request_id = request_context.get("request_id")
            query = request_context.get("original_query", "")
            query_tokens = request_context.get("query_tokens", 0)

            # Create experience data
            conversation_context = self._build_conversation_context(messages)
            memory_injected = request_context.get("memories_injected", 0)

            # Determine experience type based on content (same as original middleware)
            user_query = query
            assistant_content = assistant_response.get("content", "")
            experience_type = "conversation"
            if any(keyword in user_query.lower()
                   for keyword in ["code", "function", "implement", "write"]):
                experience_type = "tool"
            elif any(keyword in user_query.lower() for keyword in ["explain", "what", "how", "why"]):
                experience_type = "lesson"
            elif any(keyword in user_query.lower() for keyword in ["remember", "recall", "previous"]):
                experience_type = "abstraction"

            # Extract tags (same as original middleware)
            tags = []
            if "code" in user_query.lower() or "function" in user_query.lower():
                tags.append("coding")
            if "explain" in user_query.lower() or "how" in user_query.lower():
                tags.append("explanation")
            if "error" in user_query.lower() or "bug" in user_query.lower():
                tags.append("debugging")

            experience_data = {
                "type": experience_type,
                "content": f"Q: {user_query}\nA: {assistant_content}",  # <- SIMPLE Q&A FORMAT
                "context": {
                    "timestamp": datetime.now().isoformat(),
                    "messages_count": len(messages),
                    "query": user_query,
                    "response_tokens": response_tokens,
                    "query_tokens": query_tokens,
                    "memory_injected": memory_injected,
                    "request_id": request_id,
                    "quality_score": self.quality_scorer.calculate_response_quality(
                        assistant_response, {"query": query}, query
                    )
                },
                "tags": tags
            }

            # Encode experience with timing
            encode_start = time.time()

            # Track embedding generation if needed
            embedding_call_id = None
            embedding_query_tokens = 0
            if self.memory_system and memory_injected > 0:
                # Track embedding generation for memory items
                embedding_query_tokens = sum(
                    self._count_tokens(mem.get('content', ''))
                    for mem in conversation_context.get('memories_used', [])
                )

                embedding_call_id = self.metrics_collector.start_endpoint_call(
                    request_id=request_id or "",
                    endpoint_type='embedding',
                    input_tokens=embedding_query_tokens,
                    request_type='memory_encoding'
                )

            # Encode experience
            if self.memory_system:
                self.memory_system.add_experience(experience_data)

                # End embedding tracking
                if embedding_call_id:
                    self.metrics_collector.end_endpoint_call(
                        call_id=embedding_call_id,
                        success=True,
                        output_tokens=embedding_query_tokens  # Estimate
                    )

            encode_time = time.time() - encode_start

            # Record for evolution - use available record_api_request method
            if self.evolution_manager:
                self.evolution_manager.record_api_request(encode_time, True)

            logger.info(f"Encoded experience for request {request_id} in {encode_time:.3f}s")

        except Exception as e:
            logger.error(f"Error encoding experience: {e}")
            import traceback
            traceback.print_exc()

    def _extract_conversation_context(self, messages: List[Dict]) -> str:
        """Extract conversation context from messages."""
        if not messages:
            return ""

        # Get last few messages for context
        context_messages = messages[-5:]  # Last 5 messages

        context_parts = []
        for msg in context_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if content:
                context_parts.append(f"{role}: {content}")

        return " | ".join(context_parts)

    def _inject_memories(self, messages: List[Dict], memories: List[Dict]) -> List[Dict]:
        """Inject memories into the conversation."""
        if not memories:
            return messages

        # Create memory context
        memory_context = []
        for i, memory in enumerate(memories[:3]):  # Use top 3 memories
            memory_content = memory.get("content", "")
            if memory_content:
                memory_context.append(f"Memory {i + 1}: {memory_content}")

        memory_text = " | ".join(memory_context)

        # Inject after system message or as first message
        if messages and messages[0].get("role") == "system":
            # Insert after system message
            new_messages = [messages[0]]
            new_messages.append({
                "role": "system",
                "content": f"Relevant Context: {memory_text}"
            })
            new_messages.extend(messages[1:])
        else:
            # Insert as first message
            new_messages = [{
                "role": "system",
                "content": f"Relevant Context: {memory_text}"
            }]
            new_messages.extend(messages)

        return new_messages

    def _get_retrieval_limit(self) -> int:
        """Get retrieval limit from centralized config only."""
        return self.config.retrieval.default_top_k

    def _build_conversation_context(self, messages: List[Dict]) -> Dict[str, Any]:
        """Build structured conversation context."""
        message_limit = self._get_retrieval_limit()  # Centralized config only

        return {
            "messages": messages[-message_limit:],  # Last N messages from config
            "total_messages": len(messages),
            "memories_used": []  # Will be populated by memory injection
        }

    def _log_memory_retrieval_details(
            self,
            query: str,
            memories: List[Dict],
            retrieval_time: float):
        """Log detailed memory retrieval information."""
        retrieval_limit = self._get_retrieval_limit()  # Centralized config only

        if not memories:
            logger.debug(f"No memories found for query: {query[:100]}")
            return

        logger.info(f"Memory retrieval completed:")
        logger.info(f"  Query: {query[:100]}...")
        logger.info(f"  Retrieval time: {retrieval_time:.3f}s")
        logger.info(f"  Retrieval limit (top_k): {retrieval_limit}")
        logger.info(f"  Memories found: {len(memories)}")

        for i, memory in enumerate(memories[:3]):  # Log top 3
            content = memory.get("content", "")
            score = memory.get("score", 0)
            logger.info(f"    Memory {i + 1}: score={score:.3f}, content={content[:80]}...")


# Factory function for easy instantiation
def create_enhanced_middleware(memory_system, evolution_manager, config):
    """Create enhanced middleware instance."""
    return EnhancedMemoryMiddleware(memory_system, evolution_manager, config)
