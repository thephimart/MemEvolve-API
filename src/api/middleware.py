"""
Middleware for integrating memory functionality into API requests.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryMiddleware:
    """Middleware to handle memory integration for API requests."""

    def __init__(
        self,
        memory_system: Optional[Any] = None,
        evolution_manager: Optional[Any] = None
    ):
        self.memory_system = memory_system
        self.evolution_manager = evolution_manager

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
        if not self.memory_system or method != "POST" or not path.startswith("/v1/chat/completions"):
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
            logger.error(
                f"Error processing request for memory integration: {e}")
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
        if not self.memory_system or method != "POST" or not path.startswith("/v1/chat/completions"):
            return

        try:
            # Parse request and response
            request_data = json.loads(request_body)
            response_data = json.loads(response_body)

            # Extract the conversation
            messages = request_data.get("messages", [])
            assistant_response = response_data.get(
                "choices", [{}])[0].get("message", {})

            if messages and assistant_response:
                # Create experience from the interaction
                experience = self._create_experience(
                    messages, assistant_response, request_context)

                # Encode the experience
                self.memory_system.add_experience(experience)

                logger.info("Encoded new experience into memory")

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

    def _create_experience(self, messages: List[Dict[str, Any]], assistant_response: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Create an experience object from the conversation."""
        # Extract key information
        user_query = context.get("original_query", "")
        assistant_content = assistant_response.get("content", "")

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
        content = f"{query} {response}".lower()

        if "code" in content or "function" in content:
            tags.append("programming")
        if "error" in content or "bug" in content:
            tags.append("debugging")
        if "explain" in content or "what" in content:
            tags.append("explanation")
        if "api" in content or "endpoint" in content:
            tags.append("api")

        return tags
