import os
import json
from typing import Dict, List, Any, Optional, Union
from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime

from .metrics import EncodingMetricsCollector
from ...utils.config import MemEvolveConfig, load_config

logger = logging.getLogger(__name__)


class ExperienceEncoder:
    def __init__(
        self,
        base_url: Optional[str] = None,
        memory_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 600,
        max_retries: int = 3,
        encoding_strategies: Optional[List[str]] = None,
        max_tokens: int = 512,
        config: Optional[MemEvolveConfig] = None,
        evolution_encoding_strategies: Optional[List[str]] = None
    ):
        # Use memory_base_url (for memory LLM tasks), not upstream
        self.base_url = base_url or memory_base_url
        if not self.base_url:
            raise ValueError(
                "Memory base URL must be provided via base_url or memory_base_url parameter")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.client: Optional[OpenAI] = None
        self.metrics_collector = EncodingMetricsCollector()
        self._auto_model = False

        # Add configuration-driven prompts
        self.config = config or load_config()
        self.encoding_prompts = self.config.encoding_prompts

        # Set encoding strategies with priority: evolution_state > parameter > config > fallback
        self.encoding_strategies = (
            evolution_encoding_strategies or  # Priority 1: Evolution system override
            encoding_strategies or  # Priority 2: Direct parameter
            self.config.encoder.encoding_strategies or  # Priority 3: Environment via config
            self.config.encoding_prompts.encoding_strategies_fallback  # Priority 4: Config.py fallback
        )

        # Use configuration-based type descriptions
        self.type_descriptions = self.config.encoding_prompts.type_descriptions

    def _get_type_descriptions(self) -> str:
        """Generate type descriptions string for configured strategies."""
        descriptions = []
        for strategy in self.encoding_strategies:
            desc = self.type_descriptions.get(strategy, strategy)
            descriptions.append(f'"{strategy}" ({desc})')
        return ", ".join(descriptions)

    def initialize_memory_api(self):
        if self.client is not None:
            return

        try:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                max_retries=self.max_retries
            )
            logger.debug(f"Memory API client initialized successfully at {self.base_url}")

            if self.model is None and not self._auto_model:
                model_info = self.get_model_info()
                if model_info:
                    self.model = model_info.get("id")
                    self._auto_model = True
                    logger.debug(f"Auto-detected model: {self.model}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM client: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information from API."""
        try:
            import requests
        except ImportError:
            return {}

        try:
            headers = {}
            if self.api_key and self.api_key != "dummy-key":
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=5.0
            )
            response.raise_for_status()

            data = response.json()

            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]

            return {}
        except Exception:
            return {}

    def _clean_memory_api_response(self, response: str) -> str:
        """Clean LLM response to extract valid JSON.

        Handles common LLM response formats like:
        - Raw JSON
        - Markdown code blocks (```json ... ```)
        - JSON with extra text

        Args:
            response: Raw LLM response

        Returns:
            Clean JSON string
        """
        if not response:
            raise ValueError("Empty response from LLM")

        # Remove leading/trailing whitespace
        response = response.strip()

        # Handle markdown code blocks
        if response.startswith("```json"):
            response = response[7:]  # Remove ```json
        elif response.startswith("```"):
            response = response[3:]  # Remove ```

        if response.endswith("```"):
            response = response[:-3]  # Remove trailing ```

        # Remove any remaining leading/trailing whitespace
        response = response.strip()

        # Try to find the first complete JSON object
        import re

        # Find all JSON-like structures (balanced braces)
        json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)

        for match in matches:
            try:
                # Test if this is valid JSON
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue

        # Fallback: if no valid JSON found, try the original approach
        if not response.startswith('{'):
            # Look for JSON object within the response
            start_idx = response.find('{')
            end_idx = response.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                response = response[start_idx:end_idx + 1]
            else:
                raise ValueError(
                    f"No JSON object found in response: {response[:100]}...")

        return response

    def _estimate_content_size(self, experience: Dict[str, Any]) -> int:
        """Estimate token requirements for experience content."""
        import tiktoken

        try:
            # Use a standard tokenizer for estimation
            enc = tiktoken.get_encoding("cl100k_base")
            content = json.dumps(experience, indent=2)
            return len(enc.encode(content))
        except Exception:
            # Fallback: rough character-based estimation
            return len(str(experience)) // 4  # Rough estimate: 4 chars per token

    def _requires_batch_processing(self, experience: Dict[str, Any], max_tokens: int) -> bool:
        """Determine if experience content requires batch processing."""
        # Account for prompt template tokens (~150-200 tokens)
        prompt_template_tokens = 200
        safety_margin = 50

        content_size = self._estimate_content_size(experience)
        estimated_request_size = content_size + prompt_template_tokens + safety_margin

        # Also consider response size needs
        min_response_tokens = 100  # Minimum viable response

        return estimated_request_size + min_response_tokens > max_tokens

    def _semantic_chunk_experience(
            self, experience: Dict[str, Any], max_chunk_size: int) -> List[Dict[str, Any]]:
        """Split experience into semantic chunks for batch processing."""
        chunks = []

        # Get the main content that needs chunking
        experience_str = json.dumps(experience, indent=2)

        # Simple semantic chunking: split by logical boundaries
        if len(experience_str) <= max_chunk_size:
            return [experience]

        # Try to split at natural boundaries
        lines = experience_str.split('\n')
        current_chunk = ""

        for line in lines:
            test_chunk = current_chunk + line + '\n'

            if len(test_chunk) <= max_chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk.strip():
                    try:
                        chunk_obj = json.loads(current_chunk)
                        chunks.append(chunk_obj)
                    except json.JSONDecodeError:
                        # Fallback: create minimal chunk with partial content
                        chunks.append({
                            "type": "partial_experience",
                            "content": current_chunk[:max_chunk_size],
                            "chunk_id": len(chunks)
                        })

                current_chunk = line + '\n'

        # Don't forget the last chunk
        if current_chunk.strip():
            try:
                chunk_obj = json.loads(current_chunk)
                chunks.append(chunk_obj)
            except json.JSONDecodeError:
                chunks.append({
                    "type": "partial_experience",
                    "content": current_chunk[:max_chunk_size],
                    "chunk_id": len(chunks)
                })

        # Ensure we have at least one chunk
        if not chunks:
            chunks = [{
                "type": "experience",
                "content": str(experience)[:max_chunk_size],
                "chunk_id": 0,
                "truncated": True
            }]

        return chunks

    def _encode_chunk(self, chunk: Dict[str, Any], max_tokens: int,
                      chunk_index: int, total_chunks: int) -> Dict[str, Any]:
        """Encode a single chunk with adjusted prompt."""
        if self.client is None:
            raise RuntimeError(
                "Memory API client not initialized. Call initialize_memory_api() first.")

        type_descriptions = self._get_type_descriptions()

        # Use configuration-driven prompt for chunk processing
        type_descriptions = self._get_type_descriptions()
        chunk_prompt = (
            f"{self.encoding_prompts.chunk_processing_instruction}\n\n"
            f"Available types: {type_descriptions}\n\n"
            f"Chunk {chunk_index + 1} of {total_chunks}:\n{json.dumps(chunk, indent=2)}\n\n"
            f"{self.encoding_prompts.chunk_content_instruction}\n\n"
            f"Example: {self.encoding_prompts.chunk_structure_example}"
        )

        kwargs = {
            "messages": [{"role": "user", "content": chunk_prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "timeout": self.timeout
        }

        if self.model is not None:
            kwargs["model"] = self.model

        try:
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

            if content is None:
                raise RuntimeError("Empty response from LLM")

            cleaned_content = self._clean_memory_api_response(content)
            structured_data = json.loads(cleaned_content)

            # Add chunk metadata
            if "metadata" not in structured_data:
                structured_data["metadata"] = {}
            structured_data["metadata"]["chunk_index"] = chunk_index
            structured_data["metadata"]["total_chunks"] = total_chunks
            structured_data["metadata"]["encoding_method"] = "batch_chunk"

            return structured_data

        except Exception as e:
            # Fallback chunk response
            logger.warning(f"Chunk {chunk_index} encoding failed, using fallback: {e}")
            return {
                "type": "lesson",
                "content": f"Chunk {chunk_index + 1} processing: {str(chunk)[:200]}...",
                "metadata": {
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "encoding_method": "fallback_chunk",
                    "error": str(e)
                },
                "tags": ["fallback", "chunk", "processing_error"]
            }

    def _merge_encoded_chunks(
            self, encoded_chunks: List[Dict[str, Any]], original_experience: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently merge encoded chunks back into a single memory unit."""
        if len(encoded_chunks) == 1:
            return encoded_chunks[0]

        # Extract common elements and merge intelligently
        merged = {
            "type": "lesson",  # Default for merged content
            "content": "",
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "encoding_method": "batch_merged",
                "total_chunks": len(encoded_chunks),
                "original_experience_id": original_experience.get("id")
            },
            "tags": []
        }

        # Collect all unique types (prefer specific types over generic "lesson")
        types_found = [chunk.get("type", "lesson") for chunk in encoded_chunks]
        if "tool" in types_found:
            merged["type"] = "tool"
        elif "skill" in types_found:
            merged["type"] = "skill"
        elif "abstraction" in types_found:
            merged["type"] = "abstraction"

        # Merge content in order
        contents = []
        for i, chunk in enumerate(encoded_chunks):
            content = chunk.get("content", "")
            if content:
                # Add chunk separator for readability
                if i > 0:
                    contents.append(" [CHUNK BREAK] ")
                contents.append(content)

        merged["content"] = "".join(contents)

        # Merge metadata
        all_metadata = {}
        for chunk in encoded_chunks:
            chunk_metadata = chunk.get("metadata", {})
            for key, value in chunk_metadata.items():
                if key not in all_metadata:
                    all_metadata[key] = value
                elif isinstance(value, (int, float)) and isinstance(all_metadata[key], (int, float)):
                    all_metadata[key] = (all_metadata[key] + value) / 2  # Average numeric values

        merged["metadata"].update(all_metadata)

        # Merge all tags and deduplicate
        all_tags = set()
        for chunk in encoded_chunks:
            chunk_tags = chunk.get("tags", [])
            all_tags.update(chunk_tags)

        # Add batch-specific tags
        all_tags.add("batch_processed")
        all_tags.add(f"chunks_{len(encoded_chunks)}")

        merged["tags"] = list(all_tags)

        return merged

    def _encode_with_batch_processing(self,
                                      experience: Dict[str,
                                                       Any],
                                      max_tokens: int,
                                      operation_id: str,
                                      start_time: float) -> List[Dict[str, Any]]:
        """Handle batch encoding of large experiences.

        Returns:
            List of encoded chunks as separate memory units.
        """
        experience_id = experience.get("id", "unknown")

        # Calculate chunk size (leave room for prompt and response)
        # Reserve 300 tokens for prompt/response overhead
        chunk_max_size = max(100, max_tokens - 300)

        # Split experience into semantic chunks
        chunks = self._semantic_chunk_experience(experience, chunk_max_size)
        total_chunks = len(chunks)

        logger.info(f"Splitting experience into {total_chunks} chunks for batch processing")

        # Track batch processing metrics
        batch_start_time = time.time()
        encoded_chunks = []
        successful_chunks = 0

        try:
            # Encode each chunk
            for i, chunk in enumerate(chunks):
                chunk_start = time.time()

                try:
                    encoded_chunk = self._encode_chunk(chunk, max_tokens, i, total_chunks)
                    encoded_chunks.append(encoded_chunk)
                    successful_chunks += 1

                    chunk_duration = time.time() - chunk_start
                    logger.debug(f"Chunk {i + 1}/{total_chunks} encoded in {chunk_duration:.2f}s")

                except Exception as e:
                    logger.error(f"Failed to encode chunk {i + 1}/{total_chunks}: {e}")
                    # Add fallback chunk to maintain sequence
                    fallback_chunk = {
                        "type": "lesson",
                        "content": f"Chunk {i + 1} processing failed: {str(chunk)[:100]}...",
                        "metadata": {
                            "chunk_index": i,
                            "total_chunks": total_chunks,
                            "encoding_method": "chunk_error",
                            "error": str(e)
                        },
                        "tags": ["chunk_error", "fallback"]
                    }
                    encoded_chunks.append(fallback_chunk)

            # Add batch processing metrics to each chunk
            batch_duration = time.time() - batch_start_time
            chunks_per_second = total_chunks / batch_duration if batch_duration > 0 else 0
            batch_metrics = {
                "batch_processing_time": batch_duration,
                "chunks_per_second": chunks_per_second,
                "successful_chunks": successful_chunks,
                "failed_chunks": total_chunks - successful_chunks,
                "batch_efficiency": successful_chunks / total_chunks if total_chunks > 0 else 0,
                "original_experience_id": experience.get("id")
            }

            # Apply batch metrics to each chunk
            for chunk in encoded_chunks:
                chunk["metadata"].update(batch_metrics)

            total_duration = time.time() - start_time

            # End metrics collection with batch context
            # Use first chunk for metrics (representative of batch)
            self.metrics_collector.end_encoding(
                operation_id=operation_id,
                experience_id=experience_id,
                success=True,
                encoded_unit=encoded_chunks[0] if encoded_chunks else None,
                duration=total_duration
            )

            logger.info(
                f"Batch processing completed: {successful_chunks}/{total_chunks} chunks "
                f"in {batch_duration:.2f}s ({chunks_per_second:.1f} chunks/s)"
            )

            # Return list of chunks for storage as separate memory units
            return encoded_chunks

        except Exception as e:
            total_duration = time.time() - start_time

            # Record batch processing failure
            self.metrics_collector.end_encoding(
                operation_id=operation_id,
                experience_id=experience_id,
                success=False,
                error=f"Batch processing failed: {str(e)}",
                duration=total_duration
            )

            raise RuntimeError(f"Batch processing failed: {str(e)}")

    def encode_experience(self, experience: Dict[str, Any]
                          ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Encode a single experience into a memory unit.

        Returns:
            Either a single memory unit (Dict) or a list of memory units (List)
            when batch processing is used.
        """
        if self.client is None:
            raise RuntimeError(
                "Memory API client not initialized. Call initialize_memory_api() first.")

        experience_id = experience.get("id", "unknown")
        operation_id = self.metrics_collector.start_encoding(experience_id)
        start_time = time.time()

        type_descriptions = self._get_type_descriptions()

        # Check if experience content requires batch processing
        max_tokens = getattr(self, 'max_tokens', 512)  # Get from evolution config

        if self._requires_batch_processing(experience, max_tokens):
            logger.info(f"Experience requires batch processing (max_tokens={max_tokens})")
            return self._encode_with_batch_processing(
                experience, max_tokens, operation_id, start_time)

        type_descriptions = self._get_type_descriptions()
        prompt = (
            f"{self.encoding_prompts.encoding_instruction}\n\n"
            f"Available types: {type_descriptions}\n\n"
            f"Experience:\n{json.dumps(experience, indent=2)}\n\n"
            f"{self.encoding_prompts.content_instruction}\n\n"
            f"Example: {self.encoding_prompts.structure_example}"
        )

        try:
            kwargs = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "timeout": self.timeout
            }

            if self.model is not None:
                kwargs["model"] = self.model

            logger.info(f"Making Memory API call to {self.base_url} for experience encoding")
            response = self.client.chat.completions.create(**kwargs)
            logger.info(f"Memory API call completed successfully")
            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("Empty response from LLM")

            # Clean up response to extract JSON
            cleaned_content = self._clean_memory_api_response(content)

            try:
                structured_data = json.loads(cleaned_content)
            except json.JSONDecodeError as je:
                logger.error(f"Failed to parse JSON from LLM response: {je}")
                logger.error(f"LLM response (first 500 chars): {content[:500]}")
                logger.error(f"Cleaned content (first 500 chars): {cleaned_content[:500]}")
                logger.error(f"Full response length: {len(content)}")

                # Try simple JSON repair - add missing commas before closing braces
                try:
                    import re
                    repaired = re.sub(r'}\s*"', '}, "', cleaned_content)
                    repaired = re.sub(r']\s*"', '], "', repaired)
                    repaired = re.sub(r'([^\s])\s*\{', r'\1, {', repaired)
                    repaired = re.sub(r'([^\s])\s*\[', r'\1, [', repaired)
                    structured_data = json.loads(repaired)
                    logger.info("Successfully repaired malformed JSON")
                except Exception as repair_error:
                    logger.error(f"JSON repair failed: {repair_error}")
                    raise je

            duration = time.time() - start_time
            self.metrics_collector.end_encoding(
                operation_id=operation_id,
                experience_id=experience_id,
                success=True,
                encoded_unit=structured_data,
                duration=duration
            )

            return structured_data
        except Exception as e:
            duration = time.time() - start_time
            self.metrics_collector.end_encoding(
                operation_id=operation_id,
                experience_id=experience_id,
                success=False,
                error=str(e),
                duration=duration
            )
            raise RuntimeError(f"Encoding failed: {str(e)}")

    def encode_trajectory(
        self,
        trajectory: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        encoded_units = []

        for step in trajectory:
            try:
                unit = self.encode_experience(step)
                if "type" not in unit or unit["type"] == "":
                    unit["type"] = "experience"
                encoded_units.append(unit)
            except Exception as e:
                step_id = step.get("id", "unknown")
                msg = (
                    f"Warning: Failed to encode experience "
                    f"{step_id}: {str(e)}"
                )
                print(msg)

        return encoded_units

    def encode_trajectory_batch(
        self,
        trajectory: List[Dict[str, Any]],
        max_workers: int = 4,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """Encode trajectory with parallel batch processing for improved performance.

        Args:
            trajectory: List of experience dictionaries
            max_workers: Maximum number of parallel threads
            batch_size: Size of batches for processing

        Returns:
            List of encoded units
        """
        if not trajectory:
            return []

        logger = logging.getLogger(__name__)
        logger.info(
            f"Starting batch encoding of {len(trajectory)} experiences with {max_workers} workers")

        encoded_units = []
        errors = []

        # Process in batches to avoid overwhelming the system
        for i in range(0, len(trajectory), batch_size):
            batch = trajectory[i:i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1}/{(len(trajectory) + batch_size - 1) // batch_size}")

            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as executor:
                future_to_experience = {
                    executor.submit(self._encode_single_experience_safe, exp): exp
                    for exp in batch
                }

                for future in as_completed(future_to_experience):
                    experience = future_to_experience[future]
                    try:
                        result = future.result()
                        if result:
                            encoded_units.append(result)
                    except Exception as e:
                        exp_id = experience.get("id", "unknown")
                        errors.append(
                            f"Failed to encode experience {exp_id}: {str(e)}")
                        logger.warning(
                            f"Batch encoding error for experience {exp_id}: {str(e)}")

        if errors:
            logger.warning(
                f"Batch encoding completed with {
                    len(errors)} errors out of {
                    len(trajectory)} experiences")

        logger.info(
            f"Batch encoding completed: {len(encoded_units)} units encoded successfully")
        return encoded_units

    def _encode_single_experience_safe(
            self, experience: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Safely encode a single experience with error handling."""
        try:
            unit = self.encode_experience(experience)
            if "type" not in unit or unit["type"] == "":
                unit["type"] = "experience"
            return unit
        except Exception:
            # Error is already logged in encode_experience
            return None

    def get_metrics(self):
        """Get encoding metrics."""
        return self.metrics_collector.get_metrics()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of encoding metrics."""
        return self.metrics_collector.get_summary()

    def get_encoding_history(self) -> List[Dict[str, Any]]:
        """Get encoding history."""
        return self.metrics_collector.get_encoding_history()

    def reset_metrics(self):
        """Reset all encoding metrics."""
        self.metrics_collector.reset_metrics()

    def generate_abstraction(
        self,
        units: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if self.client is None:
            raise RuntimeError(
                "Memory API client not initialized. Call initialize_memory_api() first.")

        combined_content = "\n\n".join([unit["content"] for unit in units])

        prompt = (
            "Analyze these experience units and generate a high-level "
            "abstraction:\n\n"
            f"Units:\n{combined_content}\n\n"
            "Format your response as JSON with fields:\n"
            '- "abstraction": The generated abstraction\n'
            '- "summary": Brief summary of key insights\n'
            '- "tags": Key concepts identified\n\n'
            "Example output format:\n"
            '{\n  "abstraction": "...",\n  "summary": "...",\n  '
            '"tags": ["tag1", "tag2"]\n}'
        )

        try:
            kwargs = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
                "temperature": 0.7,
                "timeout": self.timeout
            }

            if self.model is not None:
                kwargs["model"] = self.model

            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("Empty response from LLM")
            abstraction_data = json.loads(content.strip())
            return abstraction_data
        except Exception as e:
            raise RuntimeError(f"Abstraction generation failed: {str(e)}")

    def save_units(self, units: List[Dict[str, Any]], filename: str):
        with open(filename, 'w') as f:
            json.dump(units, f, indent=2)

    def load_units(self, filename: str) -> List[Dict[str, Any]]:
        if not os.path.exists(filename):
            return []

        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load units: {str(e)}")

    def save_state(self) -> Dict[str, Any]:
        """Save current component state for hot-swapping.

        Returns:
            Dictionary containing component state
        """
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "timeout": self.timeout,
            "auto_model": self._auto_model,
            "initialized": self.client is not None
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore component state after hot-swapping.

        Args:
            state: State dictionary from save_state()
        """
        self.base_url = state.get("base_url", self.base_url)
        self.api_key = state.get("api_key", self.api_key)
        self.model = state.get("model", self.model)
        self.timeout = state.get("timeout", self.timeout)
        self._auto_model = state.get("auto_model", self._auto_model)

        # Reinitialize client if it was initialized before
        if state.get("initialized", False) and self.client is None:
            self.initialize_memory_api()


if __name__ == "__main__":
    encoder = ExperienceEncoder()
    encoder.initialize_memory_api()

    sample_experience = {
        "id": "exp_001",
        "timestamp": "2024-01-18T10:30:00Z",
        "action": "executed search query",
        "result": "found relevant documents",
        "feedback": "positive outcome"
    }

    encoded_unit = encoder.encode_experience(sample_experience)
    print("Encoded unit:", encoded_unit)
