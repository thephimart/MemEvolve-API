import os
import json
from typing import Dict, List, Any, Optional
from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .metrics import EncodingMetricsCollector


class ExperienceEncoder:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        # Use environment variables, with fallback to upstream for memory tasks
        self.base_url = base_url or os.getenv("MEMEVOLVE_LLM_BASE_URL")
        # For memory tasks requiring chat completion LLM, fall back to upstream if LLM base URL is empty
        if not self.base_url:
            self.base_url = os.getenv("MEMEVOLVE_UPSTREAM_BASE_URL")
        self.api_key = api_key or os.getenv("MEMEVOLVE_LLM_API_KEY", "")
        if not self.base_url:
            raise ValueError(
                "LLM base URL must be provided via base_url parameter, MEMEVOLVE_LLM_BASE_URL, or MEMEVOLVE_UPSTREAM_BASE_URL environment variable")
        self.model = model
        self.client: Optional[OpenAI] = None
        self.metrics_collector = EncodingMetricsCollector()
        self._auto_model = False

    def initialize_llm(self):
        if self.client is not None:
            return

        try:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
            print(f"LLM client initialized successfully at {self.base_url}")

            if self.model is None and not self._auto_model:
                model_info = self.get_model_info()
                if model_info:
                    self.model = model_info.get("id")
                    self._auto_model = True
                    print(f"Auto-detected model: {self.model}")
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

    def _clean_llm_response(self, response: str) -> str:
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

    def encode_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        if self.client is None:
            raise RuntimeError(
                "LLM client not initialized. Call initialize_llm() first.")

        experience_id = experience.get("id", "unknown")
        operation_id = self.metrics_collector.start_encoding(experience_id)
        start_time = time.time()

        prompt = (
            "Transform this raw experience into a structured knowledge "
            "unit:\n\n"
            f"Experience:\n{json.dumps(experience, indent=2)}\n\n"
            "Format your response as JSON with these fields:\n"
            '- "type": "lesson" (generalizable insight), "skill" (actionable technique), '
            '"tool" (reusable function/algorithm), "abstraction" (high-level concept)\n'
            '- "content": The transformed content\n'
            '- "metadata": Additional relevant metadata (for tools: include parameters, usage, etc.)\n'
            '- "tags": Relevant tags for retrieval\n\n'
            "For tools, focus on extracting reusable functionality that can be applied to similar problems.\n\n"
            "Example output formats:\n"
            '{\n  "type": "lesson",\n  "content": "Always validate input data before processing",\n  '
            '"metadata": {},\n  "tags": ["data-validation", "best-practices"]\n}\n\n'
            '{\n  "type": "tool",\n  "content": "Binary search algorithm for finding elements in sorted arrays",\n  '
            '"metadata": {"parameters": ["array", "target"], "complexity": "O(log n)"},\n  '
            '"tags": ["algorithm", "search", "binary-search"]\n}'
        )

        try:
            kwargs = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.7,
                "timeout": 300.0
            }

            if self.model is not None:
                kwargs["model"] = self.model

            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("Empty response from LLM")

            # Clean up the response to extract JSON
            cleaned_content = self._clean_llm_response(content)
            structured_data = json.loads(cleaned_content)

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
                f"Processing batch {i//batch_size + 1}/{(len(trajectory) + batch_size - 1)//batch_size}")

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
                f"Batch encoding completed with {len(errors)} errors out of {len(trajectory)} experiences")

        logger.info(
            f"Batch encoding completed: {len(encoded_units)} units encoded successfully")
        return encoded_units
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

        # If the response doesn't start with '{', try to find JSON within it
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

    def _encode_single_experience_safe(self, experience: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
                "LLM client not initialized. Call initialize_llm() first.")

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
                "timeout": 300.0
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


if __name__ == "__main__":
    encoder = ExperienceEncoder()
    encoder.initialize_llm()

    sample_experience = {
        "id": "exp_001",
        "timestamp": "2024-01-18T10:30:00Z",
        "action": "executed search query",
        "result": "found relevant documents",
        "feedback": "positive outcome"
    }

    encoded_unit = encoder.encode_experience(sample_experience)
    print("Encoded unit:", encoded_unit)
