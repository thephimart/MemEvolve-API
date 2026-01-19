import os
import json
from typing import Dict, List, Any, Optional
from openai import OpenAI
import time

from .metrics import EncodingMetricsCollector


class ExperienceEncoder:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        # Use environment variables, with no hard-coded defaults
        self.base_url = base_url or os.getenv("MEMEVOLVE_LLM_BASE_URL")
        self.api_key = api_key or os.getenv("MEMEVOLVE_LLM_API_KEY", "")
        if not self.base_url:
            raise ValueError("LLM base URL must be provided via base_url parameter or MEMEVOLVE_LLM_BASE_URL environment variable")
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
            '- "type": "lesson" | "skill" | "abstraction"\n'
            '- "content": The transformed content\n'
            '- "metadata": Additional relevant metadata\n'
            '- "tags": Relevant tags for retrieval\n\n'
            "Example output format:\n"
            '{\n  "type": "lesson",\n  "content": "...",\n  '
            '"metadata": { ... },\n  "tags": ["tag1", "tag2"]\n}'
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
            structured_data = json.loads(content.strip())

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
