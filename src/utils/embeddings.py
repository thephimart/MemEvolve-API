from typing import Callable, Optional, Any, Dict
import numpy as np
import os


class EmbeddingProvider:
    """Base class for embedding providers."""

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        raise NotImplementedError


class DummyEmbeddingProvider(EmbeddingProvider):
    """Simple dummy embedding provider for testing.

    Uses a simple hash-based approach to create consistent embeddings.
    """

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate a pseudo-embedding based on text hash."""
        text_hash = hash(text)
        np.random.seed(abs(text_hash) % 2147483647)
        embedding = np.random.randn(self.embedding_dim)
        return embedding / np.linalg.norm(embedding)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-based embedding provider using same API as encoder."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60
    ):
        # Use environment variables, with fallback to upstream for embedding functions
        self.base_url = base_url or os.getenv("MEMEVOLVE_EMBEDDING_BASE_URL")
        # For embedding functions, fall back to upstream if embedding base URL is empty
        if not self.base_url:
            self.base_url = os.getenv("MEMEVOLVE_UPSTREAM_BASE_URL")
        self.api_key = api_key or os.getenv("MEMEVOLVE_EMBEDDING_API_KEY", "")
        if not self.base_url:
            raise ValueError(
                "Embedding base URL must be provided via base_url parameter, MEMEVOLVE_EMBEDDING_BASE_URL, or MEMEVOLVE_UPSTREAM_BASE_URL environment variable")

        self.model = model
        timeout_env = os.getenv("MEMEVOLVE_EMBEDDING_TIMEOUT", "60")
        try:
            self.timeout = int(timeout_env)
        except ValueError:
            self.timeout = timeout
        self._client: Optional[Any] = None
        self._embedding_dim: Optional[int] = None
        self._auto_model = False

    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        return self._client

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding using OpenAI API."""
        try:
            client = self._get_client()

            kwargs = {"input": text, "timeout": self.timeout}

            if self.model is None and not self._auto_model:
                model_info = self.get_model_info()
                if model_info:
                    self.model = model_info.get("id")
                    self._auto_model = True

            if self.model is not None:
                kwargs["model"] = self.model

            response = client.embeddings.create(**kwargs)
            embedding = np.array(response.data[0].embedding)

            if self._embedding_dim is None:
                self._embedding_dim = embedding.shape[0]

            return embedding
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")

    def get_embedding_dim(self) -> Optional[int]:
        """Get embedding dimension."""
        if self._embedding_dim is None:
            embedding = self.get_embedding("test")
            self._embedding_dim = embedding.shape[0]
        return self._embedding_dim

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


def create_embedding_function(
    provider: str = "dummy",
    **kwargs
) -> Callable[[str], np.ndarray]:
    """Create an embedding function.

    Args:
        provider: Type of embedding provider ("dummy", "openai")
        **kwargs: Additional arguments for the provider

    Returns:
        Function that takes text and returns embedding array
    """
    if provider == "dummy":
        provider_instance = DummyEmbeddingProvider(
            embedding_dim=kwargs.get("embedding_dim", 768)
        )
    elif provider == "openai":
        provider_instance = OpenAIEmbeddingProvider(
            base_url=kwargs.get("base_url") or os.getenv(
                "MEMEVOLVE_EMBEDDING_BASE_URL"),
            api_key=kwargs.get("api_key") or os.getenv(
                "MEMEVOLVE_EMBEDDING_API_KEY", ""),
            model=kwargs.get("model") or os.getenv("MEMEVOLVE_EMBEDDING_MODEL"),
            timeout=int(os.getenv("MEMEVOLVE_EMBEDDING_TIMEOUT", "60"))
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    return provider_instance.get_embedding


def create_embedding_function_from_encoder(
    encoder: Any
) -> Callable[[str], np.ndarray]:
    """Create embedding function from ExperienceEncoder.

    This allows reusing the encoder's LLM for embeddings if needed.
    """
    def embedding_function(text: str) -> np.ndarray:
        prompt = (
            "Generate a numerical vector representation of this text "
            "that captures its semantic meaning. Respond with a JSON "
            "array of floats representing the embedding vector:\n\n"
            f"Text: {text}\n\n"
            "Response format: [0.1, -0.2, 0.3, ...]"
        )
        try:
            response = encoder.client.chat.completions.create(
                model="llama2",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0
            )
            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("Empty response from LLM")

            import json
            embedding = np.array(json.loads(content.strip()))
            return embedding / np.linalg.norm(embedding)
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")

    return embedding_function
