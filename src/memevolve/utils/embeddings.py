from typing import Callable, Optional, Any, Dict
import numpy as np

import logging

from .config import load_config

logger = logging.getLogger(__name__)

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning(
        "tiktoken not available, using character-based token estimation"
    )


class EmbeddingProvider:

    def get_embedding(self, text: str) -> np.ndarray:
        raise NotImplementedError


class DummyEmbeddingProvider(EmbeddingProvider):

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

    def get_embedding(self, text: str) -> np.ndarray:
        text_hash = hash(text)
        np.random.seed(abs(text_hash) % 2147483647)
        embedding = np.random.randn(self.embedding_dim)
        return embedding / np.linalg.norm(embedding)


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):

    def __init__(
        self,
        embedding_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
        max_tokens_per_request: Optional[int] = None,
        evolution_override_max_tokens: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        max_retries: int = 3
    ):
        self.base_url = embedding_base_url
        if not self.base_url:
            raise ValueError(
                "Embedding base URL must be provided via embedding_base_url parameter (should be resolved by config system)"
            )

        self.api_key = api_key
        self.model = model

        if evolution_override_max_tokens is not None:
            self.max_tokens_per_request = evolution_override_max_tokens
            logger.info(
                f"Using evolution override for max_tokens: {evolution_override_max_tokens}"
            )
        elif max_tokens_per_request is not None:
            self.max_tokens_per_request = max_tokens_per_request
        else:
            self.max_tokens_per_request = 512  # Default, will be overridden by config in create_embedding_function

        self._embedding_dim = embedding_dim
        self.timeout = timeout

        self._auto_model = False
        self._tokenizer = None
        self._client = None
        self.max_retries = max_retries

    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                max_retries=self.max_retries,
            )
        return self._client

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer

        if TIKTOKEN_AVAILABLE:
            try:
                model_name = self.model if self.model else "cl100k_base"
                if model_name.startswith("gpt-"):
                    self._tokenizer = tiktoken.encoding_for_model(model_name)
                else:
                    self._tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.info(f"Using tiktoken tokenizer: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load tiktoken: {e}")

        return self._tokenizer

    def _estimate_tokens(self, text: str) -> int:
        tokenizer = self._get_tokenizer()
        if tokenizer:
            return len(tokenizer.encode(text))
        return max(1, len(text) // 4)

    def _chunk_text(self, text: str, max_tokens: int) -> list[str]:
        if self._estimate_tokens(text) <= max_tokens:
            return [text]

        import re

        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        tokens = 0

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            if tokens + sentence_tokens <= max_tokens:
                current += sentence + " "
                tokens += sentence_tokens
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence + " "
                tokens = sentence_tokens

        if current:
            chunks.append(current.strip())

        return chunks

    def _get_single_embedding(self, text: str) -> np.ndarray:
        # Check if this is llama.cpp endpoint by detecting base URL or response format
        base_url = self.base_url or ""
        if "11435" in base_url or (base_url and base_url.endswith("embeddings")):
            # Use direct requests for llama.cpp to avoid OpenAI client parsing issues
            return self._get_llama_embedding(text)
        else:
            # Use OpenAI client for OpenAI-compatible endpoints
            return self._get_openai_embedding(text)

    def _get_llama_embedding(self, text: str) -> np.ndarray:
        """Direct llama.cpp embedding request using requests."""
        import requests

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "input": text,
        }

        if self.model is not None:
            data["model"] = self.model

        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()

        embedding_data = response.json()

        # llama.cpp returns OpenAI-compatible dict format
        if isinstance(embedding_data, dict):
            if "data" in embedding_data and embedding_data["data"]:
                # OpenAI format: {"data": [{"embedding": [...], "index": 0}]}
                embedding = embedding_data["data"][0]["embedding"]
            elif "embedding" in embedding_data:
                # Direct format: {"embedding": [...]}
                embedding = embedding_data["embedding"]
            else:
                raise RuntimeError(f"Unknown llama.cpp dict format: {list(embedding_data.keys())}")
        elif isinstance(embedding_data, list):
            # Fallback for list format (unlikely but handled)
            if len(embedding_data) > 0:
                first = embedding_data[0]
                if isinstance(first, dict) and "embedding" in first:
                    embedding = first["embedding"]
                elif isinstance(first, (list, tuple)):
                    embedding = first
                elif isinstance(first, (float, int)):
                    embedding = embedding_data
                else:
                    raise RuntimeError(f"Unexpected llama.cpp list format: {type(first)}")
            else:
                raise RuntimeError("Empty llama.cpp embedding response")
        else:
            raise RuntimeError(f"Unknown llama.cpp response format: {type(embedding_data)}")

        return np.asarray(embedding, dtype=np.float32).reshape(-1)

    def _get_openai_embedding(self, text: str) -> np.ndarray:
        """OpenAI-compatible embedding request using OpenAI client."""
        client = self._get_client()
        kwargs = {
            "input": text,
            "timeout": self.timeout,
        }

        if self.model is not None:
            kwargs["model"] = self.model

        response = client.embeddings.create(**kwargs)

        embedding = None

        # Direct format detection - handles all embedding services automatically
        if isinstance(response, list):
            # llama.cpp returns list[float] or list[list[float]] directly
            if len(response) > 0:
                first = response[0]
                if isinstance(first, dict) and "embedding" in first:
                    # list[dict] with embedding key
                    embedding = first["embedding"]
                elif isinstance(first, (list, tuple)):
                    # list[list[float]] - take first embedding
                    embedding = first
                elif isinstance(first, (float, int)):
                    # list[float] - direct embedding
                    embedding = response
                else:
                    raise RuntimeError(f"Unexpected list format: {type(first)}")
            else:
                raise RuntimeError("Empty embedding response list")

        elif hasattr(response, "data"):
            # OpenAI format: {"data": [{"embedding": [...], "index": 0}]}
            embedding = response.data[0].embedding

        elif isinstance(response, dict):
            # Alternative dict formats
            if "data" in response:
                # {"data": [{"embedding": [...]}]}
                embedding = response["data"][0]["embedding"]
            elif "embedding" in response:
                # {"embedding": [...]}
                embedding = response["embedding"]
            else:
                raise RuntimeError(f"Unknown dict format: {list(response.keys())}")

        if embedding is None:
            raise RuntimeError(f"Unsupported embedding response format: {type(response)}")

        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

        return embedding

    def get_embedding(self, text: str) -> np.ndarray:
        try:
            if self.model is None and not self._auto_model:
                model_info = self.get_model_info()
                if model_info:
                    self.model = model_info.get("id")
                    self._auto_model = True

            estimated_tokens = self._estimate_tokens(text)

            if estimated_tokens <= self.max_tokens_per_request:
                embedding = self._get_single_embedding(text)
            else:
                logger.warning(
                    f"Text exceeds {self.max_tokens_per_request} tokens "
                    f"(estimated {estimated_tokens}); chunking."
                )
                chunks = self._chunk_text(text, self.max_tokens_per_request)
                embeddings = [self._get_single_embedding(c) for c in chunks]
                embedding = np.mean(embeddings, axis=0)

            if self._embedding_dim is not None and embedding.shape[0] != self._embedding_dim:
                logger.warning(
                    f"Embedding dimension mismatch: expected {self._embedding_dim}, "
                    f"got {embedding.shape[0]}"
                )

            if self._embedding_dim is None:
                self._embedding_dim = embedding.shape[0]

            return embedding

        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")

    def get_embedding_dim(self) -> Optional[int]:
        if self._embedding_dim is None:
            self._embedding_dim = self.get_embedding("test").shape[0]
        return self._embedding_dim

    def get_model_info(self) -> Dict[str, Any]:
        try:
            import requests

            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            if "data" in data and data["data"]:
                return data["data"][0]

        except Exception as e:
            logger.debug(f"Model discovery failed: {e}")

        return {}


def create_embedding_function(
    provider: str = "dummy",
    evolution_manager: Optional[Any] = None,
    **kwargs
) -> Callable[[str], np.ndarray]:
    """Create embedding function with properly named base_url parameter."""
    # Convert embedding_base_url to base_url for kwargs compatibility
    if 'embedding_base_url' in kwargs:
        kwargs['base_url'] = kwargs.pop('embedding_base_url')

    # Load centralized config for fallback values
    config = load_config()

    evolution_max_tokens = None
    if evolution_manager:
        evolution_max_tokens = getattr(
            evolution_manager, "evolution_embedding_max_tokens", None
        )

    if provider == "dummy":
        instance = DummyEmbeddingProvider(
            embedding_dim=kwargs.get("embedding_dim", config.embedding.dimension or 768)
        )

    elif provider == "openai":
        instance = OpenAICompatibleEmbeddingProvider(
            embedding_base_url=kwargs.get("base_url", config.embedding.base_url),
            api_key=kwargs.get("api_key", config.embedding.api_key),
            model=kwargs.get("model", config.embedding.model),
            timeout=config.embedding.timeout,
            max_tokens_per_request=kwargs.get("max_tokens", config.embedding.max_tokens),
            embedding_dim=kwargs.get("embedding_dim", config.embedding.dimension),
            evolution_override_max_tokens=evolution_max_tokens,
            max_retries=config.embedding.max_retries,
        )

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    return instance.get_embedding


def create_embedding_function_from_encoder(
    encoder: Any
) -> Callable[[str], np.ndarray]:
    """LLM-based fallback embedding generator (not used by memory system)."""

    def embedding_function(text: str) -> np.ndarray:
        prompt = (
            "Generate a numerical vector representation of this text. "
            "Respond ONLY with a JSON array of floats.\n\n"
            f"Text: {text}"
        )

        response = encoder.client.chat.completions.create(
            model="llama2",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )

        content = response.choices[0].message.content
        import json

        embedding = np.array(json.loads(content.strip()))
        return embedding / np.linalg.norm(embedding)

    return embedding_function
