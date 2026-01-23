from typing import Callable, Optional, Any, Dict
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning(
        "tiktoken not available, using character-based token estimation")


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
        timeout: int = 60,
        max_tokens_per_request: Optional[int] = None,
        evolution_override_max_tokens: Optional[int] = None,
        embedding_dim: Optional[int] = None
    ):
        # Use environment variables, with fallback to upstream for embedding functions
        self.base_url = base_url or os.getenv("MEMEVOLVE_EMBEDDING_BASE_URL")
        # For embedding functions, fall back to upstream if embedding base URL is empty
        if not self.base_url:
            self.base_url = os.getenv("MEMEVOLVE_UPSTREAM_BASE_URL")
        self.api_key = api_key or os.getenv("MEMEVOLVE_EMBEDDING_API_KEY", "")
        if not self.base_url:
            raise ValueError(
                "Embedding base URL must be provided via base_url "
                "parameter, MEMEVOLVE_EMBEDDING_BASE_URL, or "
                "MEMEVOLVE_UPSTREAM_BASE_URL environment variable")

        self.model = model

        # Priority: evolution override > parameter > env > auto-detect > fallback
        if evolution_override_max_tokens is not None:
            self.max_tokens_per_request = evolution_override_max_tokens
            logger.info(
                f"Using evolution override for max_tokens: {evolution_override_max_tokens}")
        elif max_tokens_per_request is not None:
            self.max_tokens_per_request = max_tokens_per_request
        else:
            env_max_tokens = os.getenv("MEMEVOLVE_EMBEDDING_MAX_TOKENS")
            if env_max_tokens and env_max_tokens.strip():
                try:
                    self.max_tokens_per_request = int(env_max_tokens)
                except ValueError:
                    self.max_tokens_per_request = 512
            else:
                self.max_tokens_per_request = 512

        if embedding_dim is not None:
            self._embedding_dim = embedding_dim
        else:
            env_dim = os.getenv("MEMEVOLVE_EMBEDDING_DIMENSION")
            if env_dim and env_dim.strip():
                try:
                    self._embedding_dim = int(env_dim)
                except ValueError:
                    self._embedding_dim = 768
            else:
                self._embedding_dim = None

        timeout_env = os.getenv("MEMEVOLVE_EMBEDDING_TIMEOUT", "60")
        try:
            self.timeout = int(timeout_env)
        except ValueError:
            self.timeout = timeout
        max_retries_env = os.getenv("MEMEVOLVE_API_MAX_RETRIES", "3")
        try:
            self.max_retries = int(max_retries_env)
        except ValueError:
            self.max_retries = 3
        self._client: Optional[Any] = None
        self._auto_model = False
        self._tokenizer = None

    def _get_tokenizer(self):
        """Get tiktoken tokenizer if available."""
        if self._tokenizer is not None:
            return self._tokenizer

        if TIKTOKEN_AVAILABLE:
            try:
                # Try to use the model-specific tokenizer
                model_name = self.model if self.model else "cl100k_base"
                if model_name.startswith("gpt-"):
                    self._tokenizer = tiktoken.encoding_for_model(model_name)
                else:
                    # Default to cl100k_base for other models
                    self._tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.info(f"Using tiktoken tokenizer: {model_name}")
            except Exception as e:
                logger.warning(
                    f"Failed to load tiktoken: {e}, using estimation")
        return self._tokenizer

    def _estimate_tokens(self, text: str) -> int:
        """Count tokens using tiktoken if available, otherwise estimate."""
        tokenizer = self._get_tokenizer()

        if tokenizer is not None:
            # Use accurate token counting
            return len(tokenizer.encode(text))
        else:
            # Fallback: rough estimate ~4 characters per token for English
            return len(text) // 4

    def _chunk_text(self, text: str, max_tokens: int) -> list[str]:
        """Split text into chunks of approximately max_tokens size."""
        if self._estimate_tokens(text) <= max_tokens:
            return [text]

        # Split on sentence boundaries where possible
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk += sentence + " "
                current_tokens += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Handle very long sentences by splitting at token boundaries
                if sentence_tokens > max_tokens:
                    if self._get_tokenizer() is not None:
                        # Use tokenizer to split precisely at token boundaries
                        tokens = self._get_tokenizer().encode(sentence)
                        while len(tokens) > max_tokens:
                            chunk_tokens = tokens[:max_tokens]
                            chunk_text = self._get_tokenizer().decode(chunk_tokens)
                            chunks.append(chunk_text.strip())
                            tokens = tokens[max_tokens:]
                        current_chunk = self._get_tokenizer().decode(tokens) + " "
                        current_tokens = len(tokens)
                    else:
                        # Fallback to character-based splitting
                        while sentence_tokens > max_tokens:
                            split_point = int(
                                len(sentence) * (max_tokens / sentence_tokens))
                            chunks.append(sentence[:split_point].strip())
                            sentence = sentence[split_point:].strip()
                            sentence_tokens = self._estimate_tokens(sentence)
                        current_chunk = sentence + " "
                        current_tokens = sentence_tokens
                else:
                    current_chunk = sentence + " "
                    current_tokens = sentence_tokens

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _get_single_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text chunk."""
        client = self._get_client()
        kwargs = {"input": text, "timeout": self.timeout}

        if self.model is not None:
            kwargs["model"] = self.model

        response = client.embeddings.create(**kwargs)

        # Handle different response formats from OpenAI-compatible APIs
        if hasattr(response, 'data') and response.data:
            # Standard OpenAI format: response.data[0].embedding
            return np.array(response.data[0].embedding)
        elif isinstance(response, list) and len(response) > 0:
            # Some APIs return list directly
            return np.array(response[0])
        elif hasattr(response, 'embedding'):
            # Some APIs return embedding directly on response
            return np.array(response.embedding)
        else:
            # Try to extract from response as dict
            if isinstance(response, dict):
                if 'data' in response and response['data']:
                    return np.array(response['data'][0]['embedding'])
                elif 'embedding' in response:
                    return np.array(response['embedding'])
            raise ValueError(f"Unexpected embedding response format: {type(response)}")

    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                max_retries=self.max_retries
            )
        return self._client

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding using OpenAI API with automatic chunking for large texts."""
        try:
            if self.model is None and not self._auto_model:
                model_info = self.get_model_info()
                if model_info:
                    self.model = model_info.get("id")
                    self._auto_model = True

            # Check if text exceeds token limit
            estimated_tokens = self._estimate_tokens(text)

            if estimated_tokens <= self.max_tokens_per_request:
                # Single request is sufficient
                embedding = self._get_single_embedding(text)
            else:
                # Text is too large - chunk and embed
                logger.warning(
                    f"Text exceeds {self.max_tokens_per_request} tokens "
                    f"(estimated {estimated_tokens}). "
                    f"Splitting into chunks for embedding."
                )
                chunks = self._chunk_text(text, self.max_tokens_per_request)
                logger.info(f"Split into {len(chunks)} chunks for embedding")

                # Get embedding for each chunk
                chunk_embeddings = []
                for i, chunk in enumerate(chunks):
                    chunk_emb = self._get_single_embedding(chunk)
                    chunk_embeddings.append(chunk_emb)

                # Average the embeddings
                embedding = np.mean(chunk_embeddings, axis=0)

            # Validate dimension matches expected
            if self._embedding_dim is not None:
                actual_dim = embedding.shape[0]
                if actual_dim != self._embedding_dim:
                    logger.warning(
                        f"Embedding dimension mismatch: expected {self._embedding_dim}, "
                        f"got {actual_dim}. This may cause issues.")

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
    evolution_manager: Optional[Any] = None,
    **kwargs
) -> Callable[[str], np.ndarray]:
    """Create an embedding function.

    Args:
        provider: Type of embedding provider ("dummy", "openai")
        evolution_manager: EvolutionManager instance for evolution overrides
        **kwargs: Additional arguments for provider

    Returns:
        Function that takes text and returns embedding array
    """

    # Get evolution overrides if available
    evolution_max_tokens = None
    if evolution_manager:
        evolution_max_tokens = getattr(
            evolution_manager, 'evolution_embedding_max_tokens', None)

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
            model=kwargs.get("model") or os.getenv(
                "MEMEVOLVE_EMBEDDING_MODEL"),
            timeout=int(os.getenv("MEMEVOLVE_EMBEDDING_TIMEOUT", "60")),
            max_tokens_per_request=kwargs.get("max_tokens"),
            embedding_dim=kwargs.get("embedding_dim"),
            evolution_override_max_tokens=evolution_max_tokens
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
