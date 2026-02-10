from .embeddings import (DummyEmbeddingProvider, EmbeddingProvider,
                         OpenAICompatibleEmbeddingProvider,
                         create_embedding_function,
                         create_embedding_function_from_encoder)
from .streaming import extract_final_from_stream
from .logging_manager import LoggingManager

logger = LoggingManager.get_logger(__name__)
logger.info("Memevolve utils package initialized")

__all__ = [
    "EmbeddingProvider",
    "DummyEmbeddingProvider",
    "OpenAICompatibleEmbeddingProvider",
    "create_embedding_function",
    "create_embedding_function_from_encoder",
    "extract_final_from_stream",
]
