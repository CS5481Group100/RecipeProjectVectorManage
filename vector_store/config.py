"""Central configuration/defaults for the vector_store package."""
from typing import Optional

# Embedding model (sentence-transformers)
DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-base-zh-v1.5"

# Paths
DEFAULT_INDEX_PATH: str = "data/index.faiss"
DEFAULT_META_PATH: str = "data/meta.json"

# Query defaults
DEFAULT_K: int = 5

# Reranker defaults
# mode: 'cross' (CrossEncoder) or 'bi' (bi-encoder)
DEFAULT_RERANK_MODE: str = "cross"
DEFAULT_CROSS_RERANKER_MODEL: str = "BAAI/bge-reranker-base"

# Bi-encoder: a Chinese sentence embedding model (fast and suitable for bi-mode rerank)
DEFAULT_BI_RERANKER_MODEL: str = "shibing624/text2vec-base-chinese"
DEFAULT_WEIGHT_INITIAL_SCORE: float = 0.0
DEFAULT_RERANK_BATCH_SIZE: int = 32

# Device: if None, components will auto-detect (e.g., use CUDA if available)
DEFAULT_DEVICE: Optional[str] = "mps"
