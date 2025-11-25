"""Central configuration/defaults for the vector_store package."""
from typing import Optional

# Embedding model (sentence-transformers)
DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-base-zh-v1.5"

# Paths
DEFAULT_INDEX_PATH: str = "data/indx.faiss"
DEFAULT_META_PATH: str = "data/meta.json"

# Query defaults
DEFAULT_K: int = 5
DEFAULT_RETRIEVAL_SCORE_THRESHOLD: float = 0.4

# Reranker defaults
# mode: 'cross' (CrossEncoder) or 'bi' (bi-encoder)
DEFAULT_RERANK_MODE: str = "cross"
DEFAULT_CROSS_RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3" 
#"BAAI/bge-reranker-base"-这个模型无法理解“非A”的逻辑

# Bi-encoder: a Chinese sentence embedding model (fast and suitable for bi-mode rerank)
DEFAULT_BI_RERANKER_MODEL: str = "shibing624/text2vec-base-chinese"
DEFAULT_WEIGHT_INITIAL_SCORE: float = 0.4
DEFAULT_RERANK_BATCH_SIZE: int = 32
DEFAULT_RERANK_COMBINED_THRESHOLD: float = 0.4
DEFAULT_RERANK_SCORE_KILL_THRESHOLD: float = 0.15

# Device: if None, components will auto-detect (e.g., use CUDA if available)
DEFAULT_DEVICE: Optional[str] = "mps"
