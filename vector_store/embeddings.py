from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional

class EmbeddingModel:
    """Wrapper around a sentence-transformers model for batch encoding.

    Default model is multilingual MiniLM which supports Chinese well.
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Encode a list of texts to a numpy array of shape (n, dim).

        Returns raw embeddings (not normalized). Caller may normalize for cosine similarity.
        """
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return embs


def get_default_model(device: Optional[str] = None) -> EmbeddingModel:
    return EmbeddingModel()
