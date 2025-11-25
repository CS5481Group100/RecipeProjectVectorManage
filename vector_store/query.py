import os
import json
from typing import List, Optional

# Safety settings to avoid tokenizer/torch multiprocessing issues on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import faiss
import numpy as np

from .embeddings import EmbeddingModel
from .utils import l2_normalize_vectors, to_float32, render_source_text
from .reranker import Reranker
from .config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_PATH,
    DEFAULT_META_PATH,
    DEFAULT_K,
    DEFAULT_RERANK_MODE,
    DEFAULT_CROSS_RERANKER_MODEL,
    DEFAULT_BI_RERANKER_MODEL,
    DEFAULT_WEIGHT_INITIAL_SCORE,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_DEVICE,
)


def load_index_and_meta(index_path: str, meta_path: str):
    index = faiss.read_index(index_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        metas = json.load(f)
    return index, metas


def search(index_path: str, meta_path: str, query: str, k: int = 5, model_name: str = DEFAULT_EMBEDDING_MODEL):
    index, metas = load_index_and_meta(index_path, meta_path)
    model = EmbeddingModel(model_name=model_name)
    q_emb = model.encode([query])
    # Ensure q_emb is a 2D, C-contiguous float32 array of shape (n_queries, dim)
    q_emb = to_float32(q_emb)
    q_emb = l2_normalize_vectors(q_emb)
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    q_emb = np.ascontiguousarray(q_emb, dtype='float32')

    # Validate embedding dim matches index dimension
    try:
        idx_dim = index.d
    except Exception:
        # Some FAISS indexes may expose dimension differently; fall back to ntotal check
        idx_dim = None
    if idx_dim is not None and q_emb.shape[1] != idx_dim:
        raise ValueError(f"Embedding dimension mismatch: index expects {idx_dim} but query embedding has {q_emb.shape[1]}")

    D, I = index.search(q_emb, k)
    results = []
    meta_count = len(metas)
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        if idx >= meta_count:
            print(
                f"[warn] FAISS returned idx={idx} but only {meta_count} meta rows. Check index/meta files.",
                flush=True,
            )
            continue
        m = metas[idx]
        source_text = render_source_text(m.get('source')) or None
        results.append(
            {
                'score': float(score),
                'id': m.get('id'),
                'name': m.get('name'),
                'text': m.get('text'),
                'type': m.get('type'),
                'origin_id': m.get('origin_id'),
                'source': m.get('source'),
                'source_text': source_text,
            }
        )
    return results


if __name__ == '__main__':
    import typer
    from rich import print

    app = typer.Typer()

    @app.command()
    def main(
        query: str,
        index: str = DEFAULT_INDEX_PATH,
        meta: str = DEFAULT_META_PATH,
        k: int = DEFAULT_K,
        model: str = DEFAULT_EMBEDDING_MODEL,
        use_rerank: bool = False,
        rerank_mode: str = DEFAULT_RERANK_MODE,
        rerank_model: Optional[str] = None,
        weight_initial_score: float = DEFAULT_WEIGHT_INITIAL_SCORE,
        rerank_top_k: Optional[int] = None,
        rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
        device: Optional[str] = DEFAULT_DEVICE,
    ):
        """Query the vector DB and optionally rerank results.

        Examples:
          python -m vector_store.query "我的问题" --use-rerank --rerank-mode cross
        """
        res = search(index, meta, query, k=k, model_name=model)

        if use_rerank and len(res) > 0:
            # determine rerank model default based on mode
            if rerank_model is None:
                rerank_model = DEFAULT_CROSS_RERANKER_MODEL if rerank_mode == 'cross' else DEFAULT_BI_RERANKER_MODEL

            reranker = Reranker(
                mode=rerank_mode,
                model_name=rerank_model,
                device=device,
                weight_initial_score=weight_initial_score,
                cross_batch_size=rerank_batch_size,
            )
            res = reranker.rerank(query, res, top_k=rerank_top_k)

        print(res)

    app()
