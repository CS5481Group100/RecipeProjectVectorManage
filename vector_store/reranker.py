
import os
import multiprocessing as _mp

# Safety and parallelism settings to avoid segfaults on macOS when using
# tokenizers / transformers / torch together with multiprocessing.
# These should be set before heavy native libraries initialize threads.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    _mp.set_start_method("spawn")
except RuntimeError:
    # start method may already be set; ignore
    pass

from typing import List, Dict, Callable, Optional
import numpy as np
import torch

from sentence_transformers import CrossEncoder, SentenceTransformer


class Reranker:
    """可扩展的重排序器。

    支持两种模式：
    - 'cross': 使用 CrossEncoder 对 (query, doc) 对进行打分（更精确，但较慢）。
    - 'bi': 使用 bi-encoder（SentenceTransformer）对 query 与 doc 向量做余弦相似度重排（更快但较弱）。

    参数说明：
    - mode: 'cross' 或 'bi'
    - model_name: 具体模型名称（默认会选择适合中文/多语的模型）
    - device: 'cpu' 或 'cuda'，默认自动选择
    - weight_initial_score: 在 0..1 之间，控制初始检索分与 reranker 分数的线性融合权重（1=只用初始分，0=只用 reranker）
    - cross_batch_size: cross-encoder 批大小
    """

    def __init__(
        self,
        mode: str = "cross",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        weight_initial_score: float = 0.0,
        cross_batch_size: int = 32,
    ):
        self.mode = mode
        self.weight_initial_score = float(weight_initial_score)
        self.cross_batch_size = int(cross_batch_size)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.mode == "cross":
            if model_name is None:
                # 这是一个表现良好的 cross-encoder，多语/中文也能用
                model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self.model = CrossEncoder(model_name, device=self.device)
            self.bi_model = None
        elif self.mode == "bi":
            if model_name is None:
                model_name = "paraphrase-multilingual-MiniLM-L12-v2"
            self.bi_model = SentenceTransformer(model_name, device=self.device)
            self.model = None
        else:
            raise ValueError("mode must be 'cross' or 'bi'")

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        # Min-max normalize to [0,1]
        if scores.size == 0:
            return scores
        mn, mx = float(scores.min()), float(scores.max())
        if mx - mn <= 1e-12:
            return np.zeros_like(scores, dtype=float)
        return (scores - mn) / (mx - mn)

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: Optional[int] = None,
        hard_filter: Optional[Callable[[Dict], bool]] = None,
    ) -> List[Dict]:
        """对候选进行重排序并返回带新分数的列表。

        candidates 中每个 dict 建议包含: `id`, `name`, `text`, `score`（原检索分）
        返回的每个 candidate 会附加字段: `rerank_score`, `rerank_norm`, `init_score`, `init_norm`, `combined_score`。
        """
        if hard_filter:
            candidates = [c for c in candidates if hard_filter(c)]
        if len(candidates) == 0:
            return []

        docs = [((c.get("name", "") + "\n" + c.get("text", "")).strip()) for c in candidates]
        init_scores = np.array([float(c.get("score", 0.0)) for c in candidates], dtype=float)

        if self.mode == "cross":
            pairs = [(query, d) for d in docs]
            scores = []
            for i in range(0, len(pairs), self.cross_batch_size):
                batch = pairs[i : i + self.cross_batch_size]
                batch_scores = self.model.predict(batch)
                scores.extend(batch_scores.tolist())
            rerank_scores = np.array(scores, dtype=float)
        else:
            q_emb = self.bi_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            d_embs = self.bi_model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
            rerank_scores = (d_embs @ q_emb).astype(float)

        rerank_norm = self._normalize_scores(rerank_scores)
        init_norm = self._normalize_scores(init_scores)

        w = float(self.weight_initial_score)
        combined = w * init_norm + (1.0 - w) * rerank_norm

        for i, c in enumerate(candidates):
            c["rerank_score"] = float(rerank_scores[i])
            c["rerank_norm"] = float(rerank_norm[i])
            c["init_score"] = float(init_scores[i])
            c["init_norm"] = float(init_norm[i])
            c["combined_score"] = float(combined[i])

        candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        if top_k is not None:
            return candidates[: int(top_k)]
        return candidates


__all__ = ["Reranker"]
