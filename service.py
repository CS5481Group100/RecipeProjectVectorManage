"""FastAPI service for the recipe vector store."""
import os
import threading
import time
from functools import lru_cache
from typing import Dict, List, Literal, Optional

# Safety defaults before importing heavy deps
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from vector_store.config import (
    DEFAULT_BI_RERANKER_MODEL,
    DEFAULT_CROSS_RERANKER_MODEL,
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_PATH,
    DEFAULT_K,
    DEFAULT_META_PATH,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_MODE,
    DEFAULT_WEIGHT_INITIAL_SCORE,
)

from vector_store.embeddings import EmbeddingModel
from vector_store.query import load_index_and_meta
from vector_store.reranker import Reranker
from vector_store.utils import l2_normalize_vectors, to_float32


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query text")
    k: int = Field(DEFAULT_K, ge=1, le=100, description="Number of candidates to retrieve")
    use_rerank: bool = Field(False, description="Whether to apply reranking")
    rerank_mode: Literal["cross", "bi"] = Field(
        DEFAULT_RERANK_MODE, description="Reranking mode when enabled"
    )
    rerank_model: Optional[str] = Field(
        None, description="Override reranker model (defaults depend on mode)"
    )
    rerank_top_k: Optional[int] = Field(
        None, ge=1, le=100, description="Return top-k results after reranking"
    )
    weight_initial_score: float = Field(
        DEFAULT_WEIGHT_INITIAL_SCORE,
        ge=0.0,
        le=1.0,
        description="Blend weight between initial and rerank scores",
    )


class SearchResult(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    text: Optional[str] = None
    score: float
    rerank_score: Optional[float] = None
    rerank_norm: Optional[float] = None
    init_score: Optional[float] = None
    init_norm: Optional[float] = None
    combined_score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    total_hits: int
    use_rerank: bool
    results: List[SearchResult]
    timings_ms: Dict[str, float]


class RetrievalEngine:
    """Caches models and indexes so the FastAPI service stays fast."""

    def __init__(
        self,
        index_path: str = DEFAULT_INDEX_PATH,
        meta_path: str = DEFAULT_META_PATH,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = DEFAULT_DEVICE,
        rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    ):
        self.index_path = index_path
        self.meta_path = meta_path
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.rerank_batch_size = rerank_batch_size

        self._index = None
        self._metas: List[dict] = []
        self._index_lock = threading.RLock()
        self._reranker_lock = threading.RLock()
        self._rerankers: Dict[tuple, Reranker] = {}

        self.embedding_model = EmbeddingModel(
            model_name=self.embedding_model_name, device=self.device
        )
        self.reload()

    @property
    def index_size(self) -> int:
        with self._index_lock:
            return int(self._index.ntotal) if self._index is not None else 0

    def reload(self) -> None:
        index, metas = load_index_and_meta(self.index_path, self.meta_path)
        with self._index_lock:
            self._index = index
            self._metas = metas

    def _pick_rerank_default(self, mode: str) -> str:
        return (
            DEFAULT_CROSS_RERANKER_MODEL if mode == "cross" else DEFAULT_BI_RERANKER_MODEL
        )

    def _get_reranker(self, mode: str, model_name: str) -> Reranker:
        key = (mode, model_name)
        reranker = self._rerankers.get(key)
        if reranker is None:
            reranker = Reranker(
                mode=mode,
                model_name=model_name,
                device=self.device,
                cross_batch_size=self.rerank_batch_size,
                weight_initial_score=DEFAULT_WEIGHT_INITIAL_SCORE,
            )
            self._rerankers[key] = reranker
        return reranker

    def search(self, payload: SearchRequest) -> SearchResponse:
        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="Query must not be empty")

        with self._index_lock:
            if self._index is None or self._metas is None:
                raise HTTPException(status_code=503, detail="Index is not loaded yet")
            index = self._index
            metas = self._metas

        timings: Dict[str, float] = {}
        t0 = time.perf_counter()
        q_emb = self.embedding_model.encode([payload.query])
        q_emb = to_float32(q_emb)
        q_emb = l2_normalize_vectors(q_emb)
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        timings["encode"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        distances, indices = index.search(q_emb, payload.k)
        timings["faiss_search"] = (time.perf_counter() - t1) * 1000

        hits: List[Dict] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            meta = metas[idx]
            hits.append(
                {
                    "score": float(score),
                    "id": _opt_str(meta.get("id")),
                    "name": _opt_str(meta.get("name")),
                    "text": _opt_str(meta.get("text")),
                }
            )

        use_rerank = bool(payload.use_rerank)
        if use_rerank and hits:
            rerank_mode = payload.rerank_mode
            rerank_model = payload.rerank_model or self._pick_rerank_default(rerank_mode)
            rerank_top_k = payload.rerank_top_k or payload.k
            reranker = self._get_reranker(rerank_mode, rerank_model)
            with self._reranker_lock:
                original_weight = reranker.weight_initial_score
                reranker.weight_initial_score = payload.weight_initial_score
                try:
                    rerank_start = time.perf_counter()
                    hits = reranker.rerank(payload.query, hits, top_k=rerank_top_k)
                    timings["rerank"] = (time.perf_counter() - rerank_start) * 1000
                finally:
                    reranker.weight_initial_score = original_weight

        timings["total"] = sum(timings.values())
        return SearchResponse(
            query=payload.query,
            total_hits=len(hits),
            use_rerank=use_rerank,
            results=[SearchResult(**h) for h in hits],
            timings_ms={k: round(v, 3) for k, v in timings.items()},
        )


def _env_path(key: str, default: str) -> str:
    return os.getenv(key, default)


def _opt_str(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


@lru_cache(maxsize=1)
def get_engine() -> RetrievalEngine:
    return RetrievalEngine(
        index_path=_env_path("VECTOR_INDEX_PATH", DEFAULT_INDEX_PATH),
        meta_path=_env_path("VECTOR_META_PATH", DEFAULT_META_PATH),
        embedding_model_name=os.getenv("VECTOR_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        device=os.getenv("VECTOR_DEVICE", DEFAULT_DEVICE),
        rerank_batch_size=int(os.getenv("VECTOR_RERANK_BATCH_SIZE", DEFAULT_RERANK_BATCH_SIZE)),
    )


app = FastAPI(title="Recipe Retrieval API", version="0.1.0")


_INDEX_HTML = """
<!DOCTYPE html>
<html lang=\"zh\">
    <head>
        <meta charset=\"utf-8\" />
        <title>Recipe Retrieval</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 2rem; background: #f6f6f6; }
            .card { max-width: 860px; margin: 0 auto; background: white; border-radius: 12px; padding: 2rem; box-shadow: 0 6px 18px rgba(0,0,0,0.08); }
            h1 { margin-top: 0; }
            textarea { width: 100%; min-height: 100px; font-size: 1rem; padding: 0.75rem; border-radius: 8px; border: 1px solid #ccc; resize: vertical; }
            button { background: #2e7df6; border: none; color: white; padding: 0.8rem 1.4rem; border-radius: 8px; font-size: 1rem; cursor: pointer; }
            button:disabled { opacity: 0.6; cursor: not-allowed; }
            .controls { display: flex; gap: 1rem; align-items: center; margin: 1rem 0; flex-wrap: wrap; }
            label { font-size: 0.9rem; }
            input[type=number], select { padding: 0.4rem; border-radius: 6px; border: 1px solid #ccc; }
            .result { border-top: 1px solid #eee; padding: 1rem 0; }
            .result h3 { margin: 0 0 0.5rem; }
            .meta { font-size: 0.85rem; color: #6b7280; }
            pre { white-space: pre-wrap; background: #f4f4f4; padding: 0.75rem; border-radius: 6px; }
            .status { margin-top: 0.5rem; font-size: 0.9rem; color: #666; }
        </style>
    </head>
    <body>
        <div class=\"card\">
            <h1>Recipe Retrieval</h1>
            <p>输入自然语言问题，服务将返回召回结果，可选开启 cross/bi 重排。</p>
            <textarea id=\"query\" placeholder=\"例如：天冷了我想吃羊肉，怎么做？\"></textarea>
            <div class=\"controls\">
                <label>Top-K <input id=\"k\" type=\"number\" min=\"1\" max=\"50\" value=\"5\" /></label>
                <label>重排
                    <select id=\"rerank-mode\">
                        <option value=\"none\">关闭</option>
                        <option value=\"cross\">Cross Encoder</option>
                        <option value=\"bi\">Bi Encoder</option>
                    </select>
                </label>
                <label>重排 Top-K <input id=\"rerank-top\" type=\"number\" min=\"1\" max=\"50\" value=\"5\" /></label>
            </div>
            <button id=\"search-btn\">Search</button>
            <div class=\"status\" id=\"status\"></div>
            <div id=\"results\"></div>
        </div>
        <script>
            const btn = document.getElementById('search-btn');
            const statusEl = document.getElementById('status');
            const resultsEl = document.getElementById('results');
            btn.addEventListener('click', async () => {
                const query = document.getElementById('query').value.trim();
                const k = parseInt(document.getElementById('k').value, 10);
                const rerankMode = document.getElementById('rerank-mode').value;
                const rerankTop = parseInt(document.getElementById('rerank-top').value, 10);
                if (!query) {
                    alert('请输入查询内容');
                    return;
                }
                btn.disabled = true;
                statusEl.textContent = '查询中...';
                resultsEl.innerHTML = '';
                try {
                    const body = {
                        query,
                        k,
                        use_rerank: rerankMode !== 'none',
                        rerank_mode: rerankMode === 'none' ? 'cross' : rerankMode,
                        rerank_top_k: rerankTop
                    };
                    const resp = await fetch('/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body)
                    });
                    if (!resp.ok) {
                        const err = await resp.json();
                        throw new Error(err.detail || 'Request failed');
                    }
                    const data = await resp.json();
                    statusEl.textContent = `共 ${data.total_hits} 条，用时 ${data.timings_ms.total || '?'} ms`;
                    if (!data.results.length) {
                        resultsEl.innerHTML = '<p>无结果</p>';
                        return;
                    }
                    resultsEl.innerHTML = data.results.map((item, idx) => `
                        <div class=\"result\">
                            <div class=\"meta\">#${idx + 1} · score ${item.score.toFixed(4)}${item.combined_score ? ` · combined ${item.combined_score.toFixed(4)}` : ''}</div>
                            <h3>${item.name || item.id || '无标题'}</h3>
                            <pre>${(item.text || '').replace(/[<>]/g, '')}</pre>
                        </div>
                    `).join('');
                } catch (err) {
                    statusEl.textContent = err.message;
                } finally {
                    btn.disabled = false;
                }
            });
        </script>
    </body>
</html>
"""


@app.on_event("startup")
def _startup_event() -> None:
    # Trigger lazy engine build so startup fails fast if files missing
    get_engine()


@app.get("/healthz")
def healthz():
    engine = get_engine()
    return {
        "status": "ok",
        "index_size": engine.index_size,
        "embedding_model": engine.embedding_model_name,
        "device": engine.device,
    }


@app.get("/", response_class=HTMLResponse)
def landing_page():
    return HTMLResponse(content=_INDEX_HTML)


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest):
    engine = get_engine()
    return engine.search(payload)


@app.post("/search/docs", response_model=List[SearchResult])
def search_docs(payload: SearchRequest):
    engine = get_engine()
    response = engine.search(payload)
    return response.results


@app.post("/reload")
def reload_index():
    engine = get_engine()
    engine.reload()
    return {"status": "reloaded", "index_size": engine.index_size}


__all__ = ["app", "get_engine", "RetrievalEngine"]
