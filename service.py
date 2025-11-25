"""FastAPI service for the recipe vector store."""
import json
import logging
import os
import threading
import time
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional

# Safety defaults before importing heavy deps
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from vector_store.config import (
    DEFAULT_BI_RERANKER_MODEL,
    DEFAULT_CROSS_RERANKER_MODEL,
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_PATH,
    DEFAULT_K,
    DEFAULT_META_PATH,
    DEFAULT_RETRIEVAL_SCORE_THRESHOLD,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_MODE,
    DEFAULT_WEIGHT_INITIAL_SCORE,
)

from vector_store.embeddings import EmbeddingModel
from vector_store.query import load_index_and_meta
from vector_store.reranker import Reranker
from vector_store.utils import l2_normalize_vectors, to_float32, render_source_text


DEFAULT_SOURCE_META_PATH = "origin_data/recipe_meta.json"


logger = logging.getLogger("recipe_retrieval")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


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
    chunk_id: Optional[str] = None
    origin_id: Optional[int] = None
    type: Optional[str] = None
    name: Optional[str] = None
    text: Optional[str] = None
    chunk_text: Optional[str] = None
    source: Optional[Dict[str, Any]] = None
    source_text: Optional[str] = None
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="All chunk snippets grouped under this origin")
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
        source_meta_path: Optional[str] = DEFAULT_SOURCE_META_PATH,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = DEFAULT_DEVICE,
        rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    ):
        self.index_path = index_path
        self.meta_path = meta_path
        self.source_meta_path = source_meta_path
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.rerank_batch_size = rerank_batch_size

        self._index = None
        self._metas: List[dict] = []
        self._source_map: Dict[str, Dict[str, Any]] = {}
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
        source_map = self._load_source_meta()
        with self._index_lock:
            self._index = index
            self._metas = metas
            self._source_map = source_map

    def _pick_rerank_default(self, mode: str) -> str:
        return (
            DEFAULT_CROSS_RERANKER_MODEL if mode == "cross" else DEFAULT_BI_RERANKER_MODEL
        )

    def _load_source_meta(self) -> Dict[str, Dict[str, Any]]:
        path = self.source_meta_path
        if not path:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except FileNotFoundError:
            logger.warning("Source meta file not found: %s", path)
            return {}
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse source meta %s: %s", path, exc)
            return {}
        if isinstance(payload, dict):
            return {str(k): v for k, v in payload.items() if isinstance(v, dict)}
        logger.warning("Source meta %s should be a dict, got %s", path, type(payload).__name__)
        return {}

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

        hits_by_origin: Dict[str, Dict[str, Any]] = {}
        origin_order: List[str] = []
        meta_count = len(metas)
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            if score < DEFAULT_RETRIEVAL_SCORE_THRESHOLD:
                break
            if idx >= meta_count:
                logger.warning(
                    "FAISS returned idx=%s beyond meta_size=%s; index/meta files likely mismatched",
                    idx,
                    meta_count,
                )
                continue
            meta = metas[idx]
            origin_key = _origin_key(meta.get("origin_id"))
            source = None
            if origin_key and origin_key in self._source_map:
                source = self._source_map[origin_key]
            elif meta.get("source"):
                source = meta.get("source")
            source_text = render_source_text(source) or None
            chunk_text = _opt_str(meta.get("text"))
            result_text = source_text or chunk_text
            if not result_text:
                continue
            chunk_entry = {
                "chunk_id": _opt_str(meta.get("chunk_id")),
                "type": meta.get("type"),
                "chunk_text": chunk_text,
                "score": float(score),
            }
            key = origin_key or chunk_entry["chunk_id"] or _opt_str(meta.get("id"))
            if key is None:
                key = f"chunk-{idx}"
            aggregated = hits_by_origin.get(key)
            if aggregated is None:
                aggregated = {
                    "score": float(score),
                    "id": _opt_str(meta.get("id")),
                    "chunk_id": chunk_entry["chunk_id"],
                    "origin_id": meta.get("origin_id"),
                    "type": meta.get("type"),
                    "name": _opt_str(meta.get("name")),
                    "text": result_text,
                    "chunk_text": chunk_text,
                    "source": source,
                    "source_text": source_text,
                    "chunks": [chunk_entry],
                }
                hits_by_origin[key] = aggregated
                origin_order.append(key)
            else:
                aggregated["score"] = max(aggregated["score"], float(score))
                aggregated["chunks"].append(chunk_entry)
                # keep first chunk metadata for backwards compatibility
                if not aggregated.get("chunk_text"):
                    aggregated["chunk_text"] = chunk_text
                if not aggregated.get("chunk_id"):
                    aggregated["chunk_id"] = chunk_entry["chunk_id"]

        hits = [hits_by_origin[key] for key in origin_order]

        # Detailed recall logging before rerank
        recall_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        recall_header = (
            f"[{recall_timestamp}] Query='{payload.query.strip()}' | "
            f"requested_k={payload.k} | raw_hits={len(hits)}"
        )
        recall_lines = [recall_header, "  ── Recall Details ─────────────────────────────"]
        for idx, hit in enumerate(hits, start=1):
            origin_name = hit.get("name") or hit.get("text") or hit.get("id") or "<unnamed>"
            origin_id = hit.get("origin_id")
            origin_line = f"  [{idx}] {origin_name}"
            if origin_id is not None:
                origin_line += f" (origin_id={origin_id})"
            recall_lines.append(origin_line)
            chunks = hit.get("chunks", []) or []
            for chunk_idx, chunk in enumerate(chunks, start=1):
                chunk_type = chunk.get("type") or "-"
                chunk_score = _format_float(chunk.get("score"))
                snippet_raw = (chunk.get("chunk_text") or "").replace("\n", " ")
                snippet = snippet_raw[:40].strip()
                if len(snippet_raw) > 40:
                    snippet += "…"
                recall_lines.append(
                    f"      - chunk#{chunk_idx} type={chunk_type} sim={chunk_score} text=\"{snippet}\""
                )
            if not chunks:
                recall_lines.append("      - <no chunks>")
        logger.info("\n".join(recall_lines))

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

        log_lines = [f"query: {payload.query.strip()}"]
        if hits:
            for idx, hit in enumerate(hits, start=1):
                name = hit.get("name") or hit.get("id") or "<unnamed>"
                sim_score = hit.get("score")
                cross_score = hit.get("rerank_score")
                sim_str = f"{sim_score:.3f}" if isinstance(sim_score, (int, float)) else "-"
                cross_str = f"{cross_score:.3f}" if isinstance(cross_score, (int, float)) else "-"
                log_lines.append(
                    f"  {idx}. {name} | sim={sim_str} | cross={cross_str}"
                )
        else:
            log_lines.append("  <no hits>")
        logger.info("\n".join(log_lines))

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


def _origin_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "-"


@lru_cache(maxsize=1)
def get_engine() -> RetrievalEngine:
    return RetrievalEngine(
        index_path=_env_path("VECTOR_INDEX_PATH", DEFAULT_INDEX_PATH),
        meta_path=_env_path("VECTOR_META_PATH", DEFAULT_META_PATH),
        source_meta_path=_env_path("RECIPE_SOURCE_META_PATH", DEFAULT_SOURCE_META_PATH),
        embedding_model_name=os.getenv("VECTOR_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        device=os.getenv("VECTOR_DEVICE", DEFAULT_DEVICE),
        rerank_batch_size=int(os.getenv("VECTOR_RERANK_BATCH_SIZE", DEFAULT_RERANK_BATCH_SIZE)),
    )


app = FastAPI(title="Recipe Retrieval API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源调用
    allow_credentials=True,  # 若需携带凭证（如Token、Cookie）则设为True
    allow_methods=["*"],     # 允许所有HTTP方法（GET、POST、PUT等）
    allow_headers=["*"],     # 允许所有请求头（如Content-Type、Authorization）
)

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
    print("Received search/docs request at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    engine = get_engine()
    response = engine.search(payload)
    return response.results


@app.post("/reload")
def reload_index():
    engine = get_engine()
    engine.reload()
    return {"status": "reloaded", "index_size": engine.index_size}


__all__ = ["app", "get_engine", "RetrievalEngine"]
