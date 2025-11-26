"""Shared helpers for recipe recall evaluation tests."""
from __future__ import annotations

import json
import random
import time
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from label_config import LabelingTestConfig

PROMPT_TEMPLATE = (
    "你是一个菜谱检索系统的数据标注助手。\n"
    "请阅读下方菜谱片段，为它生成一个用户可能会问的简短中文问题，"
    "这个问题应该以该菜谱为最优答案之一。\n"
    "请只返回如下JSON格式：{{\"query\": \"...\"}}，不要添加其他解释。\n\n"
    "菜谱片段：\n{text}\n"
)


@dataclass
class AnnotationResult:
    query: str
    origin_id: int


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0

    def update(self, usage_payload: Optional[Dict[str, Any]]) -> None:
        if not usage_payload:
            return
        self.prompt_tokens += int(usage_payload.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage_payload.get("completion_tokens") or 0)
        self.total_tokens += int(usage_payload.get("total_tokens") or 0)
        self.requests += 1

    def to_dict(self) -> Dict[str, Any]:
        if not self.requests:
            avg_prompt = avg_completion = avg_total = 0.0
        else:
            avg_prompt = self.prompt_tokens / self.requests
            avg_completion = self.completion_tokens / self.requests
            avg_total = self.total_tokens / self.requests
        payload = asdict(self)
        payload.update(
            {
                "avg_prompt_tokens": round(avg_prompt, 2),
                "avg_completion_tokens": round(avg_completion, 2),
                "avg_total_tokens": round(avg_total, 2),
            }
        )
        return payload


class LabelingClient:
    """Minimal HTTP client for calling a chat-completions style API."""

    def __init__(self, config: LabelingTestConfig) -> None:
        if not config.label_api_key:
            raise ValueError("label_api_key is empty; update tests/label_config.py before running this test.")
        self.config = config
        self.token_usage = TokenUsage()

    def label(self, chunk: Dict[str, Any]) -> AnnotationResult:
        prompt = PROMPT_TEMPLATE.format(text=chunk.get("text", "").strip())
        payload = {
            "model": self.config.label_model_name,
            "temperature": self.config.label_temperature,
            "max_tokens": self.config.label_max_tokens,
            "messages": [
                {"role": "system", "content": "你是精确、遵守指令的数据标注助手。"},
                {"role": "user", "content": prompt},
            ],
        }
        response_body = self._post_json(payload)
        query_text = self._extract_query(response_body)
        return AnnotationResult(query=query_text, origin_id=chunk.get("origin_id"))

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            str(self.config.label_api_base),
            data=data,
            headers={
                "Authorization": f"Bearer {self.config.label_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.label_timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        self.token_usage.update(parsed.get("usage"))
        return parsed

    @staticmethod
    def _extract_query(api_response: Dict[str, Any]) -> str:
        choices = api_response.get("choices") or []
        if not choices:
            raise ValueError("LLM response did not include choices")
        content = choices[0].get("message", {}).get("content", "").strip()
        if not content:
            raise ValueError("LLM response message is empty")
        try:
            parsed = json.loads(content)
            candidate = parsed.get("query") or parsed.get("Query")
        except json.JSONDecodeError:
            candidate = content.splitlines()[0]
        candidate = (candidate or "").strip()
        if not candidate:
            raise ValueError("Failed to parse query text from LLM response")
        return candidate


def sample_meta_items(meta: List[Dict[str, Any]], cfg: LabelingTestConfig) -> List[Dict[str, Any]]:
    if not meta:
        return []
    by_origin: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in meta:
        origin_id = chunk.get("origin_id")
        if origin_id is None:
            continue
        by_origin.setdefault(str(origin_id), []).append(chunk)
    unique_origins = list(by_origin.keys())
    if not unique_origins:
        return []
    ratio = min(max(cfg.sample_ratio, 0.0), 1.0)
    rng = random.Random(cfg.random_seed)
    target = max(1, int(len(unique_origins) * ratio))
    target = min(target, cfg.max_samples, len(unique_origins))
    selected_origins = rng.sample(unique_origins, target)
    samples: List[Dict[str, Any]] = []
    for origin in selected_origins:
        candidates = by_origin[origin]
        samples.append(rng.choice(candidates))
    return samples


def chunk_cache_key(chunk: Dict[str, Any]) -> Optional[str]:
    for key in ("chunk_id", "id"):
        value = chunk.get(key)
        if value is not None:
            return str(value)
    return None


def load_label_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[warn] Failed to parse cache file {path}, starting fresh", flush=True)
        return {}


def save_label_cache(path: Path, data: Dict[str, Dict[str, Any]]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_token_usage(path: Path, usage: TokenUsage) -> None:
    if not path:
        return
    payload = usage.to_dict()
    payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def annotate_chunks(
    chunks: Iterable[Dict[str, Any]],
    client: LabelingClient,
    cfg: LabelingTestConfig,
) -> List[AnnotationResult]:
    chunk_list = list(chunks)
    cache = load_label_cache(cfg.label_cache_path)
    annotations: List[AnnotationResult] = []
    iterator: Iterable[Dict[str, Any]]
    if tqdm is not None:
        iterator = tqdm(chunk_list, desc="LLM labeling", unit="chunk")
    else:
        iterator = chunk_list
    updated_cache = False
    for chunk in iterator:
        cache_key = chunk_cache_key(chunk)
        cached_entry = cache.get(cache_key) if (cache_key and cache) else None
        if cached_entry:
            annotations.append(
                AnnotationResult(
                    query=str(cached_entry.get("query", "")),
                    origin_id=int(cached_entry.get("origin_id", chunk.get("origin_id", 0))),
                )
            )
            continue
        annotation = client.label(chunk)
        annotations.append(annotation)
        if cache_key:
            cache[cache_key] = {
                "query": annotation.query,
                "origin_id": annotation.origin_id,
            }
            updated_cache = True
    if updated_cache:
        save_label_cache(cfg.label_cache_path, cache)
    return annotations


SearchFn = Callable[[str, int], List[Dict[str, Any]]]


def load_cached_annotations(cfg: LabelingTestConfig) -> List[AnnotationResult]:
    cache = load_label_cache(cfg.label_cache_path)
    if not cache:
        return []
    items: List[AnnotationResult] = []
    for entry in cache.values():
        query = str((entry or {}).get("query", "")).strip()
        origin_id = entry.get("origin_id")
        if not query:
            continue
        try:
            origin_int = int(origin_id)
        except (TypeError, ValueError):
            continue
        items.append(AnnotationResult(query=query, origin_id=origin_int))
    if not items:
        return []
    print(
        f"[cache] Loaded {len(items)} annotated queries from {cfg.label_cache_path}",
        flush=True,
    )
    ratio = min(max(cfg.sample_ratio, 0.0), 1.0)
    rng = random.Random(cfg.random_seed)
    target = max(1, int(len(items) * ratio))
    target = min(target, cfg.max_samples, len(items))
    if target >= len(items):
        return items
    return rng.sample(items, target)


def evaluate_hits(
    annotations: List[AnnotationResult],
    cfg: LabelingTestConfig,
    search_fn: SearchFn,
) -> Dict[str, float]:
    if not annotations:
        return {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "mrr": 0.0, "retrieval_rate": 0.0}
    total = len(annotations)
    k_values: Sequence[int] = sorted({k for k in getattr(cfg, "eval_k_values", (1, 5)) if k > 0}) or [1]
    hits_by_k = {k: 0 for k in k_values}
    reciprocal_rank_sum = 0.0
    retrieved = 0
    search_k = max([cfg.search_top_k, max(k_values)])
    total_latency = 0.0
    iterator: Iterable[tuple[int, AnnotationResult]]
    if tqdm is not None:
        iterator = tqdm(
            enumerate(annotations, start=1),
            desc="Vector search",
            unit="query",
            total=total,
        )
    else:
        iterator = enumerate(annotations, start=1)
    for idx_counter, ann in iterator:
        start = time.perf_counter()
        results = search_fn(ann.query, search_k)
        total_latency += time.perf_counter() - start
        if not results:
            continue
        retrieved += 1
        rank = None
        for idx, item in enumerate(results, start=1):
            if item.get("origin_id") == ann.origin_id:
                rank = idx
                break
        if rank is None:
            continue
        for k in k_values:
            if rank <= k:
                hits_by_k[k] += 1
        reciprocal_rank_sum += 1.0 / rank
        if tqdm is None:
            print(f"[eval] processed {idx_counter}/{total}", end="\r", flush=True)
    if tqdm is None:
        print(flush=True)
    metrics = {f"hit@{k}": hits_by_k[k] / total for k in k_values}
    metrics["mrr"] = reciprocal_rank_sum / total
    metrics["retrieval_rate"] = retrieved / total
    metrics["avg_query_seconds"] = total_latency / total
    return metrics


__all__ = [
    "AnnotationResult",
    "LabelingClient",
    "TokenUsage",
    "annotate_chunks",
    "load_cached_annotations",
    "evaluate_hits",
    "sample_meta_items",
    "save_token_usage",
]
