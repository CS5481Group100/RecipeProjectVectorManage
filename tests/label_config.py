"""Configuration for synthetic labeling + recall evaluation tests.

Update the values in this file to match the model/API you want to call.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class LabelingTestConfig:
    """Holds settings for the annotation model and evaluation dataset."""

    # --- Annotation model settings ---
    label_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    label_api_key: str = "sk-wjrtizmtakyahakiovtuqynxrvzaafpcbrxddfdlutaglfhj"
    label_api_base: str = "https://api.siliconflow.cn/v1/chat/completions"
    label_temperature: float = 0.3
    label_max_tokens: int = 128
    label_timeout_seconds: int = 60

    # --- Sampling settings ---
    sample_ratio: float = 1  # 20% of the dataset
    max_samples: int = 2000  # Safety cap for very large datasets
    random_seed: int = 5481

    # --- Label caching ---
    label_cache_path: Path = Path("tests/label_cache2.json")
    token_usage_log_path: Path = Path("tests/label_usage2.json")

    # --- Vector store inputs ---
    index_path: Path = Path("data/indx.faiss")
    meta_path: Path = Path("data/meta.json")
    search_top_k: int = 5
    eval_k_values: Tuple[int, ...] = (1, 3, 5)

    # --- Service recall settings ---
    service_base_url: str = "http://localhost:8000"
    service_search_path: str = "/search/docs"
    service_use_rerank: bool = True
    service_rerank_mode: str = "cross"
    service_rerank_top_k: Optional[int] = None
    service_timeout_seconds: int = 30
    service_extra_headers: Dict[str, str] = field(default_factory=dict)


CONFIG = LabelingTestConfig()
