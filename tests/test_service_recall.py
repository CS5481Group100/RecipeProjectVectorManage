"""Integration test that measures recall via the FastAPI service endpoints."""
from __future__ import annotations

import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urljoin
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from label_config import CONFIG
from recall_utils import (
    evaluate_hits,
    load_cached_annotations,
)


class ServiceSearchClient:
    def __init__(self, config):
        self.config = config
        base = config.service_base_url.rstrip('/') + '/'
        self.endpoint = urljoin(base, config.service_search_path.lstrip('/'))

    def search(self, query: str, k: int):
        payload = {
            "query": query,
            "k": k,
            "use_rerank": self.config.service_use_rerank,
            "rerank_mode": self.config.service_rerank_mode,
            "rerank_top_k": self.config.service_rerank_top_k or min(k, self.config.search_top_k),
        }
        if not payload["use_rerank"]:
            payload.pop("rerank_top_k", None)
        data = json.dumps(payload).encode('utf-8')
        headers = {"Content-Type": "application/json"}
        headers.update(self.config.service_extra_headers)
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.service_timeout_seconds) as resp:
            body = resp.read().decode('utf-8')
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            results = parsed.get("results") or []
        elif isinstance(parsed, list):
            results = parsed
        else:
            raise ValueError("Unexpected response format from service")
        if not isinstance(results, list):
            raise ValueError("Service response 'results' must be a list")
        return results


class ServiceRecallTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = CONFIG
        if not cls.config.meta_path.exists():
            raise unittest.SkipTest(f"Meta file not found: {cls.config.meta_path}")
        with cls.config.meta_path.open("r", encoding="utf-8") as f:
            cls.meta = json.load(f)
        if not cls.meta:
            raise unittest.SkipTest("Meta dataset is empty.")
        if not cls.config.label_api_key:
            raise unittest.SkipTest("Configure label_api_key in tests/label_config.py before running this test.")

    def test_service_hits_with_labeled_queries(self) -> None:
        annotations = load_cached_annotations(self.config)
        if not annotations:
            raise unittest.SkipTest("label_cache.json is empty; run vector recall test to generate annotations first.")
        service_client = ServiceSearchClient(self.config)

        def service_search(query: str, k: int):
            return service_client.search(query, k)

        try:
            metrics = evaluate_hits(annotations, self.config, service_search)
        except urllib.error.URLError as exc:
            self.skipTest(f"Service call failed: {exc}")
        self.assertIn("mrr", metrics)
        self.assertGreater(len(annotations), 0, "Expected at least one annotation")
        ordered_keys = [
            key
            for key in (
                *(f"hit@{k}" for k in sorted(getattr(self.config, "eval_k_values", (1, 5)))),
                "mrr",
                "retrieval_rate",
                "avg_query_seconds",
            )
            if key in metrics
        ]
        summary = {key: round(metrics[key], 4) for key in ordered_keys}
        summary["evaluated_samples"] = len(annotations)
        print("\n===== Service Recall Report =====", flush=True)
        for key, value in summary.items():
            print(f"{key:>18}: {value}", flush=True)


if __name__ == "__main__":
    unittest.main()
