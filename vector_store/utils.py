import numpy as np
from typing import Any, Dict, Iterable, List, Tuple


def l2_normalize_vectors(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # Ensure computations happen in float32 to avoid upcasting to float64 which
    # causes FAISS to reject the array. Return a C-contiguous float32 array.
    v32 = v.astype('float32', copy=False)
    norms = np.linalg.norm(v32, axis=1, keepdims=True)
    norms = np.maximum(norms, eps).astype('float32', copy=False)
    out = v32 / norms
    return np.ascontiguousarray(out, dtype='float32')


def to_float32(v: np.ndarray) -> np.ndarray:
    return v.astype('float32')


_FIELD_LABELS: List[Tuple[str, str]] = [
    ("name", "菜名"),
    ("dish", "菜品"),
    ("description", "描述"),
    ("recipeIngredient", "原料"),
    ("recipeInstructions", "做法"),
    ("keywords", "关键词"),
]


def _format_list(value: Iterable[Any], prefix: str = "- ") -> str:
    items = [str(item).strip() for item in value if str(item).strip()]
    if not items:
        return ""
    return "\n".join(f"{prefix}{item}" for item in items)


def _format_instructions(value: Iterable[Any]) -> str:
    steps = [str(item).strip() for item in value if str(item).strip()]
    if not steps:
        return ""
    return "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps))


def render_source_text(source: Any) -> str:
    """Convert a recipe source dict into human-readable text."""
    if not isinstance(source, dict):
        return ""
    lines: List[str] = []
    for field, label in _FIELD_LABELS:
        if field not in source:
            continue
        value = source[field]
        formatted = ""
        if value is None:
            formatted = ""
        elif isinstance(value, str):
            formatted = value.strip()
        elif field == "recipeIngredient":
            formatted = _format_list(value)
        elif field == "recipeInstructions":
            formatted = _format_instructions(value)
        elif isinstance(value, Iterable):
            formatted = _format_list(value, prefix="")
        else:
            formatted = str(value)
        if formatted:
            lines.append(f"{label}：\n{formatted}" if "\n" in formatted else f"{label}：{formatted}")

    remaining_keys = [k for k in source.keys() if k not in {f for f, _ in _FIELD_LABELS}]
    for key in remaining_keys:
        value = source[key]
        if value in (None, ""):
            continue
        if isinstance(value, (list, tuple, set)):
            formatted = _format_list(value, prefix="")
        else:
            formatted = str(value)
        if formatted:
            lines.append(f"{key}：{formatted}")

    return "\n\n".join(lines)
