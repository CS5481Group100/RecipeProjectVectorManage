import json
import os
from typing import List, Optional
from datetime import datetime

import faiss
import numpy as np
from tqdm import tqdm

from .embeddings import EmbeddingModel
from .utils import l2_normalize_vectors, to_float32
from .config import DEFAULT_EMBEDDING_MODEL

def build_index(
    data_path: str,
    index_path: str,
    meta_path: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    index_meta_path: Optional[str] = None,
    batch_size: int = 32,
    show_progress: bool = True,
):
    """Build a FAISS index from a JSON file containing a list of records with keys `id`,`name`,`text`.

    The embedding is computed over the concatenation `name + "\n" + text`.
    Saves FAISS index to `index_path` and metadata list to `meta_path` (JSON array in same order).
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    texts = []
    metas = []
    ids = []
    for it in items:
        _id = it.get('id')
        name = it.get('name', '')
        text = it.get('text', '')
        combined = f"{name}\n{text}"
        texts.append(combined)
        metas.append({'id': _id, 'name': name, 'text': text})
        ids.append(_id)

    model = EmbeddingModel(model_name=model_name)

    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), disable=not show_progress, desc='Embedding'):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, batch_size=batch_size, show_progress=show_progress)
        all_embs.append(embs)

    embs = np.vstack(all_embs)
    embs = to_float32(embs)
    embs = l2_normalize_vectors(embs)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    os.makedirs(os.path.dirname(index_path) or '.', exist_ok=True)
    faiss.write_index(index, index_path)

    # save metadata records
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    # write index-level metadata (which model was used, dim, count, timestamp)
    if index_meta_path is None:
        base, _ = os.path.splitext(index_path)
        index_meta_path = base + "_index_meta.json"

    index_meta = {
        "model_name": model_name,
        "data_path": data_path,
        "index_path": index_path,
        "meta_path": meta_path,
        "num_vectors": int(embs.shape[0]),
        "dim": int(embs.shape[1]),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(index_meta_path, 'w', encoding='utf-8') as f:
        json.dump(index_meta, f, ensure_ascii=False, indent=2)

    return index, metas, index_meta


if __name__ == '__main__':
    import typer
    from .config import DEFAULT_INDEX_PATH, DEFAULT_META_PATH, DEFAULT_EMBEDDING_MODEL

    app = typer.Typer()

    @app.command()
    def main(
        data: str = 'origin_data/recipes_cleaned.json',
        index: str = DEFAULT_INDEX_PATH,
        meta: str = DEFAULT_META_PATH,
        model: str = DEFAULT_EMBEDDING_MODEL,
        index_meta: Optional[str] = None,
    ):
        """Build index from data file and save index+metadata."""
        build_index(data, index, meta, model_name=model, index_meta_path=index_meta)

    app()
