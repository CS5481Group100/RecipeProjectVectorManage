# RecipeProjectVectorManage
For managing vector database


Usage
-----

1. Install dependencies (recommend using a virtualenv):

```bash
python -m pip install -r requirements.txt
```

2. Build the FAISS index from your JSON data (expects a JSON array of objects with `id`,`name`,`text`):

```bash
python -m vector_store.index_builder --data origin_data/recipes_cleaned.json --index data/index.faiss --meta data/meta.json
```

3. Query the index:

直接召回
```bash
python -m vector_store.query "天冷了我想吃羊肉，羊肉怎么做？" --index data/index.faiss --meta data/meta.json --k 5
```

重排 - cross encoder
```bash
python -m vector_store.query "我想学怎么做肉夹馍" --use-rerank --rerank-mode cross --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 --k 50 --rerank-top-k 5
```

重排 - bi encoder
```bash
python -m vector_store.query "我想学怎么做肉夹馍" --use-rerank --rerank-mode bi --rerank-model shibing624/text2vec-base-chinese --k 50 --rerank-top-k 5
```

FastAPI 服务
------------
1. 启动服务（默认读取 `data/index.faiss` & `data/meta.json`，可用环境变量覆盖）：

```bash
uvicorn service:app --reload --host 0.0.0.0 --port 8000
```

常用环境变量：
- `VECTOR_INDEX_PATH` / `VECTOR_META_PATH`: 指定索引与元数据位置
- `VECTOR_EMBEDDING_MODEL`: 召回向量模型
- `VECTOR_DEVICE`: `cpu` / `cuda` / `mps`
- `VECTOR_RERANK_BATCH_SIZE`: cross-encoder 批量

2. 请求接口：

```bash
curl -X POST "http://localhost:8000/search" \
	-H "Content-Type: application/json" \
	-d '{
				"query": "天冷了我想吃羊肉",
				"k": 10,
				"use_rerank": true,
				"rerank_mode": "cross",
				"rerank_top_k": 5
			}'
```

		若只需要召回的文档列表（不含统计信息），可以调用 `/search/docs`：

		```bash
		curl -X POST "http://localhost:8000/search/docs" \
		  -H "Content-Type: application/json" \
		  -d '{
			  "query": "天冷了我想吃羊肉",
			  "k": 10
			}'
		```

3. 重新加载索引（无需重启服务）：

```bash
curl -X POST http://localhost:8000/reload
```

4. 内置网页：启动后访问 [http://localhost:8000/](http://localhost:8000/) 即可使用自带的检索界面，支持输入 query、设置 top-k、选择 cross/bi 重排并阅读结果。页面纯前端实现，无需额外依赖。


Design notes
------------
- Embeddings: uses `sentence-transformers` (default `paraphrase-multilingual-MiniLM-L12-v2`).
- Index: FAISS `IndexFlatIP` over L2-normalized vectors (so dot product = cosine similarity).
- Query: encode query with same model and normalize, then search top-k.
- Extensibility: there's a `vector_store.reranker.Reranker` stub for adding cross-encoder reranking later, and the code keeps metadata order aligned with FAISS index so you can replace or augment the retrieval pipeline.
