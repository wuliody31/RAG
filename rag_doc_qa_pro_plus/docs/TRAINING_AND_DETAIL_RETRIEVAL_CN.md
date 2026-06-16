# 训练模块与文档细节检索设计

## 1. 新增目标

本版本在原有 RAG 文档问答系统基础上加入两类能力：

1. **模型训练能力**：从评估集和已入库文档中构造 query-positive-negative 训练样本，用于微调 embedding bi-encoder 和 CrossEncoder reranker。
2. **文档细节检索能力**：不仅回答问题，还可以按照关键词、语义、精确短语、页码、章节、doc_id、file_name 检索原始文档片段，并返回相邻 chunk 作为阅读上下文。

## 2. 文档细节检索

新增模块：

```text
app/retrieval/detail_retriever.py
```

支持四种检索模式：

| mode | 说明 |
|---|---|
| hybrid | dense semantic search + BM25 + exact matching + RRF fusion |
| semantic | 使用 embedding 和 ChromaDB 做语义检索 |
| keyword | 使用 BM25 做关键词检索 |
| exact | 精确短语/词项匹配，适合查具体术语、编号、公式、设备参数 |

新增 API：

```text
POST /search/detail
POST /chunks/lookup
GET  /documents/{doc_id}/details
GET  /documents/{doc_id}/chunks
```

### 2.1 示例：精确检索某个术语

```bash
curl -X POST http://localhost:8000/search/detail \
  -H "Content-Type: application/json" \
  -d '{
    "query":"hydrogen storage pressure",
    "mode":"exact",
    "top_k":10,
    "include_neighbors":true,
    "neighbor_window":1,
    "full_text":true
  }'
```

### 2.2 示例：只在某个文档第 3-5 页检索

```json
{
  "query": "maintenance strategy",
  "mode": "hybrid",
  "doc_id": "你的doc_id",
  "page_from": 3,
  "page_to": 5,
  "include_neighbors": true
}
```

## 3. 训练数据构建

新增模块：

```text
app/training/dataset_builder.py
```

输入评估集格式：

```jsonl
{"question":"文档的主要结论是什么？", "expected_doc_ids":["doc_id_a"], "answer_keywords":["keyword1", "keyword2"]}
```

输出训练数据格式：

```jsonl
{"query":"...", "positive":"相关chunk", "negative":"困难负样本chunk"}
```

构建命令：

```bash
python scripts/build_training_data.py \
  --eval data/eval/questions.jsonl \
  --output data/training/retrieval_pairs.jsonl \
  --negatives 3
```

也可通过 API：

```text
POST /training/dataset/build
```

## 4. Embedding 模型微调

新增模块：

```text
app/training/train_embedder.py
```

使用 `SentenceTransformer` + `MultipleNegativesRankingLoss` 微调双塔 embedding 模型。训练后得到的模型目录可以直接作为 `EMBEDDING_MODEL` 使用。

```bash
python scripts/train_embedder.py \
  --train data/training/retrieval_pairs.jsonl \
  --output data/models/embedding_finetuned \
  --epochs 1 \
  --batch-size 4
```

训练完成后，在 `.env` 中设置：

```env
EMBEDDING_MODEL=./data/models/embedding_finetuned
```

然后需要**重新入库文档**，因为向量库中的 embedding 需要用新模型重新计算。

## 5. Reranker 微调

新增模块：

```text
app/training/train_reranker.py
```

使用 `CrossEncoder` 对 query-document pair 做二分类/相关性打分训练。训练后在 `.env` 中设置：

```env
RERANKER_MODEL=./data/models/reranker_finetuned
```

启动训练：

```bash
python scripts/train_reranker.py \
  --train data/training/retrieval_pairs.jsonl \
  --output data/models/reranker_finetuned \
  --epochs 1 \
  --batch-size 4
```

## 6. 工程价值

这次扩展使项目从普通 RAG demo 升级为更完整的 LLM/NLP 工程系统：

- 支持文档级、页级、chunk 级细节检索；
- 支持精确匹配、BM25、语义检索和 hybrid 检索；
- 支持相邻 chunk 扩展，便于定位上下文；
- 支持从评估集挖掘 hard negatives；
- 支持微调 embedding 模型和 reranker；
- 支持通过 API、CLI 和 Streamlit UI 操作。
