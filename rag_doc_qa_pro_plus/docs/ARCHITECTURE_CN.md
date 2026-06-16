# RAG 文档问答系统技术设计说明

## 1. 项目目标

本项目实现一个完整的检索增强生成系统，用于对本地或上传文档进行语义问答。用户上传 PDF、Word、Markdown、表格或网页文本后，系统将文档解析、清洗、切块、向量化，并写入持久化向量数据库。用户提问时，系统先进行混合检索，再用重排模型筛选最相关证据，最后由大语言模型基于证据生成带引用的回答。

## 2. 系统总体流程

```text
Document Upload / Folder Ingestion
        ↓
Document Loader: PDF / DOCX / TXT / MD / CSV / XLSX / HTML
        ↓
Text Cleaning + Metadata Extraction
        ↓
Recursive Chunking with Overlap
        ↓
Embedding Model
        ↓
ChromaDB Persistent Vector Store
        ↓
User Question
        ↓
Dense Retrieval + BM25 Retrieval
        ↓
Reciprocal Rank Fusion
        ↓
CrossEncoder Reranking
        ↓
Prompt Construction with Evidence Blocks
        ↓
LLM Generation
        ↓
Answer + Citations + Source Metadata
```

## 3. 模块划分

### 3.1 API 层

`app/api/main.py` 使用 FastAPI 提供 RESTful 接口：

- `/health`：检查系统状态、chunk 数量和当前 LLM provider。
- `/documents/upload`：上传文档并自动入库。
- `/documents/ingest-folder`：将本地文件夹中的文档批量入库。
- `/documents`：列出已入库文档。
- `/documents/{doc_id}`：删除指定文档。
- `/ask`：执行完整 RAG 问答。

### 3.2 文档解析层

`app/ingestion/loaders.py` 支持：

- PDF：PyMuPDF 按页提取文本。
- DOCX：python-docx 提取段落和表格。
- TXT / Markdown：多编码兼容读取。
- CSV / Excel：pandas 转为 markdown 表格。
- HTML：BeautifulSoup 去除脚本、样式、导航等噪声。

### 3.3 切块层

`app/ingestion/chunker.py` 使用递归切块策略：

1. 优先按段落切分。
2. 段落过长时按句子切分。
3. 仍过长时按字符窗口切分。
4. 使用 overlap 保留跨 chunk 上下文。

每个 chunk 保存以下 metadata：

- doc_id
- file_name
- source_path
- file_type
- page
- chunk_seq
- text_hash

### 3.4 向量化与存储层

`app/retrieval/embeddings.py` 使用 SentenceTransformers 生成 normalized embedding。

`app/retrieval/vector_store.py` 使用 ChromaDB PersistentClient，支持：

- chunk upsert
- dense query
- metadata filter
- list documents
- delete by doc_id

### 3.5 检索层

`app/retrieval/hybrid_retriever.py` 是核心检索管线：

1. Dense Retrieval：适合语义相似问题。
2. BM25 Retrieval：适合关键词、编号、术语、公式类问题。
3. Reciprocal Rank Fusion：融合两种召回结果。
4. CrossEncoder Reranking：对候选 chunk 进行 query-document pair 级别打分。
5. 返回最终上下文给生成模块。

### 3.6 生成层

`app/llm/` 中抽象出 LLM provider：

- `OllamaClient`：本地模型，适合隐私和低成本场景。
- `OpenAIClient`：API 模型，适合更高质量生成。
- `MockLLMClient`：无模型环境下用于接口测试。

`app/services/prompts.py` 中的系统提示词要求：

- 只基于上下文回答。
- 证据不足时明确说明。
- 不编造引用。
- 回答中使用 `[1]`、`[2]` 等引用标记。

## 4. 为什么这是“顶配”RAG 结构

普通 RAG 往往只有“向量检索 + 直接生成”，容易出现召回不准、错过关键词、引用不清楚、幻觉等问题。本项目加入了完整工程链路：

- 多格式文档解析。
- chunk 级 metadata 溯源。
- hybrid retrieval。
- RRF 融合。
- CrossEncoder reranking。
- 引用生成。
- 多 LLM provider。
- API + UI + CLI。
- 评估脚本。
- Docker 部署。

## 5. 可配置参数

在 `.env` 中可以调整：

```env
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
CHUNK_SIZE=900
CHUNK_OVERLAP=150
DENSE_TOP_K=30
BM25_TOP_K=30
RERANK_TOP_K=8
FINAL_CONTEXT_K=6
LLM_PROVIDER=ollama
```

## 6. 评估指标

当前提供基础评估：

- Recall@3
- Recall@5
- Answer keyword coverage

后续可以扩展：

- MRR
- nDCG
- faithfulness judge
- context precision
- answer relevance
- latency / cost evaluation

## 7. 可以写进报告的技术贡献

1. 设计并实现了端到端 RAG 文档问答系统。
2. 使用混合检索提升文档召回覆盖率。
3. 引入 CrossEncoder reranking 提升最终上下文相关性。
4. 通过 metadata 和引用机制增强回答可解释性。
5. 支持本地和云端 LLM provider，可在隐私、成本和性能之间切换。
6. 提供 API、前端、CLI、评估和 Docker，体现完整软件工程实践。

## 8. v1.1 扩展：训练与细节检索

### 8.1 Detail Retriever

`DetailRetriever` 是面向文档审查和证据定位的检索层。它和标准 `HybridRetriever` 的区别是：

- 标准 RAG 检索：目标是给 LLM 提供少量高质量上下文；
- 细节检索：目标是让用户定位文档原文、页码、chunk、章节和相邻上下文。

新增检索路径：

```text
query
 ├─ exact phrase matching
 ├─ BM25 keyword retrieval
 ├─ dense semantic retrieval
 └─ Reciprocal Rank Fusion + optional CrossEncoder rerank
       ↓
page/doc/section filters
       ↓
neighbor chunk expansion
       ↓
full-text evidence response
```

### 8.2 Training Pipeline

训练模块采用轻量但完整的监督数据闭环：

```text
evaluation questions + indexed chunks
        ↓
hybrid retrieval candidate pool
        ↓
positive mining by expected_doc_ids / answer_keywords
        ↓
hard negative mining from retrieved but irrelevant chunks
        ↓
retrieval_pairs.jsonl
        ↓
embedding fine-tuning / reranker fine-tuning
```

训练后的模型输出目录可以直接写入 `.env`：

```env
EMBEDDING_MODEL=./data/models/embedding_finetuned
RERANKER_MODEL=./data/models/reranker_finetuned
```

如果替换 embedding 模型，需要清空或更换 Chroma collection 并重新入库文档，保证文档向量与查询向量来自同一模型空间。
