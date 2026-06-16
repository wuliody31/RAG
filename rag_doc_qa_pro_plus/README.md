# RAG Doc QA Pro：检索增强生成文档问答系统

这是一个完整的、工程化的 RAG 文档问答系统模板，适合课程项目、毕业设计原型和作品集展示。系统支持 PDF、DOCX、TXT、Markdown、CSV、XLSX、HTML 文档入库，并提供：

- FastAPI 后端接口
- Streamlit 简易前端
- 文档解析与切块
- 向量检索 ChromaDB 持久化存储
- BM25 + Dense hybrid retrieval
- Reciprocal Rank Fusion 结果融合
- CrossEncoder reranker 重排
- 严格基于文档的回答生成
- 引用来源与页码/文件名返回
- Ollama 本地模型或 OpenAI API 可切换
- 命令行 ingestion / ask 脚本
- 检索评估脚本
- Dockerfile 与 docker-compose
- 文档细节检索：精确短语、页码、章节、chunk 邻接上下文
- 模型训练：构造 query-positive-negative 数据，微调 embedding 和 reranker

## 1. 环境安装

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -e .
cp .env.example .env
```

## 2. 本地 LLM 方式：Ollama

安装 Ollama 后运行：

```bash
ollama pull qwen2.5:7b-instruct
ollama serve
```

`.env` 保持：

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b-instruct
```

## 3. OpenAI API 方式

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=你的key
OPENAI_MODEL=gpt-4.1-mini
```

## 4. 启动后端

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

API 文档：

```text
http://localhost:8000/docs
```

## 5. 启动前端

```bash
streamlit run ui/streamlit_app.py
```

## 6. 命令行入库

把文档放进 `data/raw/`：

```bash
python scripts/ingest_cli.py --path data/raw
```

## 7. 命令行问答

```bash
python scripts/ask_cli.py "这份文档的核心结论是什么？"
```

## 8. API 示例

### 上传并入库

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "files=@./data/raw/example.pdf"
```

### 提问

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"文档里如何定义 RAG？", "filters": null}'
```

## 9. 评估格式

`data/eval/questions.jsonl`：

```jsonl
{"question":"问题1", "expected_doc_ids":["doc_id_a"], "answer_keywords":["关键词1", "关键词2"]}
```

运行：

```bash
python -m app.evaluation.evaluate --dataset data/eval/questions.jsonl
```

## 10. 文档细节检索

```bash
python scripts/detail_search_cli.py "maintenance strategy" --mode hybrid --top-k 10 --neighbors
python scripts/detail_search_cli.py "specific phrase" --mode exact --doc-id <doc_id>
```

API：

```text
POST /search/detail
POST /chunks/lookup
GET  /documents/{doc_id}/details
GET  /documents/{doc_id}/chunks
```

支持 `hybrid / semantic / keyword / exact` 四种检索模式，并支持 `doc_id`、`file_name`、`page_from`、`page_to` 和 `section` 过滤。

## 11. 模型训练

先根据评估集构造训练样本：

```bash
python scripts/build_training_data.py --eval data/eval/questions.jsonl --output data/training/retrieval_pairs.jsonl --negatives 3
```

微调 embedding 模型：

```bash
python scripts/train_embedder.py --train data/training/retrieval_pairs.jsonl --output data/models/embedding_finetuned --epochs 1 --batch-size 4
```

微调 reranker：

```bash
python scripts/train_reranker.py --train data/training/retrieval_pairs.jsonl --output data/models/reranker_finetuned --epochs 1 --batch-size 4
```

训练后在 `.env` 中设置：

```env
EMBEDDING_MODEL=./data/models/embedding_finetuned
RERANKER_MODEL=./data/models/reranker_finetuned
```

注意：换 embedding 模型后要重新入库文档。

## 12. 项目结构

```text
rag_doc_qa_pro/
├── app/
│   ├── api/              # FastAPI 路由与接口模型
│   ├── core/             # 配置、日志、通用数据结构
│   ├── ingestion/        # 文档解析、切块、入库
│   ├── retrieval/        # 向量库、BM25、hybrid retrieval、reranker、detail retriever
│   ├── llm/              # Ollama/OpenAI/mock LLM provider
│   ├── services/         # RAG 编排服务
│   ├── evaluation/       # 检索与回答评估
│   ├── training/         # 训练数据构建、embedding/reranker 微调
│   └── utils/            # 文件工具
├── scripts/              # CLI 工具
├── ui/                   # Streamlit 前端
├── tests/                # 单元测试
├── data/                 # 文档、上传、向量库、评估数据
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## 13. 技术设计文档

详细中文技术设计见：`docs/ARCHITECTURE_CN.md` 和 `docs/TRAINING_AND_DETAIL_RETRIEVAL_CN.md`。

## 14. 设计亮点

1. 不强依赖 LangChain，核心流程透明，方便写报告和调试。
2. 向量召回 + BM25 关键词召回，适合技术文档、学术论文、报告等场景。
3. 使用 CrossEncoder 进行二阶段重排，提升最终上下文质量。
4. Prompt 明确要求不能编造，并强制输出引用。
5. 每个 chunk 都保留文件名、页码、chunk id，方便溯源。
6. 配置集中在 `.env`，可替换 embedding、reranker、LLM 模型。
7. 训练模块支持 hard negative mining、embedding fine-tuning 和 reranker fine-tuning。
8. 细节检索模块支持页码/章节过滤和 chunk neighbor expansion，适合调试引用和做文档审查。
