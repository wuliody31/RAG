# RAG Document QA System

本项目是一个面向学术论文场景的本地化 RAG 文档问答系统，用于缓解长文档问答中的幻觉问题，并提升回答的证据可追溯性。

## 项目功能
- PDF 文档解析与清洗
- Metadata-aware chunking
- SentenceTransformer 向量检索
- BM25 关键词检索
- Exact Phrase Matching
- RRF 检索结果融合
- CrossEncoder Reranker 重排序
- Ollama 本地大模型生成
- Citation-grounded generation

## 技术栈
Python, Streamlit, SentenceTransformers, BM25, CrossEncoder, Ollama, Docker

## 项目结构
