from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(page_title="RAG Doc QA Pro", page_icon="📚", layout="wide")
DEFAULT_API_BASE = os.getenv("API_BASE", "http://localhost:8000")

with st.sidebar:
    API_BASE = st.text_input("API base URL", DEFAULT_API_BASE)
    st.header("Upload")
    files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md", "csv", "xlsx", "xls", "html", "htm"],
        accept_multiple_files=True,
    )
    if st.button("Index uploaded files", disabled=not files):
        multipart = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
        res = requests.post(f"{API_BASE}/documents/upload", files=multipart, timeout=600)
        if res.ok:
            st.success("Indexed")
            st.json(res.json())
        else:
            st.error(res.text)

    st.header("System")
    if st.button("Health check"):
        res = requests.get(f"{API_BASE}/health", timeout=30)
        st.json(res.json() if res.ok else res.text)

st.title("📚 RAG Doc QA Pro")
st.caption("Hybrid RAG + detail retrieval + trainable embedding/reranker pipeline")

ask_tab, detail_tab, docs_tab, train_tab = st.tabs(["Ask", "Detail Search", "Documents", "Training"])

with ask_tab:
    question = st.text_area("Ask a question", placeholder="例如：这份报告的主要结论是什么？")
    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving and generating..."):
            res = requests.post(f"{API_BASE}/ask", json={"question": question, "filters": None}, timeout=600)
        if not res.ok:
            st.error(res.text)
        else:
            data = res.json()
            st.subheader("Answer")
            st.write(data["answer"])
            st.subheader("Citations")
            for c in data["citations"]:
                with st.expander(
                    f"[{c['source_number']}] {c.get('file_name')} | page={c.get('page')} | score={c.get('score'):.4f}"
                ):
                    st.write(c.get("preview"))
                    st.code(c.get("chunk_id"))

with detail_tab:
    st.subheader("Fine-grained document retrieval")
    col_a, col_b, col_c = st.columns([3, 1, 1])
    with col_a:
        query = st.text_input("Search query / exact phrase", placeholder="输入关键词、句子或专业术语")
    with col_b:
        mode = st.selectbox("Mode", ["hybrid", "semantic", "keyword", "exact"])
    with col_c:
        top_k = st.number_input("Top K", min_value=1, max_value=100, value=10)

    col1, col2, col3, col4 = st.columns(4)
    doc_id = col1.text_input("doc_id filter", value="")
    file_name = col2.text_input("file_name filter", value="")
    page_from = col3.number_input("page from", min_value=0, value=0)
    page_to = col4.number_input("page to", min_value=0, value=0)

    col5, col6, col7 = st.columns(3)
    section = col5.text_input("section contains", value="")
    include_neighbors = col6.checkbox("Include neighboring chunks", value=True)
    rerank = col7.checkbox("Rerank", value=True)

    if st.button("Search details", type="primary", disabled=not query.strip()):
        payload = {
            "query": query,
            "mode": mode,
            "top_k": int(top_k),
            "doc_id": doc_id or None,
            "file_name": file_name or None,
            "page_from": int(page_from) if page_from else None,
            "page_to": int(page_to) if page_to else None,
            "section": section or None,
            "include_neighbors": include_neighbors,
            "neighbor_window": 1,
            "rerank": rerank,
            "full_text": True,
        }
        res = requests.post(f"{API_BASE}/search/detail", json=payload, timeout=600)
        if not res.ok:
            st.error(res.text)
        else:
            data = res.json()
            st.write(f"Total: {data['total']}")
            for c in data["chunks"]:
                title = f"{c['chunk_id']} | {c.get('file_name')} | page={c.get('page')} | score={c.get('score'):.4f} | {c.get('source')}"
                with st.expander(title):
                    st.write(c.get("text"))
                    st.json({k: c.get(k) for k in ["doc_id", "chunk_seq", "section_title", "source_path"]})

with docs_tab:
    st.subheader("Indexed documents")
    if st.button("Refresh documents"):
        res = requests.get(f"{API_BASE}/documents", timeout=60)
        if res.ok:
            st.dataframe(res.json().get("documents", []), use_container_width=True)
        else:
            st.error(res.text)

    st.subheader("Document profile / outline")
    profile_doc_id = st.text_input("Document ID for details")
    if st.button("Load document details", disabled=not profile_doc_id.strip()):
        res = requests.get(f"{API_BASE}/documents/{profile_doc_id}/details", timeout=60)
        if res.ok:
            data = res.json()
            st.json({k: data.get(k) for k in ["doc_id", "file_name", "file_type", "chunks", "page_count", "source_path"]})
            st.subheader("Sections")
            st.dataframe(data.get("sections", []), use_container_width=True)
            st.subheader("Preview")
            for item in data.get("preview", []):
                with st.expander(str(item.get("chunk_id"))):
                    st.write(item.get("preview"))
        else:
            st.error(res.text)

with train_tab:
    st.subheader("Model training pipeline")
    st.info("Training is optional and can be slow on CPU. For coursework/demo, use a small max_examples first.")

    st.markdown("### 1. Build training pairs")
    eval_path = st.text_input("Eval JSONL", "data/eval/questions.jsonl")
    output_path = st.text_input("Training output JSONL", "data/training/retrieval_pairs.jsonl")
    negatives = st.number_input("Negatives per query", min_value=1, max_value=20, value=3)
    if st.button("Build training dataset"):
        payload = {
            "eval_dataset_path": eval_path,
            "output_path": output_path,
            "negatives_per_query": int(negatives),
            "top_k_pool": 50,
        }
        res = requests.post(f"{API_BASE}/training/dataset/build", json=payload, timeout=600)
        st.json(res.json() if res.ok else res.text)

    st.markdown("### 2. Fine-tune embedding model")
    emb_out = st.text_input("Embedding output dir", "data/models/embedding_finetuned")
    max_examples = st.number_input("Max examples for quick test", min_value=0, value=20)
    if st.button("Train embedder"):
        payload = {
            "train_path": output_path,
            "output_dir": emb_out,
            "epochs": 1,
            "batch_size": 4,
            "max_examples": int(max_examples) if max_examples else None,
        }
        res = requests.post(f"{API_BASE}/training/embedder", json=payload, timeout=3600)
        st.json(res.json() if res.ok else res.text)

    st.markdown("### 3. Fine-tune reranker")
    rerank_out = st.text_input("Reranker output dir", "data/models/reranker_finetuned")
    if st.button("Train reranker"):
        payload = {
            "train_path": output_path,
            "output_dir": rerank_out,
            "epochs": 1,
            "batch_size": 4,
            "max_examples": int(max_examples) if max_examples else None,
        }
        res = requests.post(f"{API_BASE}/training/reranker", json=payload, timeout=3600)
        st.json(res.json() if res.ok else res.text)
