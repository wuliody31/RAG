from __future__ import annotations

import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.api.models import (
    AskRequest,
    AskResponse,
    ChunkLookupRequest,
    DetailSearchRequest,
    DetailSearchResponse,
    EmbedderTrainRequest,
    HealthResponse,
    RerankerTrainRequest,
    TrainResponse,
    TrainingDatasetBuildRequest,
    TrainingDatasetBuildResponse,
    UploadResponse,
)
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.ingestion.ingestor import Ingestor
from app.retrieval.detail_retriever import DetailRetriever, result_to_detail_dict
from app.retrieval.vector_store import VectorStore
from app.services.rag_service import RAGService
from app.training.dataset_builder import build_training_pairs
from app.training.train_embedder import train_embedder
from app.training.train_reranker import train_reranker
from app.utils.file_utils import SUPPORTED_EXTENSIONS, safe_copy_to_dir

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title="RAG Doc QA Pro",
    description="Hybrid retrieval + reranking + citation-based document QA system",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    store = VectorStore(settings)
    return HealthResponse(
        status="ok",
        chunks=store.count(),
        collection=settings.collection_name,
        llm_provider=settings.llm_provider,
    )


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    uploaded_paths: list[Path] = []
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for file in files:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
            temp_path = tmp_dir / (file.filename or f"upload{suffix}")
            content = await file.read()
            temp_path.write_bytes(content)
            uploaded_paths.append(safe_copy_to_dir(temp_path, settings.upload_dir))

    ingestor = Ingestor(settings)
    ingestion = {"documents": [], "chunks_added": 0, "files_seen": len(uploaded_paths)}
    for path in uploaded_paths:
        result = ingestor.ingest_file(path)
        ingestion["documents"].append(result)
        ingestion["chunks_added"] += result.get("chunks", 0)

    return UploadResponse(files_uploaded=len(uploaded_paths), ingestion=ingestion)


@app.post("/documents/ingest-folder")
def ingest_folder(path: str) -> dict:
    folder = Path(path)
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")
    return Ingestor(settings).ingest_path(folder)


@app.get("/documents")
def list_documents() -> dict:
    return {"documents": VectorStore(settings).list_documents()}


@app.get("/documents/{doc_id}/details")
def document_details(doc_id: str, preview_chunks: int = 5) -> dict:
    profile = VectorStore(settings).document_profile(doc_id, preview_chunks=preview_chunks)
    if not profile.get("found"):
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    return profile


@app.get("/documents/{doc_id}/chunks")
def document_chunks(doc_id: str, full_text: bool = False) -> dict:
    chunks = VectorStore(settings).get_document_chunks(doc_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"Document not found or has no chunks: {doc_id}")
    return {
        "doc_id": doc_id,
        "total": len(chunks),
        "chunks": [result_to_detail_dict(c, full_text=full_text) for c in chunks],
    }


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    VectorStore(settings).delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@app.post("/search/detail", response_model=DetailSearchResponse)
def detail_search(request: DetailSearchRequest) -> DetailSearchResponse:
    try:
        chunks = DetailRetriever(settings).search(
            query=request.query,
            mode=request.mode,
            top_k=request.top_k,
            doc_id=request.doc_id,
            file_name=request.file_name,
            page_from=request.page_from,
            page_to=request.page_to,
            section=request.section,
            include_neighbors=request.include_neighbors,
            neighbor_window=request.neighbor_window,
            rerank=request.rerank,
        )
        return DetailSearchResponse(
            query=request.query,
            mode=request.mode,
            total=len(chunks),
            chunks=[result_to_detail_dict(c, full_text=request.full_text) for c in chunks],
        )
    except Exception as exc:
        logger.exception("Detail search failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chunks/lookup", response_model=DetailSearchResponse)
def chunk_lookup(request: ChunkLookupRequest) -> DetailSearchResponse:
    chunks = DetailRetriever(settings).get_chunk_with_neighbors(request.chunk_id, window=request.neighbor_window)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"Chunk not found: {request.chunk_id}")
    return DetailSearchResponse(
        query=request.chunk_id,
        mode="chunk_lookup",
        total=len(chunks),
        chunks=[result_to_detail_dict(c, full_text=request.full_text) for c in chunks],
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        bundle = RAGService(settings).answer(request.question, filters=request.filters)
        return AskResponse(question=bundle.question, answer=bundle.answer, citations=bundle.citations)
    except Exception as exc:
        logger.exception("Ask failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/training/dataset/build", response_model=TrainingDatasetBuildResponse)
def build_training_dataset(request: TrainingDatasetBuildRequest) -> TrainingDatasetBuildResponse:
    try:
        report = build_training_pairs(
            settings=settings,
            eval_dataset_path=Path(request.eval_dataset_path),
            output_path=Path(request.output_path),
            negatives_per_query=request.negatives_per_query,
            top_k_pool=request.top_k_pool,
        )
        return TrainingDatasetBuildResponse(**report)
    except Exception as exc:
        logger.exception("Training dataset build failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/training/embedder", response_model=TrainResponse)
def run_embedder_training(request: EmbedderTrainRequest) -> TrainResponse:
    try:
        report = train_embedder(
            settings=settings,
            train_path=Path(request.train_path),
            base_model=request.base_model,
            output_dir=Path(request.output_dir),
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            max_examples=request.max_examples,
        )
        return TrainResponse(**report)
    except Exception as exc:
        logger.exception("Embedder training failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/training/reranker", response_model=TrainResponse)
def run_reranker_training(request: RerankerTrainRequest) -> TrainResponse:
    try:
        report = train_reranker(
            settings=settings,
            train_path=Path(request.train_path),
            base_model=request.base_model,
            output_dir=Path(request.output_dir),
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            max_examples=request.max_examples,
        )
        return TrainResponse(**report)
    except Exception as exc:
        logger.exception("Reranker training failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
