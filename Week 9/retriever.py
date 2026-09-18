"""Topic 2 (Retrieval tuning & reranking): a two-stage retriever.

Stage 1 (bi-encoder, FAISS): cheap, wide recall pass over every chunk.
Stage 2 (cross-encoder): slower, query-aware relevance scoring that reorders
the stage-1 candidates and drops anything the reranker considers irrelevant.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

INDEX_DIR = Path(__file__).parent / "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INITIAL_TOP_K = 10  # stage 1: cast a wide net
FINAL_TOP_K = 4  # stage 2: keep only the best-ranked chunks
MIN_RERANK_SCORE = -2.0  # below this, the cross-encoder considers it noise


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class Retriever:
    def __init__(self) -> None:
        if not INDEX_DIR.exists():
            raise RuntimeError("No FAISS index found. Run `python ingest.py` first.")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.store = FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        self.reranker = CrossEncoder(RERANKER_MODEL)

    def retrieve(self, question: str, final_top_k: int = FINAL_TOP_K) -> list[RetrievedChunk]:
        candidates = self.store.similarity_search(question, k=INITIAL_TOP_K)
        if not candidates:
            return []

        pairs = [(question, doc.page_content) for doc in candidates]
        rerank_scores = self.reranker.predict(pairs)

        ranked = sorted(zip(candidates, rerank_scores), key=lambda pair: pair[1], reverse=True)
        return [
            RetrievedChunk(
                text=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                score=float(score),
            )
            for doc, score in ranked
            if score >= MIN_RERANK_SCORE
        ][:final_top_k]
