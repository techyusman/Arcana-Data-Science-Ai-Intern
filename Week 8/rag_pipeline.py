"""A small, dependency-light Retrieval-Augmented Generation pipeline.

The module implements the complete flow:
documents -> chunks -> embeddings -> vector database -> query embedding ->
retrieval -> top-k context -> LLM -> cited answer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['_-][A-Za-z0-9]+)*")


@dataclass(frozen=True)
class Document:
    text: str
    source: str
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int | None
    chunk_index: int


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def _read_pdf(path: Path) -> list[Document]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "PDF support requires pypdfium2. Install requirements.txt or use TXT/MD files."
        ) from exc

    pdf = pdfium.PdfDocument(str(path))
    pages: list[Document] = []
    try:
        for page_number in range(len(pdf)):
            page = pdf[page_number]
            text_page = page.get_textpage()
            try:
                text = text_page.get_text_range().strip()
            finally:
                text_page.close()
                page.close()
            if text:
                pages.append(Document(text=text, source=path.name, page=page_number + 1))
    finally:
        pdf.close()
    return pages


def load_documents(path: str | Path) -> list[Document]:
    """Load a TXT/MD/PDF file or all supported files in a directory."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Document path does not exist: {input_path}")

    files = (
        [input_path]
        if input_path.is_file()
        else sorted(
            file
            for file in input_path.rglob("*")
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    )

    documents: list[Document] = []
    for file in files:
        suffix = file.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        if suffix == ".pdf":
            documents.extend(_read_pdf(file))
        else:
            text = file.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                documents.append(Document(text=text, source=file.name))
    return documents


def chunk_documents(
    documents: Iterable[Document], chunk_size: int = 120, overlap: int = 25
) -> list[Chunk]:
    """Split documents into overlapping word-based chunks."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for document in documents:
        words = document.text.split()
        for index, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start : start + chunk_size]
            if not chunk_words:
                break
            text = " ".join(chunk_words)
            identity = f"{document.source}|{document.page}|{index}|{text}"
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    source=document.source,
                    page=document.page,
                    chunk_index=index,
                )
            )
            if start + chunk_size >= len(words):
                break
    return chunks


class HashEmbeddingModel:
    """Deterministic local feature-hashing embeddings with no model download.

    It is intentionally lightweight for demonstration and offline execution.
    Production systems can replace this class with a neural embedding provider
    while retaining the rest of the pipeline.
    """

    def __init__(self, dimensions: int = 384):
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = [token.lower() for token in WORD_PATTERN.findall(text)]
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class SQLiteVectorStore:
    """Persistent vector store using SQLite and exact cosine search."""

    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                page INTEGER,
                chunk_index INTEGER NOT NULL,
                embedding TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteVectorStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def upsert(self, chunk: Chunk, embedding: Sequence[float]) -> None:
        self.connection.execute(
            """
            INSERT INTO chunks (id, text, source, page, chunk_index, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              text=excluded.text, source=excluded.source, page=excluded.page,
              chunk_index=excluded.chunk_index, embedding=excluded.embedding
            """,
            (
                chunk.id,
                chunk.text,
                chunk.source,
                chunk.page,
                chunk.chunk_index,
                json.dumps(list(embedding)),
            ),
        )

    def commit(self) -> None:
        self.connection.commit()

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def search(
        self, query_embedding: Sequence[float], top_k: int = 5, min_score: float = 0.0
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        results: list[SearchResult] = []
        rows = self.connection.execute(
            "SELECT id, text, source, page, chunk_index, embedding FROM chunks"
        )
        for chunk_id, text, source, page, chunk_index, raw_embedding in rows:
            embedding = json.loads(raw_embedding)
            score = sum(a * b for a, b in zip(query_embedding, embedding))
            if score >= min_score:
                results.append(
                    SearchResult(
                        chunk=Chunk(chunk_id, text, source, page, chunk_index),
                        score=score,
                    )
                )
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]


class AnswerGenerator(Protocol):
    def generate(self, question: str, context: str) -> str: ...


class GroqAnswerGenerator:
    """Generate grounded answers using the Groq chat completion API."""

    def __init__(self, model: str = "openai/gpt-oss-20b"):
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("Install the groq package to use Groq generation.") from exc
        self.client = Groq()
        self.model = model

    def generate(self, question: str, context: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer using only the supplied context. Do not invent facts. "
                        "If the answer is absent, say: 'I could not find that in the "
                        "indexed documents.' Preserve the [Source ...] citations."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0.1,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()


class ExtractiveAnswerGenerator:
    """Offline fallback that returns the most relevant evidence verbatim."""

    def generate(self, question: str, context: str) -> str:
        del question
        if not context.strip():
            return "I could not find that in the indexed documents."
        return "Relevant information from the indexed documents:\n\n" + context


class RAGPipeline:
    def __init__(
        self,
        vector_store: SQLiteVectorStore,
        embedding_model: HashEmbeddingModel | None = None,
        answer_generator: AnswerGenerator | None = None,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model or HashEmbeddingModel()
        self.answer_generator = answer_generator or ExtractiveAnswerGenerator()

    def ingest(
        self, path: str | Path, chunk_size: int = 120, overlap: int = 25
    ) -> tuple[int, int]:
        documents = load_documents(path)
        chunks = chunk_documents(documents, chunk_size, overlap)
        for chunk in chunks:
            self.vector_store.upsert(chunk, self.embedding_model.embed(chunk.text))
        self.vector_store.commit()
        return len(documents), len(chunks)

    def ask(
        self, question: str, top_k: int = 5, min_score: float = 0.05
    ) -> tuple[str, list[SearchResult]]:
        if not question.strip():
            raise ValueError("question cannot be empty")
        query_embedding = self.embedding_model.embed(question)
        results = self.vector_store.search(query_embedding, top_k, min_score)
        context = "\n\n".join(self._format_result(result) for result in results)
        answer = self.answer_generator.generate(question, context)
        return answer, results

    @staticmethod
    def _format_result(result: SearchResult) -> str:
        page = f", page {result.chunk.page}" if result.chunk.page is not None else ""
        return (
            f"[Source: {result.chunk.source}{page}; relevance={result.score:.3f}]\n"
            f"{result.chunk.text}"
        )

