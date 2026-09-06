"""Command-line interface for the Week 8 RAG pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from rag_pipeline import (
    ExtractiveAnswerGenerator,
    GroqAnswerGenerator,
    RAGPipeline,
    SQLiteVectorStore,
)


BASE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_DATABASE = BASE_DIRECTORY / "data" / "rag.db"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local document RAG pipeline")
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Index a file or directory")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--chunk-size", type=int, default=120)
    ingest.add_argument("--overlap", type=int, default=25)

    ask = subparsers.add_parser("ask", help="Ask a question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--min-score", type=float, default=0.05)
    ask.add_argument(
        "--provider",
        choices=("auto", "groq", "local"),
        default="auto",
        help="auto uses Groq when GROQ_API_KEY exists, otherwise local evidence output",
    )
    ask.add_argument("--model", default="openai/gpt-oss-20b")

    subparsers.add_parser("stats", help="Show the number of indexed chunks")
    return parser


def choose_generator(provider: str, model: str):
    has_key = bool(os.getenv("GROQ_API_KEY"))
    if provider == "groq" and not has_key:
        raise RuntimeError("GROQ_API_KEY is required when --provider groq is selected.")
    if provider == "groq" or (provider == "auto" and has_key):
        return GroqAnswerGenerator(model)
    return ExtractiveAnswerGenerator()


def main() -> None:
    load_dotenv(BASE_DIRECTORY / ".env")
    args = build_parser().parse_args()
    with SQLiteVectorStore(args.db) as store:
        if args.command == "ingest":
            pipeline = RAGPipeline(store)
            document_count, chunk_count = pipeline.ingest(
                args.path, args.chunk_size, args.overlap
            )
            print(f"Indexed {document_count} document unit(s) as {chunk_count} chunk(s).")
            print(f"Vector database: {args.db.resolve()}")
        elif args.command == "ask":
            generator = choose_generator(args.provider, args.model)
            pipeline = RAGPipeline(store, answer_generator=generator)
            answer, results = pipeline.ask(args.question, args.top_k, args.min_score)
            print(f"Retrieved {len(results)} relevant chunk(s).\n")
            print(answer)
        elif args.command == "stats":
            print(f"Indexed chunks: {store.count()}")
            print(f"Vector database: {args.db.resolve()}")


if __name__ == "__main__":
    main()

