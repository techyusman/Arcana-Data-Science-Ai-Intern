"""Simple desktop interface for the Week 8 RAG pipeline."""

from __future__ import annotations

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from dotenv import load_dotenv

from rag_pipeline import (
    ExtractiveAnswerGenerator,
    GroqAnswerGenerator,
    RAGPipeline,
    SQLiteVectorStore,
)


BASE_DIRECTORY = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIRECTORY / "data" / "rag.db"
SAMPLE_DOCUMENTS = BASE_DIRECTORY / "sample_documents"


class RAGApplication(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Week 8 — RAG Question Answering")
        self.geometry("900x680")
        self.minsize(720, 560)

        self.document_path = tk.StringVar(value=str(SAMPLE_DOCUMENTS))
        self.provider = tk.StringVar(value="auto")
        self.status = tk.StringVar(value="Ready")
        self._build_interface()

    def _build_interface(self) -> None:
        outer = ttk.Frame(self, padding=18)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(5, weight=1)

        ttk.Label(
            outer,
            text="RAG Document Assistant",
            font=("Segoe UI", 20, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            outer,
            text="Index TXT, Markdown, or PDF files and ask questions about their content.",
        ).grid(row=1, column=0, sticky="w", pady=(2, 16))

        document_frame = ttk.LabelFrame(outer, text="1. Knowledge base", padding=12)
        document_frame.grid(row=2, column=0, sticky="ew")
        document_frame.columnconfigure(0, weight=1)
        ttk.Entry(document_frame, textvariable=self.document_path).grid(
            row=0, column=0, sticky="ew", padx=(0, 8)
        )
        ttk.Button(document_frame, text="Browse folder", command=self._browse).grid(
            row=0, column=1, padx=(0, 8)
        )
        self.index_button = ttk.Button(
            document_frame, text="Build index", command=self._start_indexing
        )
        self.index_button.grid(row=0, column=2)

        question_frame = ttk.LabelFrame(outer, text="2. Ask a question", padding=12)
        question_frame.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        question_frame.columnconfigure(0, weight=1)
        self.question_entry = ttk.Entry(question_frame, font=("Segoe UI", 11))
        self.question_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.question_entry.insert(0, "What is the annual leave policy?")
        self.question_entry.bind("<Return>", lambda _event: self._start_question())

        ttk.Label(question_frame, text="Answer mode:").grid(row=0, column=1, padx=(0, 4))
        ttk.Combobox(
            question_frame,
            textvariable=self.provider,
            values=("auto", "local", "groq"),
            state="readonly",
            width=8,
        ).grid(row=0, column=2, padx=(0, 8))
        self.ask_button = ttk.Button(
            question_frame, text="Ask", command=self._start_question
        )
        self.ask_button.grid(row=0, column=3)

        ttk.Label(
            outer,
            text=(
                "Auto uses Groq when GROQ_API_KEY is configured; otherwise it returns "
                "the retrieved evidence locally."
            ),
            foreground="#555555",
        ).grid(row=4, column=0, sticky="w", pady=(7, 10))

        answer_frame = ttk.LabelFrame(outer, text="Answer and sources", padding=10)
        answer_frame.grid(row=5, column=0, sticky="nsew")
        answer_frame.columnconfigure(0, weight=1)
        answer_frame.rowconfigure(0, weight=1)
        self.output = ScrolledText(
            answer_frame,
            wrap="word",
            font=("Segoe UI", 11),
            padx=10,
            pady=10,
            state="disabled",
        )
        self.output.grid(row=0, column=0, sticky="nsew")

        ttk.Separator(outer).grid(row=6, column=0, sticky="ew", pady=(12, 8))
        ttk.Label(outer, textvariable=self.status).grid(row=7, column=0, sticky="w")

    def _browse(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.document_path.get())
        if selected:
            self.document_path.set(selected)

    def _set_busy(self, busy: bool, status: str) -> None:
        state = "disabled" if busy else "normal"
        self.index_button.configure(state=state)
        self.ask_button.configure(state=state)
        self.status.set(status)

    def _show_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")

    def _run_background(self, operation, working_status: str) -> None:
        self._set_busy(True, working_status)

        def worker() -> None:
            try:
                message, final_status = operation()
            except Exception as exc:  # Display operational errors without crashing the UI.
                self.after(0, lambda: messagebox.showerror("RAG error", str(exc)))
                self.after(0, lambda: self._set_busy(False, "Operation failed"))
            else:
                self.after(0, lambda: self._show_output(message))
                self.after(0, lambda: self._set_busy(False, final_status))

        threading.Thread(target=worker, daemon=True).start()

    def _start_indexing(self) -> None:
        source = Path(self.document_path.get().strip())

        def index_documents() -> tuple[str, str]:
            with SQLiteVectorStore(DATABASE_PATH) as store:
                documents, chunks = RAGPipeline(store).ingest(source)
                total = store.count()
            message = (
                "Index built successfully.\n\n"
                f"Document units processed: {documents}\n"
                f"Chunks added or updated: {chunks}\n"
                f"Total chunks in database: {total}\n"
                f"Database: {DATABASE_PATH}"
            )
            return message, f"Ready — {total} chunks indexed"

        self._run_background(index_documents, "Reading, chunking, and embedding documents…")

    def _start_question(self) -> None:
        question = self.question_entry.get().strip()
        if not question:
            messagebox.showwarning("Question required", "Enter a question first.")
            return

        provider = self.provider.get()

        def answer_question() -> tuple[str, str]:
            if not DATABASE_PATH.exists():
                raise RuntimeError("Build the document index before asking a question.")

            has_key = bool(os.getenv("GROQ_API_KEY"))
            if provider == "groq" and not has_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not configured. Choose local mode or add the key to .env."
                )
            generator = (
                GroqAnswerGenerator()
                if provider == "groq" or (provider == "auto" and has_key)
                else ExtractiveAnswerGenerator()
            )
            with SQLiteVectorStore(DATABASE_PATH) as store:
                answer, results = RAGPipeline(
                    store, answer_generator=generator
                ).ask(question, top_k=5)

            source_lines = []
            for result in results:
                page = f", page {result.chunk.page}" if result.chunk.page else ""
                source_lines.append(
                    f"• {result.chunk.source}{page} — relevance {result.score:.3f}"
                )
            sources = "\n".join(source_lines) or "No relevant sources found."
            return f"{answer}\n\nSources\n{sources}", f"Retrieved {len(results)} chunk(s)"

        self._run_background(answer_question, "Embedding question and retrieving context…")


def main() -> None:
    load_dotenv(BASE_DIRECTORY / ".env")
    application = RAGApplication()
    application.mainloop()


if __name__ == "__main__":
    main()

