"""Topic 3 (Hallucination & grounding): combine the retriever with Groq, and
force the model to answer from retrieved context only, with citations, and an
explicit refusal when the context does not contain the answer.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from retriever import Retriever, RetrievedChunk

load_dotenv(Path(__file__).parent / ".env")

REFUSAL = "I don't know based on the provided documents."

GROUNDING_SYSTEM_PROMPT = f"""You are the Arcana Info internship roadmap assistant.
Answer the question using ONLY the context below. Do not use outside knowledge and
do not guess. If the answer is not in the context, reply exactly: "{REFUSAL}"
Cite the source file for every claim, like [Source: phase2_machine_learning.md]."""


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[Source: {c.source}]\n{c.text}" for c in chunks)


class RAGChatbot:
    def __init__(self, model: str = "openai/gpt-oss-120b") -> None:
        self.retriever = Retriever()
        self.llm = ChatGroq(model=model, temperature=0.1, api_key=os.environ.get("GROQ_API_KEY"))

    def ask(self, question: str) -> tuple[str, list[RetrievedChunk]]:
        chunks = self.retriever.retrieve(question)
        if not chunks:
            return REFUSAL, []

        context = format_context(chunks)
        response = self.llm.invoke(
            [
                {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ]
        )
        return response.content, chunks
