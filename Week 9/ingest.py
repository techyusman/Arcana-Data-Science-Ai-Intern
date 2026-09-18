"""Topic 1 (LangChain): load the custom knowledge base, chunk it, embed it, and
persist a FAISS vector store. Run this once before app.py or evaluate.py, and
again after editing anything under data/roadmap_kb.
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).parent / "data" / "roadmap_kb"
INDEX_DIR = Path(__file__).parent / "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_index() -> None:
    loader = DirectoryLoader(
        str(DATA_DIR), glob="*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    if not documents:
        raise RuntimeError(f"No .md files found in {DATA_DIR}")

    # 500 chars / 80 overlap keeps each roadmap entry mostly intact in one chunk
    # while still splitting the longer phase files into a few retrievable pieces.
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(INDEX_DIR))
    print(f"Indexed {len(documents)} documents -> {len(chunks)} chunks -> {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
