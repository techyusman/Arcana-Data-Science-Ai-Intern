# Week 8: Complete RAG Pipeline

This project implements the complete Retrieval-Augmented Generation flow:

```text
Documents -> Chunking -> Embeddings -> Vector Database
User Query -> Query Embedding -> Retrieval -> Top-K Chunks -> LLM -> Final Answer
```

It supports TXT, Markdown, and PDF documents, overlapping chunks, deterministic local embeddings, a persistent SQLite vector database, cosine-similarity retrieval, Top-K selection, source citations, and grounded generation through Groq. If no API key is configured, it remains fully runnable and returns the retrieved evidence locally.

## Files

- `rag_pipeline.py` — loaders, chunker, embedding model, vector store, retriever, and answer generators
- `app.py` — command-line application
- `sample_documents/` — knowledge base for the demonstration
- `tests/` — automated pipeline tests
- `RAG_Pipeline_Architecture.md` — architecture report and viva explanation

## Setup

From PowerShell:

```powershell
cd "Week 8"
python -m pip install -r requirements.txt
```

Groq generation is optional. To enable it, copy `.env.example` to `.env` and add a valid key, or set the environment variable for the current terminal:

```powershell
$env:GROQ_API_KEY = "your-key"
```

Never commit the `.env` file or an API key.

## Run the Complete Flow

### Simple desktop UI

Launch the graphical interface:

```powershell
python ui.py
```

Select a folder, click **Build index**, enter a question, and click **Ask**.

### Command line

Index the sample knowledge base:

```powershell
python app.py ingest sample_documents
```

Ask a question. `auto` uses Groq when a key is available and the local fallback otherwise:

```powershell
python app.py ask "How many annual leave days do employees receive?"
```

Force local mode:

```powershell
python app.py ask "What is the remote work policy?" --provider local --top-k 3
```

Force Groq generation:

```powershell
python app.py ask "When is a medical certificate required?" --provider groq
```

Inspect the index:

```powershell
python app.py stats
```

Index another document or folder:

```powershell
python app.py ingest "C:\path\to\documents"
```

## Run Tests

```powershell
python -m unittest discover -s tests -v
```

## Implementation Notes

The included feature-hashing embedder is deterministic, private, and requires no download. It demonstrates the complete embedding and vector-search architecture but is less semantically powerful than a production neural embedding model. The components are separated so a hosted or local neural embedder can replace it without changing document loading, storage, retrieval, or generation.

SQLite exact search is suitable for this learning deliverable and small document collections. A production system with millions of chunks should use an approximate-nearest-neighbor vector engine, add authentication and metadata permission filters, and evaluate retrieval quality on a representative test set.
