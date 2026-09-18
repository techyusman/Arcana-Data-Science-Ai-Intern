# Week 9: Working RAG Chatbot with Custom Knowledge Base

This project upgrades the Week 8 hand-rolled RAG pipeline into an end-to-end
application, covering every Week 9 topic:

| File | Topic |
|---|---|
| `ingest.py` | LangChain document loaders, splitter, embeddings, FAISS |
| `retriever.py` | Retrieval tuning & reranking (bi-encoder + cross-encoder) |
| `rag_chain.py` | Hallucination & grounding (strict context-only prompting, citations, refusal) |
| `app.py` | Streamlit front-end |
| `evaluate.py` | Evaluation of RAG outputs (retrieval hit rate + LLM-as-judge) |
| `data/roadmap_kb/` | The custom knowledge base (the Arcana Info roadmap itself, split into Markdown files) |

```text
Documents (.md) -> RecursiveCharacterTextSplitter -> HuggingFace embeddings -> FAISS
User question -> FAISS top-10 -> cross-encoder rerank -> top-4 -> Groq LLM (grounded prompt) -> cited answer
```

## Setup

```powershell
cd "Week 9"
python -m pip install -r requirements.txt
copy .env.example .env
# edit .env and paste a free key from https://console.groq.com/keys
```

## Run

Build the vector index (re-run this whenever `data/roadmap_kb` changes):

```powershell
python ingest.py
```

Launch the chatbot:

```powershell
streamlit run app.py
```

Run the evaluation suite:

```powershell
python evaluate.py
```

Results are written to `eval/results.csv`; a summary (retrieval hit rate,
average faithfulness, average relevance) prints to the console.

## Using your own knowledge base

Replace the files in `data/roadmap_kb/` with your own `.md`/`.txt` documents
(class notes, a product FAQ, project docs — anything you have rights to use),
delete the `faiss_index/` folder, and re-run `python ingest.py`. Update
`eval/eval_questions.json` with questions whose answers you can verify against
your new documents, including at least one question with no answer in the
knowledge base (`"expected_source": null`) to confirm the chatbot refuses
instead of guessing.

## Design notes

- **Embeddings run locally** (`sentence-transformers/all-MiniLM-L6-v2`) so
  only the final answer generation call needs a network request — this keeps
  the project runnable on Groq's free tier without an embeddings bill.
- **Reranking** uses `cross-encoder/ms-marco-MiniLM-L-6-v2`: FAISS retrieves
  10 candidates cheaply, the cross-encoder rescoring keeps the 4 most
  relevant, per the precision-vs-speed trade-off described in the Week 9
  retrieval-tuning notes.
- **Grounding** is enforced at the prompt level (answer only from context,
  cite the source file, refuse verbatim when the context doesn't contain the
  answer) and checked at evaluation time by both a retrieval hit-rate metric
  and an LLM-as-judge faithfulness score.
