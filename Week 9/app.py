"""Topic 4 (Streamlit front-end): chat UI over the RAG chatbot, with retrieved
sources shown alongside every answer so responses stay checkable, not just
confident-sounding.
"""
import streamlit as st

from rag_chain import RAGChatbot

st.set_page_config(page_title="Arcana Roadmap RAG Chatbot", page_icon="📘")
st.title("📘 Arcana Roadmap RAG Chatbot")
st.caption(
    "Ask questions about the Data Science & AI internship roadmap. "
    "Answers are grounded in the indexed documents only — swap data/roadmap_kb "
    "for your own knowledge base to reuse this for another domain."
)


@st.cache_resource
def load_chatbot() -> RAGChatbot:
    return RAGChatbot()


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"]):
                    st.write(f"{i + 1}. {source}")

query = st.chat_input("Ask a question about the roadmap...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            try:
                chatbot = load_chatbot()
                answer, chunks = chatbot.ask(query)
                sources = [f"{c.source} (relevance={c.score:.2f})" for c in chunks]
            except Exception as exc:
                answer = f"Error: {exc}\n\nDid you run `python ingest.py` and set GROQ_API_KEY in .env?"
                sources = []
        st.write(answer)
        if sources:
            with st.expander("Sources"):
                for i, source in enumerate(sources):
                    st.write(f"{i + 1}. {source}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
