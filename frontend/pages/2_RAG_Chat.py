import streamlit as st
import httpx
import os

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.header("RAG Chat")
st.caption("Upload PDFs, then ask questions. Answers include citations.")

with st.sidebar:
    st.subheader("Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    collection = st.text_input("Collection name", value="documents")
    if uploaded_file and st.button("Ingest PDF"):
        with st.spinner("Ingesting..."):
            resp = httpx.post(
                f"{BACKEND}/rag/ingest",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                data={"collection_name": collection},
                timeout=120,
            )
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Ingested {data['chunks_added']} chunks from {data['filename']}")
        else:
            st.error(resp.text)
    mode = st.radio("Mode", ["fast", "smart"], key="rag_mode")
    top_k = st.slider("Top-K chunks", 1, 10, 5)

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

for msg in st.session_state.rag_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Citations"):
                for c in msg["citations"]:
                    st.markdown(f"**[{c['source']}]** (page {c.get('page', '?')}, score {c['score']})")
                    st.caption(c["text"])

if query := st.chat_input("Ask a question about your documents..."):
    st.session_state.rag_messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = httpx.post(
                f"{BACKEND}/rag/chat",
                json={"query": query, "collection_name": collection, "top_k": top_k, "mode": mode},
                timeout=60,
            )
        if resp.status_code == 200:
            data = resp.json()
            st.markdown(data["answer"])
            st.caption(f"Model: {data['model_used']} | Tokens: {data['tokens_used']} | {data['latency_ms']}ms")
            if data["citations"]:
                with st.expander("Citations"):
                    for c in data["citations"]:
                        st.markdown(f"**[{c['source']}]** (page {c.get('page', '?')}, score {c['score']})")
                        st.caption(c["text"])
            st.session_state.rag_messages.append({
                "role": "assistant",
                "content": data["answer"],
                "citations": data["citations"],
            })
        else:
            st.error(resp.text)
