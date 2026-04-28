import streamlit as st

st.set_page_config(
    page_title="AI Platform",
    page_icon="🤖",
    layout="wide",
)
st.title("🤖 Production AI Platform")
st.markdown("""
Navigate using the sidebar:
- **Prompt Playground** — A/B test prompts with structured JSON output
- **RAG Chat** — Upload PDFs and chat with your documents
- **Research Agent** — AI-powered research with web search and reflection
- **Eval Dashboard** — Metrics, cost tracking, and RAGAS scores
""")
