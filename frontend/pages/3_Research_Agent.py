import streamlit as st
import httpx
import os

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.header("Research Agent + CrewAI + AutoGen")

tab1, tab2, tab3 = st.tabs(["LangGraph Research Agent", "CrewAI Blog Writer", "AutoGen Q&A"])

with tab1:
    st.caption("Searches the web, reads results, reflects on draft, returns cited summary.")
    question = st.text_input("Research question", placeholder="What are the latest advances in RAG systems?")
    mode = st.radio("Mode", ["fast", "smart"], key="research_mode")
    if st.button("Research", type="primary"):
        with st.spinner("Searching → drafting → reflecting → refining..."):
            resp = httpx.post(
                f"{BACKEND}/agents/research",
                json={"question": question, "mode": mode},
                timeout=120,
            )
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Latency: {data['latency_ms']}ms | Reflection used: {data['reflection_used']}")
            st.markdown(data["summary"])
            if data["sources"]:
                st.subheader("Sources")
                for s in data["sources"]:
                    st.markdown(f"- {s}")
        else:
            st.error(resp.text)

with tab2:
    st.caption("Three agents: Researcher → Writer → Editor. Input a topic, get a blog post.")
    topic = st.text_input("Blog topic", placeholder="The future of agentic AI systems")
    mode2 = st.radio("Mode", ["fast", "smart"], key="crew_mode")
    if st.button("Generate Blog Post", type="primary"):
        with st.spinner("Running Researcher → Writer → Editor..."):
            resp = httpx.post(
                f"{BACKEND}/agents/crew",
                json={"topic": topic, "mode": mode2},
                timeout=180,
            )
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Latency: {data['latency_ms']}ms")
            with st.expander("Researcher Notes"):
                st.markdown(data["researcher_notes"])
            st.subheader("Final Blog Post")
            st.markdown(data["blog_post"])
        else:
            st.error(resp.text)

with tab3:
    st.caption(
        "AutoGen GroupChat pattern: **Assistant** answers → **Critic** reviews → "
        "**Refiner** improves (if needed). 2–3 agent rounds, self-correcting output."
    )
    ag_question = st.text_input(
        "Question",
        placeholder="Explain the trade-offs between RAG and fine-tuning",
        key="ag_q",
    )
    ag_mode = st.radio("Mode", ["fast", "smart"], key="ag_mode")

    if st.button("Run AutoGen", type="primary"):
        with st.spinner("Assistant → Critic → Refiner..."):
            resp = httpx.post(
                f"{BACKEND}/agents/autogen",
                json={"question": ag_question, "mode": ag_mode},
                timeout=120,
            )
        if resp.status_code == 200:
            data = resp.json()
            revised_label = "Revised after critique" if data["revised"] else "Approved on first pass"
            st.success(f"Latency: {data['latency_ms']}ms | Rounds: {data['rounds']} | {revised_label}")

            with st.expander("Round 1 — Assistant's initial answer"):
                st.markdown(data["initial_answer"])

            with st.expander("Round 2 — Critic's review"):
                st.markdown(data["critique"])

            st.subheader("Final Answer")
            st.markdown(data["final_answer"])
        else:
            st.error(resp.text)
