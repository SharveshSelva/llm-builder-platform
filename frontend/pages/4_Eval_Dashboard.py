import streamlit as st
import httpx
import pandas as pd
import plotly.express as px
import os

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.header("Eval Dashboard")

tab1, tab2, tab3 = st.tabs(["RAGAS Evaluator", "DeepEval", "Cost & Latency Metrics"])

with tab1:
    st.caption("Evaluate any QA pair using RAGAS metrics (faithfulness, relevancy, context precision).")
    query = st.text_input("Question", key="ragas_q")
    answer = st.text_area("Answer", height=100, key="ragas_a")
    contexts = st.text_area("Contexts (one per line)", height=120, key="ragas_c")
    ground_truth = st.text_input("Ground truth (optional — enables Context Precision)", key="ragas_gt")

    if st.button("Evaluate with RAGAS", type="primary"):
        ctx_list = [c.strip() for c in contexts.strip().split("\n") if c.strip()]
        with st.spinner("Running RAGAS (LLM-as-judge via Groq)..."):
            resp = httpx.post(
                f"{BACKEND}/eval/run",
                json={
                    "query": query,
                    "answer": answer,
                    "contexts": ctx_list,
                    "ground_truth": ground_truth or None,
                },
                timeout=120,
            )
        if resp.status_code == 200:
            data = resp.json()
            col1, col2, col3 = st.columns(3)
            col1.metric("Faithfulness", f"{data['faithfulness']:.2%}")
            col2.metric("Answer Relevancy", f"{data['answer_relevancy']:.2%}")
            if data.get("context_precision") is not None:
                col3.metric("Context Precision", f"{data['context_precision']:.2%}")
        else:
            st.error(resp.text)

with tab2:
    st.caption(
        "DeepEval-style evaluation: **Faithfulness**, **Answer Relevancy**, and **Hallucination**. "
        "All three are reference-free — no ground truth needed. Scored by Groq LLM judge."
    )
    de_query = st.text_input("Question", key="de_q")
    de_answer = st.text_area("Answer", height=100, key="de_a")
    de_contexts = st.text_area("Contexts (one per line)", height=120, key="de_c")

    if st.button("Evaluate with DeepEval", type="primary"):
        ctx_list = [c.strip() for c in de_contexts.strip().split("\n") if c.strip()]
        if not ctx_list:
            st.warning("Add at least one context line.")
        else:
            with st.spinner("Running DeepEval (LLM-as-judge via Groq)..."):
                resp = httpx.post(
                    f"{BACKEND}/eval/deepeval",
                    json={"query": de_query, "answer": de_answer, "contexts": ctx_list},
                    timeout=120,
                )
            if resp.status_code == 200:
                data = resp.json()
                col1, col2, col3 = st.columns(3)
                col1.metric("Faithfulness", f"{data['faithfulness']:.2%}")
                col2.metric("Answer Relevancy", f"{data['answer_relevancy']:.2%}")
                col3.metric(
                    "Hallucination",
                    f"{data['hallucination']:.2%}",
                    delta=f"{-data['hallucination']:.2%}",
                    delta_color="inverse",
                )
                st.divider()
                st.caption(f"Faithfulness: {data['faithfulness_reason']}")
                st.caption(f"Relevancy: {data['relevancy_reason']}")
                st.caption(f"Hallucination: {data['hallucination_reason']}")
                st.caption(f"Latency: {data['latency_ms']:.0f}ms")
            else:
                st.error(resp.text)

with tab3:
    st.caption("Live metrics from the last 1000 LLM calls.")
    if st.button("Refresh"):
        resp = httpx.get(f"{BACKEND}/eval/logs", params={"limit": 500}, timeout=10)
        if resp.status_code == 200:
            logs = resp.json()["logs"]
            if not logs:
                st.info("No logs yet. Run some prompts first.")
            else:
                df = pd.DataFrame(logs)
                col1, col2, col3 = st.columns(3)
                col1.metric("Total calls", len(df))
                col2.metric("Total cost (USD)", f"${df['cost_usd'].sum():.4f}")
                col3.metric("Avg latency", f"{df['latency_ms'].mean():.0f}ms")

                fig = px.histogram(df, x="latency_ms", nbins=30, title="Latency distribution (ms)")
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.bar(
                    df.groupby("model")["cost_usd"].sum().reset_index(),
                    x="model", y="cost_usd",
                    title="Cost by model (USD)",
                )
                st.plotly_chart(fig2, use_container_width=True)

                fig3 = px.bar(
                    df.groupby("route")["latency_ms"].mean().reset_index(),
                    x="route", y="latency_ms",
                    title="Avg latency by route (ms)",
                )
                st.plotly_chart(fig3, use_container_width=True)
