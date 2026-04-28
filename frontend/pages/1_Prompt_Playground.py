import streamlit as st
import httpx
import json
import os

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.header("Prompt Playground")
st.caption("Run prompts with structured output validation, live streaming, or side-by-side comparison.")

tab1, tab2, tab3 = st.tabs(["Single Prompt", "A/B Compare", "Live Stream"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        prompt = st.text_area("Prompt", height=150, placeholder="Enter your prompt...")
        system = st.text_area("System prompt (optional)", height=80)
    with col2:
        mode = st.radio("Mode", ["fast", "smart"], index=0)
        st.caption("fast = Llama 8B, smart = Llama 70B")
        schema_raw = st.text_area(
            "Output JSON schema (optional)",
            height=120,
            placeholder='{"name": "str", "score": 0}',
        )

    if st.button("Run", type="primary"):
        schema = {}
        if schema_raw.strip():
            try:
                schema = json.loads(schema_raw)
            except Exception:
                st.error("Invalid JSON schema")
                st.stop()
        with st.spinner("Running..."):
            resp = httpx.post(
                f"{BACKEND}/prompt/run",
                json={"prompt": prompt, "system_prompt": system, "mode": mode, "output_schema": schema},
                timeout=60,
            )
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Model: {data['model_used']} | Tokens: {data['tokens_used']} | Latency: {data['latency_ms']}ms")
            st.json(data["data"])
        else:
            st.error(f"Error {resp.status_code}: {resp.text}")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        prompt_a = st.text_area("Prompt A", height=180, key="pa")
    with col2:
        prompt_b = st.text_area("Prompt B", height=180, key="pb")

    system_ab = st.text_input("System prompt (shared)", key="sys_ab")
    if st.button("Compare", type="primary"):
        with st.spinner("Running both prompts in parallel..."):
            resp = httpx.post(
                f"{BACKEND}/prompt/compare",
                json={"prompt_a": prompt_a, "prompt_b": prompt_b, "system_prompt": system_ab, "output_schema": {}},
                timeout=60,
            )
        if resp.status_code == 200:
            result = resp.json()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Prompt A")
                ra = result["result_a"]
                st.caption(f"Model: {ra['model_used']} | Tokens: {ra['tokens_used']} | {ra['latency_ms']}ms")
                st.json(ra["data"])
            with col2:
                st.subheader("Prompt B")
                rb = result["result_b"]
                st.caption(f"Model: {rb['model_used']} | Tokens: {rb['tokens_used']} | {rb['latency_ms']}ms")
                st.json(rb["data"])
        else:
            st.error(resp.text)

with tab3:
    st.caption("Tokens stream in real time via Server-Sent Events. Best for long responses.")
    stream_prompt = st.text_area("Prompt", height=150, placeholder="Write a detailed essay on...", key="stream_p")
    stream_system = st.text_area("System prompt (optional)", height=60, key="stream_s")
    stream_mode = st.radio("Mode", ["fast", "smart"], key="stream_mode")

    if st.button("Stream", type="primary"):
        if not stream_prompt.strip():
            st.warning("Enter a prompt first.")
        else:
            output_box = st.empty()
            full_text = ""
            with httpx.stream(
                "POST",
                f"{BACKEND}/prompt/stream",
                json={"prompt": stream_prompt, "system_prompt": stream_system, "mode": stream_mode},
                timeout=120,
            ) as resp:
                if resp.status_code != 200:
                    st.error(f"Error {resp.status_code}")
                else:
                    for line in resp.iter_lines():
                        if line.startswith("data: "):
                            token = line[6:]
                            if token == "[DONE]":
                                break
                            full_text += token
                            output_box.markdown(full_text)
