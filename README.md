# Production AI Platform

A production-grade AI platform demonstrating the full LLM engineering stack — RAG pipelines, multi-agent orchestration, evaluation frameworks, guardrails, and CI/CD deployment.

**Live Demo:** https://ai-platform-frontend-vtvta23tia-uc.a.run.app
**GitHub:** https://github.com/SharveshSelva/llm-builder-platform

---

## Features

| Feature                        | Description                                                              |
|-------------------------------|--------------------------------------------------------------------------|
| Prompt Playground             | A/B testing, structured JSON outputs, live SSE streaming                 |
| RAG Chatbot                   | PDF ingestion, ChromaDB vector search, citations by source + page        |
| Research Agent (LangGraph)    | StateGraph: search → draft → reflect → refine                           |
| CrewAI Pipeline               | 3-agent sequential flow: Researcher → Writer → Editor                   |
| AutoGen Q&A                   | 3-agent GroupChat: Assistant → Critic → Refiner with APPROVE/REVISE loop|
| RAGAS Evaluation              | Faithfulness, Answer Relevancy, Context Precision (LLM-as-judge)        |
| DeepEval                      | Faithfulness, Relevancy, Hallucination scoring (reference-free)         |
| Guardrails                    | Prompt injection detection, PII redaction (Presidio), bias + toxicity   |
| Monitoring                    | Per-request latency, token count, cost logging via Redis                 |
| Continuous Eval               | Auto-scores every RAG response in the background, surfaced in dashboard  |

---

## Models

| Mode     | Model                    | Use Case                  |
|----------|--------------------------|---------------------------|
| fast     | llama-3.1-8b-instant     | Default, low latency       |
| smart    | llama-3.1-70b-versatile  | Complex reasoning          |
| fallback | gemma2-9b-it             | Timeout / error fallback   |

All models served via **Groq** (free tier).

---

## Architecture

```
User
 └── Streamlit Frontend (Cloud Run :8501)
          └── FastAPI Backend (Cloud Run :8000)
                   ├── ChromaDB (vector store, local /tmp)
                   ├── Groq API (LLM calls)
                   ├── Upstash Redis (embedding cache + logs)
                   └── GCP Secret Manager (API keys)
```

```
backend/
├── routers/
│   ├── prompt.py      — /prompt/run, /compare, /stream
│   ├── rag.py         — /rag/ingest, /chat
│   ├── agents.py      — /agents/research, /crew, /autogen
│   └── eval.py        — /eval/run, /deepeval, /logs, /rag-eval-logs
├── services/
│   ├── llm.py         — fallback chain (fast → fallback on error)
│   ├── embeddings.py  — sentence-transformers + Redis cache
│   ├── vector_store.py— ChromaDB add/query
│   ├── rag_service.py — PDF chunking + RAG chat + background eval
│   ├── guardrails.py  — injection, bias, toxicity, PII redaction
│   ├── evaluation.py  — RAGAS scoring
│   ├── deepeval_service.py — DeepEval LLM-as-judge metrics
│   ├── research_agent.py   — LangGraph StateGraph agent
│   ├── crew_service.py     — CrewAI-style 3-agent pipeline
│   └── autogen_service.py  — AutoGen GroupChat pattern
└── middleware/
    └── request_logger.py   — latency, tokens, cost per request
```

---

## Local Development

### 1. Clone and configure

```bash
git clone https://github.com/SharveshSelva/llm-builder-platform.git
cd llm-builder-platform
cp .env.example .env
# Edit .env — add GROQ_API_KEY and TAVILY_API_KEY
```

### 2. Install dependencies

```bash
pip install -r requirements.txt --prefer-binary
python -m spacy download en_core_web_lg
```

### 3. Run

```bash
# Terminal 1 — backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend
streamlit run frontend/app.py --server.port 8501
```

| Service  | URL                        |
|----------|----------------------------|
| Frontend | http://localhost:8501      |
| Backend  | http://localhost:8000/docs |

---

## Tests

```bash
pytest tests/ -v
```

- `tests/unit/` — guardrails, LLM service, RAG chunking (mocked Groq)
- `tests/integration/` — API endpoints via FastAPI TestClient

---

## CI/CD

Every push to `main`:
1. Ruff lint
2. pytest unit + integration tests
3. Docker build → push to GCP Artifact Registry
4. Deploy to GCP Cloud Run (backend + frontend)

---

## Smoke Tests

```bash
# Health check
curl https://ai-platform-backend-vtvta23tia-uc.a.run.app/health

# Run a prompt
curl -X POST https://ai-platform-backend-vtvta23tia-uc.a.run.app/prompt/run \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is RAG?","mode":"fast"}'

# RAG chat
curl -X POST https://ai-platform-backend-vtvta23tia-uc.a.run.app/rag/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Summarise the document","collection_name":"documents","top_k":5,"mode":"fast"}'

# View eval logs
curl https://ai-platform-backend-vtvta23tia-uc.a.run.app/eval/logs?limit=10
```
