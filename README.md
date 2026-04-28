# Production AI Platform

A single production-grade AI platform combining:

- **Prompt Engineering Playground** — A/B testing, Pydantic structured outputs
- **RAG Chatbot** — ChromaDB, PDF ingestion, citations
- **LangGraph Research Agent** — search → reflect → summarize
- **CrewAI Multi-Agent Pipeline** — Researcher → Writer → Editor
- **RAGAS Evaluation + Guardrails** — faithfulness scoring, PII redaction, injection detection
- **Model Fallback Chain** — cost-accuracy toggle (fast/smart)
- **Full Docker + async FastAPI backend**

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env — fill in ANTHROPIC_API_KEY (required), OPENAI_API_KEY, TAVILY_API_KEY
```

### 2. Local development

```bash
# Start infrastructure
docker-compose up redis chromadb -d

# Install dependencies
pip install -r requirements.txt

# Start backend
uvicorn backend.main:app --reload --port 8000

# Start frontend (new terminal)
streamlit run frontend/app.py --server.port 8501
```

### 3. Full Docker stack

```bash
docker-compose up --build
```

| Service   | URL                          |
|-----------|------------------------------|
| Frontend  | http://localhost:8501        |
| Backend   | http://localhost:8000/docs   |
| ChromaDB  | http://localhost:8001        |

## Smoke Tests

```bash
# Health check
curl http://localhost:8000/health

# Run a prompt
curl -X POST http://localhost:8000/prompt/run \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is 2+2?","mode":"fast"}'

# Compare two prompts
curl -X POST http://localhost:8000/prompt/compare \
  -H "Content-Type: application/json" \
  -d '{"prompt_a":"Explain RAG briefly","prompt_b":"Explain RAG in detail","system_prompt":""}'

# Ingest a PDF
curl -X POST http://localhost:8000/rag/ingest \
  -F "file=@your_file.pdf" \
  -F "collection_name=documents"

# RAG chat
curl -X POST http://localhost:8000/rag/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What is this document about?","collection_name":"documents","top_k":5,"mode":"fast"}'

# Research agent
curl -X POST http://localhost:8000/agents/research \
  -H "Content-Type: application/json" \
  -d '{"question":"What is LangGraph?","mode":"fast"}'

# View eval logs
curl http://localhost:8000/eval/logs?limit=10
```

## Architecture

```
FastAPI backend
├── /prompt  — LLM calls with fallback chain + injection detection
├── /rag     — PDF ingestion + retrieval-augmented chat
├── /agents  — LangGraph research agent + CrewAI blog pipeline
└── /eval    — RAGAS scoring + cost/latency metrics

Services
├── llm.py          — model fallback chain (Sonnet → Haiku on error)
├── embeddings.py   — sentence-transformers + Redis cache
├── vector_store.py — ChromaDB add/query
├── rag_service.py  — PDF chunking + RAG chat
├── guardrails.py   — injection detection + PII redaction (Presidio)
├── evaluation.py   — RAGAS faithfulness + answer relevancy
├── research_agent.py — LangGraph: search → draft → reflect → refine
└── crew_service.py — CrewAI: Researcher → Writer → Editor
```

## Models

| Mode  | Model               | Use case              |
|-------|---------------------|-----------------------|
| fast  | claude-sonnet-4     | Default, cost-effective |
| smart | claude-opus-4       | Complex reasoning     |
| fallback | claude-haiku-4   | Timeout/error fallback |
