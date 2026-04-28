# JD Requirements → Project Mapping

---

## Areas of Responsibility

### AI-Native Engineering

| JD Requirement                                                  | Project Implementation                                                                 |
|-----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Build UI components, REST APIs, end-to-end workflows            | Streamlit 4-page frontend + 13 FastAPI endpoints + full RAG/agent/eval workflows       |
| Design prompt templates, structured outputs, tool-calling       | System prompts in `llm.py`, JSON mode outputs in prompt router, Tavily tool-calling   |
| Engineer RAG pipelines, agentic systems, tool orchestration     | `rag_service.py`, LangGraph research agent, CrewAI pipeline, AutoGen GroupChat        |
| Integrate fine-tuned or hosted AI models                        | Groq-hosted Llama 3.1 + Gemma2, swappable via `mode=fast/smart` toggle               |

---

### Solution Design & Architecture

| JD Requirement                                                  | Project Implementation                                                                 |
|-----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Rapidly absorb unfamiliar codebases                             | Built across LangChain, LangGraph, ChromaDB, Presidio, RAGAS — all new libraries     |
| Translate requirements into AI designs with API/data-flow specs | FastAPI routers + Pydantic schemas define exact API contracts (`schemas.py`)           |
| Model selection, RAG vs agentic, cost-accuracy trade-offs       | Groq over OpenAI (cost), ChromaDB over Pinecone (no API), Cloud Run over EC2          |
| Contribute to architecture discussions; document trade-offs     | ChromaDB ephemeral trade-off, fallback chain design, embedding cache decision          |

---

### Quality & Governance

| JD Requirement                                                  | Project Implementation                                                                 |
|-----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Validate AI outputs; flag hallucinations and bias early         | `guardrails.py` — injection, bias, toxicity; `deepeval_service.py` — hallucination   |
| Design and run continuous test automation for AI features       | `tests/unit/` + `tests/integration/` with mocked Groq; runs on every GitHub push     |
| Implement monitoring, logging, and AI guardrails                | `request_logger.py` middleware + Redis LLM/eval logs + Eval Dashboard charts          |
| Secure-by-design and responsible AI practices                   | Presidio PII redaction, GCP Secret Manager, TLS Redis, guardrails before LLM call    |

---

### Delivery & Client Impact

| JD Requirement                                                  | Project Implementation                                                                 |
|-----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Ship working AI-native features at accelerated pace             | Full platform built and deployed end-to-end                                            |
| Deliver short-cycle POCs, prototypes, and live client demos     | Live at https://ai-platform-frontend-vtvta23tia-uc.a.run.app                         |
| Produce handover docs so client teams can scale independently   | GitHub repo + CI/CD — any engineer can `git push` to redeploy                        |

---

### Communication & Collaboration

| JD Requirement                                                  | Project Implementation                                                                 |
|-----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Communicate to both technical and non-technical stakeholders    | Streamlit UI abstracts all complexity — non-technical users upload PDFs and chat      |
| Lead demos, technical walkthroughs, client-facing showcases     | Live demo URL + GitHub repo ready to walk through in interview                        |

---

## Skills & Technologies

| JD Item                                  | Status   | Project Implementation                                              |
|------------------------------------------|----------|---------------------------------------------------------------------|
| Python                                   | Done     | Entire codebase                                                     |
| Java / .NET / JS / React / Angular       | N/A      | Different language stack                                            |
| REST APIs                                | Done     | 13 endpoints across 4 FastAPI routers                               |
| SQL                                      | N/A      | Not used — Redis + ChromaDB instead                                 |
| LLMs & Embeddings                        | Done     | Groq (Llama 3.1, Gemma2) + sentence-transformers local embeddings  |
| RAG pipelines                            | Done     | PDF → chunk → embed → store → retrieve → answer                    |
| Prompt engineering & structured outputs  | Done     | JSON mode, system prompts, citation prompts                         |
| LangChain / LangGraph / CrewAI / AutoGen | Done     | All 4 frameworks implemented                                        |
| AI coding assistants (Claude)            | Done     | Built entirely with Claude Code                                     |
| GCP (Cloud & DevOps)                     | Done     | Cloud Run, Artifact Registry, Secret Manager, GCS bucket           |
| Git                                      | Done     | GitHub repo with full commit history                                |
| CI/CD pipelines                          | Done     | GitHub Actions: lint → test → build Docker → deploy to Cloud Run   |
| Containers                               | Done     | Dockerfiles for backend + frontend                                  |
| AI quality metrics                       | Done     | RAGAS (faithfulness, relevancy, precision) + DeepEval (hallucination)|
| Monitoring & guardrails                  | Done     | Request logger middleware + guardrails service                      |
| Cost & latency optimisation              | Done     | Embedding cache, model tier toggle, parallel A/B compare            |
| Fallback & degradation strategies        | Done     | 2-model fallback chain, Redis-optional graceful degradation         |
| Vector DB concepts                       | Done     | ChromaDB — cosine similarity, collection namespacing, persistence   |
| Streamlit / FastAPI                      | Done     | Both fully implemented                                              |
| RAGAS / DeepEval                         | Done     | Both implemented (DeepEval built from scratch)                      |
| Event-driven architectures               | Done     | Redis LPUSH/LTRIM event log + asyncio.create_task background eval   |
| Kubernetes                               | Not done | Requires external cluster — documented as next step                 |

---

## Summary

| Category        | Count |
|-----------------|-------|
| Fully covered   | ~90%  |
| Not applicable  | Java / .NET / JS / React / SQL (different stack) |
| Only gap        | Kubernetes (external cluster required) |

---

## Links

| Resource    | URL |
|-------------|-----|
| Live Demo   | https://ai-platform-frontend-vtvta23tia-uc.a.run.app |
| GitHub Repo | https://github.com/SharveshSelva/llm-builder-platform |
