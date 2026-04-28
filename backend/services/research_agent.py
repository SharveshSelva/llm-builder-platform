import time
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from backend.config import get_settings
from backend.models.schemas import ResearchResponse

settings = get_settings()


class AgentState(TypedDict):
    question: str
    search_results: list[dict]
    draft: str
    reflection: str
    final_summary: str
    sources: list[str]
    mode: str


def _get_llm(mode: str) -> ChatGroq:
    model = settings.smart_model if mode == "smart" else settings.primary_model
    return ChatGroq(
        model=model,
        groq_api_key=settings.groq_api_key,
        temperature=0,
    )


def _search_node(state: AgentState) -> dict:
    tool = TavilySearchResults(max_results=3, api_key=settings.tavily_api_key)
    results = tool.invoke(state["question"])
    return {"search_results": results}


def _draft_node(state: AgentState) -> dict:
    llm = _get_llm(state["mode"])
    context = "\n\n".join([
        f"Source: {r.get('url', '')}\n{r.get('content', '')}"
        for r in state["search_results"]
    ])
    prompt = (
        f"Based on the following search results, write a concise research summary "
        f"answering: {state['question']}\n\nSearch results:\n{context}"
    )
    draft = llm.invoke(prompt).content
    sources = [r.get("url", "") for r in state["search_results"] if r.get("url")]
    return {"draft": draft, "sources": sources}


def _reflect_node(state: AgentState) -> dict:
    llm = _get_llm(state["mode"])
    context = "\n\n".join([r.get("content", "") for r in state["search_results"]])
    prompt = (
        f"You wrote this draft:\n{state['draft']}\n\n"
        f"Compare it against the source material:\n{context}\n\n"
        f"Identify any inaccuracies, unsupported claims, or missing key points. "
        f"Be specific."
    )
    reflection = llm.invoke(prompt).content
    return {"reflection": reflection}


def _refine_node(state: AgentState) -> dict:
    llm = _get_llm(state["mode"])
    prompt = (
        f"Refine this draft based on the reflection critique.\n\n"
        f"Original draft:\n{state['draft']}\n\n"
        f"Critique:\n{state['reflection']}\n\n"
        f"Write the improved final summary."
    )
    final = llm.invoke(prompt).content
    return {"final_summary": final}


def build_research_graph():
    graph = StateGraph(AgentState)
    graph.add_node("search", _search_node)
    graph.add_node("draft", _draft_node)
    graph.add_node("reflect", _reflect_node)
    graph.add_node("refine", _refine_node)
    graph.set_entry_point("search")
    graph.add_edge("search", "draft")
    graph.add_edge("draft", "reflect")
    graph.add_edge("reflect", "refine")
    graph.add_edge("refine", END)
    return graph.compile()


async def run_research(question: str, mode: str = "fast") -> ResearchResponse:
    start = time.monotonic()
    graph = build_research_graph()
    state = graph.invoke({
        "question": question,
        "search_results": [],
        "draft": "",
        "reflection": "",
        "final_summary": "",
        "sources": [],
        "mode": mode,
    })
    latency = (time.monotonic() - start) * 1000
    return ResearchResponse(
        summary=state["final_summary"],
        sources=state["sources"],
        reflection_used=True,
        latency_ms=round(latency, 2),
    )
