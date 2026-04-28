from fastapi import APIRouter, Depends
from backend.models.schemas import (
    ResearchRequest, ResearchResponse,
    CrewRequest, CrewResponse,
    AutoGenRequest, AutoGenResponse,
)
from backend.services.research_agent import run_research
from backend.services.crew_service import run_crew
from backend.services.autogen_service import run_autogen
from backend.dependencies import get_redis

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest, redis=Depends(get_redis)):
    return await run_research(req.question, req.mode)


@router.post("/crew", response_model=CrewResponse)
async def crew(req: CrewRequest, redis=Depends(get_redis)):
    return await run_crew(req.topic, req.mode)


@router.post("/autogen", response_model=AutoGenResponse)
async def autogen(req: AutoGenRequest, redis=Depends(get_redis)):
    """
    AutoGen-style 3-agent GroupChat: Assistant → Critic → Refiner.
    Mirrors AutoGen's GroupChat pattern using Groq (free).
    """
    return await run_autogen(req.question, req.mode)
