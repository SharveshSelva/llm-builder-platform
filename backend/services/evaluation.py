from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.config import get_settings
from backend.models.schemas import EvalResult

settings = get_settings()

_ragas_llm = LangchainLLMWrapper(
    ChatGroq(
        model=settings.primary_model,
        groq_api_key=settings.groq_api_key,
        temperature=0,
    )
)
_ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=settings.embedding_model)
)


async def run_ragas_eval(
    query: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
) -> EvalResult:
    """
    Runs RAGAS evaluation using Groq as the LLM judge
    and sentence-transformers for embeddings (both free).
    """
    data = {
        "question": [query],
        "answer": [answer],
        "contexts": [contexts],
    }
    metrics = [faithfulness, answer_relevancy]
    if ground_truth:
        data["ground_truth"] = [ground_truth]
        metrics.append(context_precision)

    for metric in metrics:
        metric.llm = _ragas_llm
        metric.embeddings = _ragas_embeddings

    dataset = Dataset.from_dict(data)
    result = evaluate(dataset, metrics=metrics)
    scores = result.to_pandas().iloc[0]

    return EvalResult(
        faithfulness=round(float(scores.get("faithfulness", 0)), 4),
        answer_relevancy=round(float(scores.get("answer_relevancy", 0)), 4),
        context_precision=round(float(scores.get("context_precision", 0)), 4) if ground_truth else None,
    )
