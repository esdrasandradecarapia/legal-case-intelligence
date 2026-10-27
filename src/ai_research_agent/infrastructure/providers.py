from ai_research_agent.domain.ports import LLM, Embedder
from ai_research_agent.infrastructure.config import Settings
from ai_research_agent.infrastructure.ollama_adapter import OllamaEmbedder, OllamaLLM
from ai_research_agent.infrastructure.openai_adapter import OpenAIEmbedder, OpenAILLM


def build_embedder(settings: Settings) -> Embedder:
    if settings.llm_provider == "openai":
        return OpenAIEmbedder(api_key=settings.openai_api_key)
    return OllamaEmbedder()


def build_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "openai":
        return OpenAILLM(api_key=settings.openai_api_key)
    return OllamaLLM()