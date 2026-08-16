from __future__ import annotations

import httpx

from llmcascade.adapters.base import BaseAdapter
from llmcascade.adapters.cerebras_adapter import CerebrasAdapter
from llmcascade.adapters.cloudflare_adapter import CloudflareAdapter
from llmcascade.adapters.cohere_adapter import CohereAdapter
from llmcascade.adapters.deepinfra_adapter import DeepInfraAdapter
from llmcascade.adapters.deepseek_adapter import DeepSeekAdapter
from llmcascade.adapters.gemini_adapter import GeminiAdapter
from llmcascade.adapters.groq_adapter import GroqAdapter
from llmcascade.adapters.huggingface_adapter import HuggingFaceAdapter
from llmcascade.adapters.jina_adapter import JinaAdapter
from llmcascade.adapters.mistral_adapter import MistralAdapter
from llmcascade.adapters.mixedbread_adapter import MixedbreadAdapter
from llmcascade.adapters.nomic_adapter import NomicAdapter
from llmcascade.adapters.nvidia_adapter import NvidiaAdapter
from llmcascade.adapters.openrouter_adapter import OpenRouterAdapter
from llmcascade.adapters.sambanova_adapter import SambaNovaAdapter
from llmcascade.adapters.siliconflow_adapter import SiliconFlowAdapter
from llmcascade.adapters.together_adapter import TogetherAdapter
from llmcascade.adapters.voyage_adapter import VoyageAdapter
from llmcascade.exceptions import ProviderError
from llmcascade.registry import ModelConfig, resolve_auth_env

_ADAPTERS: dict[str, type[BaseAdapter]] = {
    "groq": GroqAdapter,
    "gemini": GeminiAdapter,
    "openrouter": OpenRouterAdapter,
    "together": TogetherAdapter,
    "cerebras": CerebrasAdapter,
    "mistral": MistralAdapter,
    "sambanova": SambaNovaAdapter,
    "deepseek": DeepSeekAdapter,
    "huggingface": HuggingFaceAdapter,
    "cloudflare": CloudflareAdapter,
    "cohere": CohereAdapter,
    "nvidia": NvidiaAdapter,
    "deepinfra": DeepInfraAdapter,
    "jina": JinaAdapter,
    "voyage": VoyageAdapter,
    "nomic": NomicAdapter,
    "mixedbread": MixedbreadAdapter,
    "siliconflow": SiliconFlowAdapter,
}


def get_adapter(model: ModelConfig, client: httpx.AsyncClient | None = None) -> BaseAdapter:
    cls = _ADAPTERS.get(model.provider)
    if cls is None:
        raise ProviderError(f"unsupported provider: {model.provider}", retryable=False, provider=model.provider)
    key = resolve_auth_env(model.auth_env_var, provider=model.provider, key_tier=model.key_tier)
    if not key:
        raise ProviderError(
            f"missing env {model.auth_env_var}",
            retryable=False,
            provider=model.provider,
            model=model.name,
        )
    return cls(model, key, client=client)
