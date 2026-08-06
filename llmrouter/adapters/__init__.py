from __future__ import annotations

import httpx

from llmrouter.adapters.base import BaseAdapter
from llmrouter.adapters.cerebras_adapter import CerebrasAdapter
from llmrouter.adapters.cloudflare_adapter import CloudflareAdapter
from llmrouter.adapters.cohere_adapter import CohereAdapter
from llmrouter.adapters.deepinfra_adapter import DeepInfraAdapter
from llmrouter.adapters.deepseek_adapter import DeepSeekAdapter
from llmrouter.adapters.gemini_adapter import GeminiAdapter
from llmrouter.adapters.groq_adapter import GroqAdapter
from llmrouter.adapters.huggingface_adapter import HuggingFaceAdapter
from llmrouter.adapters.mistral_adapter import MistralAdapter
from llmrouter.adapters.nvidia_adapter import NvidiaAdapter
from llmrouter.adapters.openrouter_adapter import OpenRouterAdapter
from llmrouter.adapters.sambanova_adapter import SambaNovaAdapter
from llmrouter.adapters.together_adapter import TogetherAdapter
from llmrouter.exceptions import ProviderError
from llmrouter.registry import ModelConfig, resolve_auth_env

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
}


def get_adapter(model: ModelConfig, client: httpx.AsyncClient | None = None) -> BaseAdapter:
    cls = _ADAPTERS.get(model.provider)
    if cls is None:
        raise ProviderError(f"unsupported provider: {model.provider}", retryable=False, provider=model.provider)
    key = resolve_auth_env(model.auth_env_var)
    if not key:
        raise ProviderError(
            f"missing env {model.auth_env_var}",
            retryable=False,
            provider=model.provider,
            model=model.name,
        )
    return cls(model, key, client=client)
