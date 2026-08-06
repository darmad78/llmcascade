"""Free-tier LLM dispatcher — library entrypoints."""

from llmrouter.adapters.base import LLMResponse
from llmrouter.exceptions import (
    AllModelsExhaustedError,
    ProviderError,
    QueueFullError,
    RegistryError,
)
from llmrouter.queue_worker import RouterClient
from llmrouter.rate_limiter import InMemoryBudgetStore, RateLimiter
from llmrouter.registry import ModelConfig, load_registry
from llmrouter.selector import ModelSelector
from llmrouter.tokens import estimate_tokens

__all__ = [
    "AllModelsExhaustedError",
    "InMemoryBudgetStore",
    "LLMResponse",
    "ModelConfig",
    "ModelSelector",
    "ProviderError",
    "QueueFullError",
    "RateLimiter",
    "RegistryError",
    "RouterClient",
    "estimate_tokens",
    "load_registry",
]

__version__ = "0.1.0"
