"""Free-tier LLM dispatcher — library entrypoints."""

from llmcascade.adapters.base import LLMResponse
from llmcascade.cascade import (
    GeminiCascadeManager,
    ModelCooldownTracker,
    classify_failure,
    resolve_cascade_order,
)
from llmcascade.exceptions import (
    AllModelsExhaustedError,
    ProviderError,
    QueueFullError,
    RegistryError,
)
from llmcascade.queue_worker import RouterClient
from llmcascade.rate_limiter import InMemoryBudgetStore, RateLimiter
from llmcascade.registry import ModelConfig, load_registry
from llmcascade.selector import ModelSelector
from llmcascade.tokens import estimate_tokens

__all__ = [
    "AllModelsExhaustedError",
    "GeminiCascadeManager",
    "InMemoryBudgetStore",
    "LLMResponse",
    "ModelConfig",
    "ModelCooldownTracker",
    "ModelSelector",
    "ProviderError",
    "QueueFullError",
    "RateLimiter",
    "RegistryError",
    "RouterClient",
    "classify_failure",
    "estimate_tokens",
    "load_registry",
    "resolve_cascade_order",
]

__version__ = "0.1.0"
