"""Free-tier LLM dispatcher — library entrypoints."""

from llmrouter.adapters.base import LLMResponse
from llmrouter.cascade import (
    GeminiCascadeManager,
    ModelCooldownTracker,
    classify_failure,
    resolve_cascade_order,
)
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
