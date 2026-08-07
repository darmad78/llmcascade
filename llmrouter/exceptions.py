"""llmrouter exceptions."""


class RegistryError(Exception):
    pass


class ProviderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        provider: str | None = None,
        model: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.provider = provider
        self.model = model
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}


class AllModelsExhaustedError(Exception):
    pass


class QueueFullError(Exception):
    pass
