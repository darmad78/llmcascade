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
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.provider = provider
        self.model = model


class AllModelsExhaustedError(Exception):
    pass


class QueueFullError(Exception):
    pass
