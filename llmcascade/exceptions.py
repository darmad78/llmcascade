"""llmcascade exceptions."""


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

    @property
    def safe_message(self) -> str:
        """Status/provider/model only — never include provider response bodies."""
        parts = []
        if self.provider:
            parts.append(self.provider)
        if self.model:
            parts.append(self.model)
        who = "/".join(parts) if parts else "provider"
        if self.status_code is not None:
            return f"{who} HTTP {self.status_code}"
        return f"{who} error"


class AllModelsExhaustedError(Exception):
    def __init__(self, message: str, *, http_status: int = 502) -> None:
        super().__init__(message)
        self.http_status = http_status


class QueueFullError(Exception):
    pass


def safe_error_message(exc: BaseException) -> str:
    """Public-safe error string for logs, events, and HTTP responses."""
    if isinstance(exc, ProviderError):
        return exc.safe_message
    if isinstance(exc, AllModelsExhaustedError):
        text = str(exc)
        if "; last error:" in text:
            return text.split("; last error:", 1)[0].strip()
        return text
    return type(exc).__name__
