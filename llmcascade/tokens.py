"""Cheap token estimate for TPM gating (no tiktoken)."""


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
