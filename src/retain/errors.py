"""Exception hierarchy for retain."""

__all__ = [
    "RetainConfigError",
    "RetainError",
    "RetainLLMError",
    "RetainNotImplementedError",
    "RetainStorageError",
]


class RetainError(Exception):
    """Base exception for all retain errors."""


class RetainStorageError(RetainError):
    """Database or storage operation failed."""


class RetainConfigError(RetainError):
    """Invalid configuration."""


class RetainLLMError(RetainError):
    """LLM API call failed."""


class RetainNotImplementedError(RetainError, NotImplementedError):
    """Feature not yet implemented."""
