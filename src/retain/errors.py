"""Exception hierarchy for retain."""

__all__ = [
    "RetainError",
    "RetainStorageError",
    "RetainConfigError",
    "RetainNotImplementedError",
]


class RetainError(Exception):
    """Base exception for all retain errors."""


class RetainStorageError(RetainError):
    """Database or storage operation failed."""


class RetainConfigError(RetainError):
    """Invalid configuration."""


class RetainNotImplementedError(RetainError, NotImplementedError):
    """Feature not yet implemented."""
