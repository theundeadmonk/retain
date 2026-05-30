"""API key authentication middleware.

In Phase C this will validate API keys against the database.
For now, every request passes through.
"""

from fastapi import Request


async def authenticate(request: Request) -> None:
    """Validate the API key in the Authorization header."""
    return
