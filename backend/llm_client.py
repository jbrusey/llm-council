"""Unified client that routes requests to the configured LLM provider."""

from typing import List, Dict, Any, Optional

from .config import LLM_PROVIDER
from . import openrouter, ollama


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
) -> Optional[Dict[str, Any]]:
    """Query a single model using the configured provider."""
    if LLM_PROVIDER == "ollama":
        return await ollama.query_model(model, messages, timeout)
    return await openrouter.query_model(model, messages, timeout)


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Query multiple models in parallel using the configured provider."""
    import asyncio

    tasks = [query_model(model, messages) for model in models]
    responses = await asyncio.gather(*tasks)
    return {model: response for model, response in zip(models, responses)}
