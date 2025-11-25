"""Ollama API client for making LLM requests."""

import time
import httpx
from typing import List, Dict, Any, Optional
from .config import OLLAMA_API_URL

DEFAULT_TIMEOUT = 120.0


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = DEFAULT_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via the Ollama API.

    Args:
        model: Ollama model identifier.
        messages: List of message dicts with 'role' and 'content'.
        timeout: Request timeout in seconds.

    Returns:
        Response dict with 'content', or None if failed.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    try:
        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                OLLAMA_API_URL,
                json=payload,
            )
            response.raise_for_status()

            elapsed = time.perf_counter() - start_time

            data = response.json()
            message = data.get("message", {})

            return {
                "content": message.get("content"),
                "reasoning_details": message.get("reasoning_details"),
                "response_time": elapsed,
            }
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        detail = exc.response.text
        request_summary = f"{exc.request.method} {exc.request.url}"
        print(
            "Ollama request failed with status "
            f"{status} for {request_summary}: {detail}"
        )
    except httpx.RequestError as exc:
        print(f"Ollama request error for {getattr(exc.request, 'url', 'unknown URL')}: {exc}")
    except ValueError as exc:
        print(f"Error parsing Ollama response for model {model}: {exc}")
    except Exception as e:
        print(f"Unexpected error querying ollama model {model}: {e}")

    return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of Ollama model identifiers.
        messages: List of message dicts to send to each model.

    Returns:
        Dict mapping model identifier to response dict (or None if failed).
    """
    import asyncio

    tasks = [query_model(model, messages) for model in models]
    responses = await asyncio.gather(*tasks)
    return {model: response for model, response in zip(models, responses)}
