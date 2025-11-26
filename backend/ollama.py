"""Ollama API client for making LLM requests and fetching models."""

import time
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin

import httpx

from .config import OLLAMA_API_URL
from .settings import get_settings

DEFAULT_TIMEOUT = 120.0


def _get_chat_url() -> str:
    settings = get_settings()
    return settings.get("ollama_api_url", OLLAMA_API_URL)


def _get_base_url() -> str:
    chat_url = _get_chat_url()
    if "/api/chat" in chat_url:
        return chat_url.split("/api/chat", maxsplit=1)[0]
    if chat_url.endswith("/api"):
        return chat_url[: -len("/api")]
    if chat_url.endswith("/api/"):
        return chat_url[: -len("/api/")]
    return chat_url.rstrip("/")


def _tags_url() -> str:
    base = _get_base_url()
    return urljoin(base + "/", "api/tags")


async def list_models(timeout: float = 15.0) -> List[Dict[str, Any]]:
    """Return the list of available Ollama models."""
    tags_url = _tags_url()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(tags_url)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            return [
                {
                    "name": model.get("name"),
                    "modified_at": model.get("modified_at"),
                    "size": model.get("size"),
                }
                for model in models
                if model.get("name")
            ]
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
        print(f"Error parsing Ollama response: {exc}")
    except Exception as exc:  # pragma: no cover - unexpected failures
        print(f"Unexpected error listing ollama models: {exc}")

    return []


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

    chat_url = _get_chat_url()

    try:
        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                chat_url,
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
