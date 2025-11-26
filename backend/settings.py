"""Runtime settings management for the LLM Council."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any

from .config import (
    LLM_PROVIDER,
    COUNCIL_MODELS,
    CHAIRMAN_MODEL,
    TITLE_MODEL,
    OLLAMA_API_URL,
    LOCAL_DEFAULT_MODEL,
)

SETTINGS_PATH = os.getenv("SETTINGS_PATH", "data/settings.json")

# Defaults derived from environment/config
DEFAULT_SETTINGS = {
    "llm_provider": LLM_PROVIDER,
    "council_models": COUNCIL_MODELS,
    "chairman_model": CHAIRMAN_MODEL,
    "title_model": TITLE_MODEL,
    "ollama_api_url": OLLAMA_API_URL,
    "local_default_model": LOCAL_DEFAULT_MODEL,
}

_settings_cache: Dict[str, Any] | None = None


def _ensure_settings_dir() -> None:
    """Ensure the settings directory exists."""
    Path(SETTINGS_PATH).parent.mkdir(parents=True, exist_ok=True)


def _load_settings_from_file() -> Dict[str, Any]:
    """Load settings from disk and merge with defaults."""
    if not os.path.exists(SETTINGS_PATH):
        return deepcopy(DEFAULT_SETTINGS)

    try:
        with open(SETTINGS_PATH, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return deepcopy(DEFAULT_SETTINGS)

    merged = deepcopy(DEFAULT_SETTINGS)
    merged.update(data)
    return merged


def save_settings(settings: Dict[str, Any]) -> None:
    """Persist settings to disk and refresh cache."""
    global _settings_cache
    _ensure_settings_dir()
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)
    _settings_cache = deepcopy(settings)


def get_settings() -> Dict[str, Any]:
    """Retrieve the current settings, loading from disk if needed."""
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = _load_settings_from_file()
    return deepcopy(_settings_cache)


def update_settings(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update settings with partial values and persist them."""
    current = get_settings()

    normalized_updates = {}
    for key, value in updates.items():
        if value is None:
            continue
        if key == "llm_provider" and isinstance(value, str):
            normalized_updates[key] = value.lower()
        else:
            normalized_updates[key] = value

    current.update(normalized_updates)
    save_settings(current)
    return current
