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

# Default prompt templates
DEFAULT_RANKING_PROMPT = """You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line \"FINAL RANKING:\" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., \"1. Response A\")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

DEFAULT_CHAIRMAN_PROMPT = """You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

DEFAULT_TITLE_PROMPT = """Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

# Defaults derived from environment/config
DEFAULT_SETTINGS = {
    "llm_provider": LLM_PROVIDER,
    "council_models": COUNCIL_MODELS,
    "chairman_model": CHAIRMAN_MODEL,
    "title_model": TITLE_MODEL,
    "ollama_api_url": OLLAMA_API_URL,
    "local_default_model": LOCAL_DEFAULT_MODEL,
    "ranking_prompt": DEFAULT_RANKING_PROMPT,
    "chairman_prompt": DEFAULT_CHAIRMAN_PROMPT,
    "title_prompt": DEFAULT_TITLE_PROMPT,
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
