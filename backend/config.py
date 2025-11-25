"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM provider selection: "openrouter" (default) or "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Default local model to fall back to when an Ollama-hosted option is missing
LOCAL_DEFAULT_MODEL = os.getenv("LOCAL_DEFAULT_MODEL", "llama3.1")

# Council members - list of model identifiers
COUNCIL_MODELS = [
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4",
]

# Chairman model - synthesizes final response
if LLM_PROVIDER == "ollama":
    CHAIRMAN_MODEL = os.getenv("CHAIRMAN_MODEL", os.getenv("OLLAMA_CHAIRMAN_MODEL", LOCAL_DEFAULT_MODEL))
else:
    CHAIRMAN_MODEL = os.getenv("CHAIRMAN_MODEL", "google/gemini-3-pro-preview")

# Model used for generating concise conversation titles
if LLM_PROVIDER == "ollama":
    TITLE_MODEL = os.getenv("TITLE_MODEL", os.getenv("OLLAMA_TITLE_MODEL", CHAIRMAN_MODEL))
else:
    TITLE_MODEL = os.getenv("TITLE_MODEL", "google/gemini-2.5-flash")

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Ollama API endpoint
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/chat")

# Data directory for conversation storage
DATA_DIR = "data/conversations"
