"""Backwards-compatible re-export.

The chat client moved to :mod:`quantagent.llm.client` once LLM forecasting became part
of the product rather than just the agent's plumbing. Keeping this module means older
imports (``from .llm import LLMClient``) keep working.
"""

from ..llm.client import ChatResponse, LLMClient, LLMError, MockClient, extract_json

__all__ = ["ChatResponse", "LLMClient", "LLMError", "MockClient", "extract_json"]
