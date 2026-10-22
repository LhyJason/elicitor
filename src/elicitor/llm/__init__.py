"""LLM layer (SPEC §14): roles, providers, a reply cache and one client that ties them together.

The rest of the package only talks to `LLMClient.call(role, context, seed)`, which returns a
validated pydantic object or raises `LLMParseError` (every attempt failed validation) or
`LLMTransportError` (the server could not be reached).
"""

from elicitor.llm.client import LLMClient, LLMParseError, LLMTransportError, build_client

__all__ = ["LLMClient", "LLMParseError", "LLMTransportError", "build_client"]
