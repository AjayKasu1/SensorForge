"""LLM adapter: one interface, three backends, chosen by env var.

The agent depends only on ``LLMClient.generate``. Provider-specific shapes
(Anthropic's separate system arg, OpenAI's chat completions, Ollama's local
chat) are normalized here. See ADR 005.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMClient(Protocol):
    def generate(self, messages: list[Message]) -> str: ...


def _as_dicts(messages: list[Message]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


class OllamaClient:
    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL):
        import ollama

        self.model = model
        self._chat = ollama.chat

    def generate(self, messages: list[Message]) -> str:
        resp = self._chat(model=self.model, messages=_as_dicts(messages))
        return resp["message"]["content"]


class OpenAIClient:
    def __init__(self, model: str = DEFAULT_OPENAI_MODEL):
        import openai

        self.model = model
        self.client = openai.OpenAI()

    def generate(self, messages: list[Message]) -> str:
        resp = self.client.chat.completions.create(model=self.model, messages=_as_dicts(messages))
        return resp.choices[0].message.content


class AnthropicClient:
    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic()

    def generate(self, messages: list[Message]) -> str:
        # Anthropic takes the system prompt separately from the turn list.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = _as_dicts([m for m in messages if m.role != "system"])
        resp = self.client.messages.create(
            model=self.model, system=system, messages=turns, max_tokens=1024
        )
        return resp.content[0].text


def make_llm_client(provider: str | None = None) -> LLMClient:
    """Build the client named by ``provider`` or the SENSORFORGE_LLM env var."""
    provider = (provider or os.environ.get("SENSORFORGE_LLM", "ollama")).lower()
    if provider == "ollama":
        return OllamaClient()
    if provider == "openai":
        return OpenAIClient()
    if provider == "anthropic":
        return AnthropicClient()
    raise ValueError(f"unknown LLM provider {provider!r}; use ollama, openai, or anthropic")
