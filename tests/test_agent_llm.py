from unittest.mock import MagicMock, patch

import pytest

from sensorforge.agent.llm import (
    AnthropicClient,
    Message,
    OllamaClient,
    OpenAIClient,
    make_llm_client,
)


def test_ollama_client_normalizes_messages():
    with patch("ollama.chat", return_value={"message": {"content": "hi"}}) as chat:
        out = OllamaClient(model="llama3.2").generate([Message("user", "hello")])
    assert out == "hi"
    assert chat.call_args.kwargs["messages"] == [{"role": "user", "content": "hello"}]


def test_anthropic_splits_system_prompt():
    fake = MagicMock()
    fake.messages.create.return_value.content = [MagicMock(text="ok")]
    with patch("anthropic.Anthropic", return_value=fake):
        out = AnthropicClient().generate([Message("system", "be terse"), Message("user", "go")])
    assert out == "ok"
    kwargs = fake.messages.create.call_args.kwargs
    assert kwargs["system"] == "be terse"  # system pulled out of the turn list
    assert kwargs["messages"] == [{"role": "user", "content": "go"}]


def test_openai_returns_first_choice():
    fake = MagicMock()
    fake.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="x"))]
    with patch("openai.OpenAI", return_value=fake):
        out = OpenAIClient().generate([Message("user", "q")])
    assert out == "x"


def test_factory_routes_by_env(monkeypatch):
    monkeypatch.setenv("SENSORFORGE_LLM", "ollama")
    assert isinstance(make_llm_client(), OllamaClient)


def test_factory_model_override(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    client = make_llm_client("ollama", "llama3.1:8b")
    assert isinstance(client, OllamaClient)
    assert client.model == "llama3.1:8b"
    # No override falls back to the provider default.
    assert make_llm_client("ollama").model == "llama3.2"


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        make_llm_client("gpt5")
