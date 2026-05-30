# ADR 005: LLM adapter layer

- **Date:** 2026-05-30
- **Status:** Accepted

## Context

The calibration agent needs an LLM to diagnose the sim-vs-real gap and propose
parameter updates. We want a local-first default for cost and reproducibility,
but the local model may be too weak, so we need a clean escape hatch to a hosted
model without rewriting the agent.

## Decision

Define one `LLMClient` interface, `generate(messages) -> str`, with three
concrete implementations (Ollama, OpenAI, Anthropic) selected at runtime by the
`SENSORFORGE_LLM` env var, defaulting to Ollama. The rest of the agent depends
only on the interface.

## Consequences

- The graph and tools never import a vendor SDK; swapping providers is an env
  var, not a code change.
- Provider quirks stay behind the adapter: Anthropic's separate `system`
  parameter, OpenAI's chat-completions shape, and Ollama's local `chat` are all
  normalized to the same `list[Message] -> str` contract.
- Ollama (Llama 3.2) is the default; OpenAI and Anthropic are opt-in fallbacks
  for when the local model cannot hit the calibration target.
- Testing mocks the underlying SDKs, so the adapter and the whole agent run in
  CI with no API keys and no local model.
- We pass plain strings, not structured tool-calls, so the proposal contract is
  a JSON schema the prompt asks for and we validate, keeping all three providers
  on equal footing.
