# ADR 002: LangGraph for the calibration loop

- **Date:** 2026-05-30
- **Status:** Accepted

## Context

The calibration agent is an iterative loop with state: measure, diagnose,
propose, apply, repeat until a stop condition. We need an agent framework whose
native idiom is exactly that, with explicit nodes, conditional edges, and a
typed state object, not a role-playing crew.

## Decision

Use LangGraph. The loop is a `StateGraph` over a pydantic `AgentState` with two
nodes (measure, propose) and a conditional edge that ends the run when a stop
reason is set.

## Consequences

- The control flow is explicit and inspectable: `measure -> [continue|stop] ->
  propose -> measure`. Stop conditions live in one place.
- State is a typed pydantic object, so the run record (history, assumptions,
  best) is serializable and testable without the LLM.
- CrewAI and similar role-play frameworks were rejected: they model
  collaborating personas, which adds ceremony with no fit for a single
  numeric optimization loop.
- LangGraph's recursion limit must be raised above twice the iteration budget,
  since each round runs two nodes.
- The LLM sits behind our own adapter (ADR 005), so we use LangGraph for
  orchestration only, not for its model integrations.
