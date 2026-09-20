# ADR 0003: Generate workflows from declarative AgentSpec objects

- Status: accepted
- Date: 2026-09-20

## Decision

Represent the input to agent generation as a validated, version-controlled `AgentSpec`.
Compile it first into a runtime-neutral workflow plan. A LangGraph adapter may execute
that plan, but LangGraph types must not appear in the domain model.

## Rationale

This boundary makes agent generation an explicit and testable research contribution.
It also permits controlled ablations of retrieval, reranking, tools, and verification
without rewriting application code.

## Consequences

- Generated workflows can be inspected before execution.
- Agent configurations can be stored with experimental results.
- A different graph runtime can be introduced without changing the research model.

