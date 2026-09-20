# ADR 0001: Use a modular monolith

- Status: accepted
- Date: 2026-09-20

## Decision

Implement the research platform as one deployable Python application with explicit
domain, application, infrastructure, and API boundaries.

## Rationale

The dissertation needs reproducible comparisons between retrieval and agent-generation
methods. A modular monolith keeps those comparisons in one environment without coupling
domain logic to external providers. Ports allow individual components to be replaced in
experiments. Microservices would add operational variables without answering a research
question.

## Consequences

- All experiments use the same domain models and evaluation contracts.
- Infrastructure adapters can be replaced independently.
- A service can be extracted later if deployment requirements justify it.

