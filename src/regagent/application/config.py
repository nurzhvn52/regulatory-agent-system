"""Load version-controlled research configurations."""

from pathlib import Path

import yaml

from regagent.domain.agents import AgentSpec


def load_agent_spec(path: Path) -> AgentSpec:
    """Load and validate an agent specification from YAML."""

    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Agent specification must be a mapping: {path}")
    return AgentSpec.model_validate(raw)

