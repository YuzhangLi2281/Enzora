from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..core.types import BioArtifact


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    description: str
    capabilities: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    priority: int = 100
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or any(char.isspace() for char in self.name):
            raise ValueError("Tool names must be non-empty and contain no whitespace")
        if not self.capabilities or not self.outputs:
            raise ValueError(f"Tool {self.name!r} must declare capabilities and outputs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": list(self.capabilities),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "priority": self.priority,
            "properties": dict(self.properties),
        }


@dataclass(frozen=True)
class ToolHealth:
    available: bool
    details: tuple[str, ...] = ()


@dataclass
class ToolContext:
    run_dir: Path
    config: dict[str, Any]
    artifacts: BioArtifact
    step_id: str

    def __post_init__(self):
        if not isinstance(self.artifacts, BioArtifact):
            self.artifacts = BioArtifact(data=dict(self.artifacts))


class AgentTool(ABC):
    spec: ToolSpec

    def healthcheck(self) -> ToolHealth:
        return ToolHealth(True)

    def validate_input(self, context: ToolContext) -> None:
        missing = [name for name in self.spec.inputs if name not in context.artifacts]
        if missing:
            raise ValueError(f"Tool {self.spec.name!r} is missing inputs: {missing}")

    @abstractmethod
    def run(self, context: ToolContext) -> Mapping[str, Any]:
        """Return a mapping containing every output declared by the specification."""

    def validate_output(self, result: Mapping[str, Any]) -> None:
        missing = [name for name in self.spec.outputs if name not in result]
        if missing:
            raise ValueError(f"Tool {self.spec.name!r} omitted outputs: {missing}")
