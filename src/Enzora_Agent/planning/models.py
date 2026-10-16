from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GoalSpec:
    capabilities: tuple[str, ...]
    initial_artifacts: tuple[str, ...] = ("protein_sequence",)
    constraints: dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""


@dataclass(frozen=True)
class PlanStep:
    id: str
    tool: str
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkflowPlan:
    steps: tuple[PlanStep, ...]
    requested_capabilities: tuple[str, ...]
    initial_artifacts: tuple[str, ...]
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_capabilities": list(self.requested_capabilities),
            "initial_artifacts": list(self.initial_artifacts),
            "rationale": self.rationale,
            "steps": [
                {"id": step.id, "tool": step.tool, "depends_on": list(step.depends_on)}
                for step in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WorkflowPlan":
        raw_steps = value.get("steps")
        if not isinstance(raw_steps, list):
            raise ValueError("Plan steps must be an array")
        return cls(
            steps=tuple(
                PlanStep(
                    id=str(item["id"]),
                    tool=str(item["tool"]),
                    depends_on=tuple(map(str, item.get("depends_on", []))),
                )
                for item in raw_steps
            ),
            requested_capabilities=tuple(map(str, value.get("requested_capabilities", []))),
            initial_artifacts=tuple(map(str, value.get("initial_artifacts", ["protein_sequence"]))),
            rationale=str(value.get("rationale", "")),
        )
