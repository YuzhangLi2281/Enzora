"""工具工作流的规划、依赖校验与执行。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from ..tools import ToolContext, ToolRegistry
from ..core.types import BioArtifact
from .models import GoalSpec, PlanStep, WorkflowPlan


class PlanValidationError(ValueError):
    pass


def validate_plan(plan: WorkflowPlan, registry: ToolRegistry) -> None:
    known_steps: dict[str, set[str]] = {}
    initial = set(plan.initial_artifacts)
    capabilities: set[str] = set()
    for step in plan.steps:
        if not step.id or step.id in known_steps:
            raise PlanValidationError('Plan step IDs must be nonempty and unique')
        unknown = set(step.depends_on) - known_steps.keys()
        if unknown:
            raise PlanValidationError(f'Step {step.id!r} has missing or forward dependencies: {sorted(unknown)}')
        try:
            tool = registry.get(step.tool)
        except KeyError as exc:
            raise PlanValidationError(str(exc)) from exc
        health = tool.healthcheck()
        if not health.available:
            raise PlanValidationError(f'Tool {step.tool!r} is unavailable: {list(health.details)}')
        supplied = set(initial)
        for dependency in step.depends_on:
            supplied.update(known_steps[dependency])
        missing = set(tool.spec.inputs) - supplied
        if missing:
            raise PlanValidationError(f'Step {step.id!r} is missing dependency artifacts: {sorted(missing)}')
        known_steps[step.id] = supplied | set(tool.spec.outputs)
        capabilities.update(tool.spec.capabilities)
    missing_capabilities = set(plan.requested_capabilities) - capabilities
    if missing_capabilities:
        raise PlanValidationError(f'Plan does not satisfy capabilities: {sorted(missing_capabilities)}')


class JsonPlannerClient(Protocol):
    def json_object(self, **kwargs: Any) -> dict[str, Any]: ...


class WorkflowPlanner:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def deterministic(self, goal: GoalSpec) -> WorkflowPlan:
        artifacts = set(goal.initial_artifacts)
        remaining = list(dict.fromkeys(goal.capabilities))
        steps: list[PlanStep] = []
        used: set[str] = set()
        while remaining:
            progress = False
            for capability in list(remaining):
                if capability not in remaining:
                    continue
                candidates = [
                    tool for tool in self.registry.providers(capability)
                    if tool.spec.name not in used and set(tool.spec.inputs) <= artifacts
                ]
                if not candidates:
                    continue
                tool = candidates[0]
                step_id = f"step_{len(steps) + 1}_{tool.spec.name}"
                steps.append(PlanStep(step_id, tool.spec.name, tuple(step.id for step in steps)))
                used.add(tool.spec.name)
                artifacts.update(tool.spec.outputs)
                remaining = [cap for cap in remaining if cap not in tool.spec.capabilities]
                progress = True
            if not progress:
                unavailable = {
                    capability: [tool.spec.name for tool in self.registry.providers(capability, available_only=False)]
                    for capability in remaining
                }
                raise PlanValidationError(f"Cannot satisfy requested capabilities: {unavailable}")
        plan = WorkflowPlan(tuple(steps), goal.capabilities, goal.initial_artifacts, "Rule-based provider selection")
        validate_plan(plan, self.registry)
        return plan

    def with_llm(self, goal: GoalSpec, *, client: JsonPlannerClient, model: str) -> WorkflowPlan:
        system_prompt = (
            "You plan a typed scientific workflow. Return JSON only. Select only listed tool names. "
            "Never invent tools or executable code. Steps must be topologically ordered. Each step has "
            "id, tool, and depends_on. Preserve requested_capabilities and initial_artifacts."
        )
        raw = client.json_object(
            model=model,
            system_prompt=system_prompt,
            user_prompt=json.dumps(
                {"goal": {"capabilities": goal.capabilities, "initial_artifacts": goal.initial_artifacts,
                          "constraints": goal.constraints}, "tool_catalog": self.registry.catalog()},
                ensure_ascii=False,
            ),
        )
        raw["requested_capabilities"] = list(goal.capabilities)
        raw["initial_artifacts"] = list(goal.initial_artifacts)
        plan = WorkflowPlan.from_dict(raw)
        validate_plan(plan, self.registry)
        return plan


class WorkflowExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, plan: WorkflowPlan, *, config: dict[str, Any], run_dir: Path,
                initial_artifacts: dict[str, Any] | BioArtifact) -> BioArtifact:
        validate_plan(plan, self.registry)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "00_workflow_plan.json").write_text(
            json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        missing_initial = set(plan.initial_artifacts) - initial_artifacts.keys()
        if missing_initial:
            raise ValueError(f'Missing initial artifacts: {sorted(missing_initial)}')
        from copy import deepcopy
        artifacts = deepcopy(initial_artifacts) if isinstance(initial_artifacts, BioArtifact) else BioArtifact(data=dict(initial_artifacts))
        by_step: dict[str, BioArtifact] = {}
        trace: list[dict[str, Any]] = []
        for step in plan.steps:
            tool = self.registry.get(step.tool)
            supplied = deepcopy(initial_artifacts) if isinstance(initial_artifacts, BioArtifact) else BioArtifact(data=dict(initial_artifacts))
            for dependency in step.depends_on:
                supplied.merge(by_step[dependency])
            context = ToolContext(run_dir, config, supplied, step.id)
            tool.validate_input(context)
            result = tool.run(context)
            tool.validate_output(result)
            if isinstance(result, BioArtifact):
                supplied.merge(result)
            else:
                supplied.update(result)
            by_step[step.id] = supplied
            artifacts.merge(supplied)
            trace.append({"step": step.id, "tool": step.tool, "outputs": list(result)})
        (run_dir / "01_execution_trace.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return artifacts
