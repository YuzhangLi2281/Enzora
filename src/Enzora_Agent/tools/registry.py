from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Iterable

from .base import AgentTool, ToolHealth


class ToolRegistry:
    """Runtime tool catalog with explicit registration and plug-in discovery."""

    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}

    def register(self, tool: AgentTool, *, replace: bool = False) -> None:
        name = tool.spec.name
        if name in self._tools and not replace:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def unregister(self, name: str) -> AgentTool:
        try:
            return self._tools.pop(name)
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def get(self, name: str) -> AgentTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def all(self) -> tuple[AgentTool, ...]:
        return tuple(self._tools[name] for name in sorted(self._tools))

    def providers(self, capability: str, *, available_only: bool = True) -> list[AgentTool]:
        providers = [tool for tool in self._tools.values() if capability in tool.spec.capabilities]
        if available_only:
            providers = [tool for tool in providers if tool.healthcheck().available]
        return sorted(providers, key=lambda tool: (tool.spec.priority, tool.spec.name))

    def catalog(self) -> list[dict]:
        catalog = []
        for tool in self.all():
            health = tool.healthcheck()
            item = tool.spec.to_dict()
            item["health"] = {"available": health.available, "details": list(health.details)}
            catalog.append(item)
        return catalog

    def discover(self, roots: Iterable[str | Path]) -> list[str]:
        """Load plug-ins from directories containing tool.json and tool.py.

        tool.py must expose create_tool(manifest) and return an AgentTool. Importing
        arbitrary Python is an explicit administrator action through configured roots.
        """
        loaded: list[str] = []
        for root_value in roots:
            root = Path(root_value).expanduser().resolve()
            if not root.is_dir():
                continue
            for manifest_path in sorted(root.glob("*/tool.json")):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                module_path = manifest_path.with_name("tool.py")
                if not module_path.is_file():
                    raise ValueError(f"Plug-in has no tool.py: {manifest_path.parent}")
                module_name = f"fixed_enzyme_plugin_{manifest_path.parent.name}"
                module_spec = importlib.util.spec_from_file_location(module_name, module_path)
                if module_spec is None or module_spec.loader is None:
                    raise ImportError(f"Cannot load plug-in: {module_path}")
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
                factory = getattr(module, "create_tool", None)
                if not callable(factory):
                    raise ValueError(f"Plug-in must expose create_tool(manifest): {module_path}")
                tool = factory(manifest)
                if not isinstance(tool, AgentTool):
                    raise TypeError(f"Plug-in factory did not return AgentTool: {module_path}")
                if tool.spec.name != manifest.get("name"):
                    raise ValueError(f"Manifest and tool names differ: {manifest_path}")
                self.register(tool)
                loaded.append(tool.spec.name)
        return loaded
