from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

from .config import load_config


def _print_json(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _resolve_api_key(args, cfg: dict) -> str | None:
    llm_enabled = bool(cfg.get("llm", {}).get("enabled", False))
    if not llm_enabled:
        return None
    if getattr(args, "api_key", None):
        return args.api_key
    key_name = {"minimax": "MINIMAX_API_KEY", "gemini": "GEMINI_API_KEY"}[cfg["llm"]["provider"]]
    if os.environ.get(key_name):
        return os.environ[key_name]
    if getattr(args, "ask_api_key", False):
        return getpass.getpass(f"{cfg['llm']['provider']} API key: ")
    return None


def _apply_run_overrides(args, cfg: dict) -> None:
    request = getattr(args, "request", None)
    if request:
        cfg["user_request"] = request
        cfg.setdefault("llm", {})["enabled"] = True
    if getattr(args, "enable_llm", False):
        cfg.setdefault("llm", {})["enabled"] = True
    if getattr(args, "disable_llm", False):
        cfg.setdefault("llm", {})["enabled"] = False
    report_lang = getattr(args, "report_lang", None)
    if report_lang:
        cfg.setdefault("report", {})["language"] = report_lang


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--request",
        help="自然语言筛选目标；提供后自动启用 MiniMax 参数解析",
    )
    parser.add_argument(
        "--api-key",
        help="MiniMax API key（会进入 shell history；更推荐环境变量或 --ask-api-key）",
    )
    parser.add_argument(
        "--ask-api-key",
        action="store_true",
        help="在终端安全交互输入 MiniMax API key，不回显",
    )
    parser.add_argument("--enable-llm", action="store_true", help="强制启用 MiniMax")
    parser.add_argument("--disable-llm", action="store_true", help="本次运行关闭 MiniMax")
    parser.add_argument(
        "--print-llm",
        action="store_true",
        help="将 MiniMax 参数解析 JSON 和结果分析同时输出到终端",
    )
    parser.add_argument(
        "--report-lang",
        choices=["zh", "en"],
        help="Markdown/MiniMax 分析输出语言；覆盖 config.report.language",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed Enzyme Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="运行完整固定链路")
    p_run.add_argument("config")
    p_run.add_argument("--cpu", action="store_true", help="完整链路强制 CPU（非常慢，仅调试）")
    _add_run_args(p_run)

    p_cpu = sub.add_parser("validate-cpu", help="真实加载 EpHod/PHENOM/UniStab 并在 CPU 各跑一次")
    p_cpu.add_argument("config")

    p_gpu = sub.add_parser("check-gpu", help="检查三个模型 Python 环境是否都能看到选定 GPU")
    p_gpu.add_argument("config")

    p_paths = sub.add_parser("check-paths", help="只检查仓库、入口、模型权重路径")
    p_paths.add_argument("config")

    p_ds = sub.add_parser("gemini-test", help="从命令行测试 Gemini API，并把响应打印到终端")
    p_ds.add_argument("--api-key")
    p_ds.add_argument("--ask-api-key", action="store_true")
    p_ds.add_argument("--model", default="gemini-2.5-flash", choices=["gemini-2.5-flash", "gemini-2.5-pro"])
    p_ds.add_argument("--prompt", default="请只回复：Gemini API OK")
    p_ds.add_argument("--base-url", default="https://generativelanguage.googleapis.com/v1beta")

    p_tools = sub.add_parser("tools", help="List tools discovered from plug-in roots")
    p_tools.add_argument("--plugin-root", action="append", default=[])
    p_tools.add_argument("--config", help="Also register built-in tools from this configuration")

    p_plan = sub.add_parser("plan", help="Build and validate a capability-based workflow plan")
    p_plan.add_argument("--capability", action="append", required=True)
    p_plan.add_argument("--initial-artifact", action="append", default=["protein_sequence"])
    p_plan.add_argument("--plugin-root", action="append", default=[])
    p_plan.add_argument("--config", help="Also register built-in tools from this configuration")
    p_plan.add_argument("--request", default="")
    p_plan.add_argument("--llm", action="store_true")
    p_plan.add_argument("--model", default="MiniMax-M2.5")
    p_plan.add_argument("--api-key")
    p_plan.add_argument("--ask-api-key", action="store_true")
    p_plan.add_argument("--base-url", default="https://api.minimax.cn/v1")

    args = parser.parse_args()

    if args.command in {"tools", "plan"}:
        from .planning import GoalSpec, WorkflowPlanner
        from .tools import ToolRegistry, register_configured_tools

        registry = ToolRegistry()
        if args.config:
            register_configured_tools(registry, load_config(args.config))
        plugin_roots = args.plugin_root or ["plugins"]
        registry.discover(Path(root) for root in plugin_roots)
        if args.command == "tools":
            _print_json(registry.catalog())
            return
        goal = GoalSpec(
            capabilities=tuple(args.capability),
            initial_artifacts=tuple(dict.fromkeys(args.initial_artifact)),
            interpretation=args.request,
        )
        planner = WorkflowPlanner(registry)
        if args.llm:
            from .llm.gemini import MiniMaxClient
            key = args.api_key or os.environ.get('MINIMAX_API_KEY')
            if not key and args.ask_api_key:
                key = getpass.getpass('MiniMax API key: ')
            client = MiniMaxClient(api_key=key,base_url=args.base_url)
            plan = planner.with_llm(goal,client=client,model=args.model)
        else:
            plan = planner.deterministic(goal)
        _print_json(plan.to_dict())
        return

    if args.command == "gemini-test":
        # Lazy import keeps API connectivity testing independent of pandas/matplotlib/workflow imports.
        from .llm.gemini import GeminiClient

        key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not key and args.ask_api_key:
            key = getpass.getpass("Gemini API key: ")
        client = GeminiClient(api_key=key, base_url=args.base_url)
        text = client.text(
            model=args.model,
            system_prompt="You are a concise API connectivity tester.",
            user_prompt=args.prompt,
            max_tokens=256,
        )
        print(text)
        return

    cfg = load_config(args.config)

    if args.command == "run":
        from .workflow import FixedEnzymeWorkflow

        _apply_run_overrides(args, cfg)
        api_key = _resolve_api_key(args, cfg)
        report = FixedEnzymeWorkflow(
            cfg,
            force_cpu=args.cpu,
            api_key=api_key,
            print_llm=args.print_llm,
        ).run()
        _print_json(report)
    elif args.command == "validate-cpu":
        from .diagnostics import validate_three_models_cpu

        _print_json(validate_three_models_cpu(cfg))
    elif args.command == "check-gpu":
        from .diagnostics import gpu_preflight

        result = gpu_preflight(cfg)
        _print_json(result)
        if not all(v.get("ok", False) for v in result.values()):
            raise SystemExit(2)
    elif args.command == "check-paths":
        from .diagnostics import check_paths

        errors = check_paths(cfg)
        _print_json({"ok": not errors, "errors": errors})
        if errors:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
