from __future__ import annotations

import copy
from typing import Any

from ..config import validate_candidate_and_selection
from .gemini import GeminiClient, MiniMaxClient


SYSTEM_PROMPT = r"""
你是 FixedEnzymeAgent 的“参数解析器”。
你的唯一任务是把用户的蛋白质工程自然语言需求转换为严格 JSON。

重要边界：
1. 你只能建议 candidate 与 selection 下允许修改的白名单参数。
2. 你绝不能修改 sequence、output_dir、模型路径、checkpoint、Python 环境、GPU、cache、timeout 或 API key。
3. activity_log10 is PHENOM predicted log10(kcat / s^-1) for the supplied ligand. It is not a measured activity; sigma_log10 is nominal model uncertainty.
4. UniStab 的 ddG 好坏方向由 stability_direction 决定；用户未明确时返回 null，不要猜。
5. 用户没有明确给出的参数请返回 null，由程序保留原配置。
6. allowed_positions / excluded_positions 使用 1-based 残基位置整数数组。
7. weights 是偏好权重，每项在 0–1 之间且三项之和必须等于 1；将用户的比例换算成符合条件的小数。
8. 密码子优化与一个或多个目标表达物种由程序独立解析，不属于 candidate/selection；不要把“密码子优化”、宿主名称、拉丁学名或 TaxID 映射成筛选参数。
9. candidate.top_k 是程序内部候选池大小，禁止根据用户自然语言修改，必须始终返回 null。
10. 用户说“返回 N 个”“最终返回 N 个”“给我 N 个候选”“Top N”“最终选 N 个”等，N 只能写入 selection.top_k。
11. candidate.top_k 和 selection.top_k 含义完全不同：
    - candidate.top_k：内部候选池大小，由程序配置决定，不允许 LLM 修改。
    - selection.top_k：最终返回给用户的候选数量，可以根据用户要求修改。
12. 只能输出 JSON，不要输出 Markdown、解释段落或代码围栏。

必须输出如下 JSON 结构：
{
  "candidate": {
    "top_k": null,
    "allowed_positions": null,
    "excluded_positions": null
  },
  "selection": {
    "target_ph": null,
    "ph_tolerance": null,
    "min_activity_log10": null,
    "stability_direction": null,
    "stability_operator": null,
    "stability_threshold": null,
    "weights": {
      "ph": null,
      "activity": null,
      "stability": null
    },
    "top_k": null
  },
  "interpretation": "用一句中文概括你如何理解用户目标"
}
""".strip()


def _merge_non_null(
    dst: dict[str, Any],
    src: dict[str, Any],
    keys: list[str],
) -> None:
    for key in keys:
        value = src.get(key)
        if value is not None:
            dst[key] = value


def parse_user_request(
    cfg: dict[str, Any],
    *,
    client: GeminiClient | MiniMaxClient,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = str(cfg.get("user_request") or "").strip()
    if not request:
        raise ValueError("已启用 LLM 参数解析，但没有 user_request/--request")

    llm_cfg = cfg["llm"]

    # candidate.top_k 只作为上下文告诉 LLM 当前内部候选池大小，
    # 但 LLM 无权修改它。
    current = {
        "candidate": cfg["candidate"],
        "selection": cfg["selection"],
    }

    parsed = client.json_object(
        model=llm_cfg["parameter_model"],
        system_prompt=SYSTEM_PROMPT,
        user_prompt=(
            "用户需求如下：\n"
            f"{request}\n\n"
            "当前参数如下。\n"
            "注意：candidate.top_k 是内部候选池大小，只供参考，禁止修改；"
            "用户要求的最终候选数量只能写入 selection.top_k。\n"
            "仅在用户明确要求时覆盖其他允许参数。请输出 JSON：\n"
            f"{current}"
        ),
        max_tokens=int(llm_cfg.get("max_tokens", 4000)),
    )

    resolved = copy.deepcopy(cfg)

    candidate_src = parsed.get("candidate") or {}
    selection_src = parsed.get("selection") or {}

    if not isinstance(candidate_src, dict) or not isinstance(selection_src, dict):
        raise ValueError("LLM 参数 JSON 中 candidate/selection 必须是 object")

    # ------------------------------------------------------------
    # 强制保护 candidate.top_k
    # ------------------------------------------------------------
    # 即使 LLM 错误返回 candidate.top_k，也永远忽略。
    candidate_src["top_k"] = None

    # 同时修改 parsed，保证 --print-llm 输出不会再显示错误的
    # candidate.top_k=3。
    if not isinstance(parsed.get("candidate"), dict):
        parsed["candidate"] = {}
    parsed["candidate"]["top_k"] = None

    # candidate 中只允许 LLM 修改位置约束。
    _merge_non_null(
        resolved["candidate"],
        candidate_src,
        [
            "allowed_positions",
            "excluded_positions",
        ],
    )

    # selection.top_k 才是用户要求的“最终返回 N 个”。
    _merge_non_null(
        resolved["selection"],
        selection_src,
        [
            "target_ph",
            "ph_tolerance",
            "min_activity_log10",
            "stability_direction",
            "stability_operator",
            "stability_threshold",
            "top_k",
        ],
    )

    weights_src = selection_src.get("weights") or {}
    if not isinstance(weights_src, dict):
        raise ValueError(
            "LLM 参数 JSON 中 selection.weights 必须是 object"
        )

    for key in ["ph", "activity", "stability"]:
        value = weights_src.get(key)
        if value is not None:
            resolved["selection"]["weights"][key] = value

    validate_candidate_and_selection(resolved, strict_ph_target=True)
    weights = resolved['selection']['weights']
    total = sum(weights.values())
    for key in weights:
        weights[key] /= total

    return resolved, parsed