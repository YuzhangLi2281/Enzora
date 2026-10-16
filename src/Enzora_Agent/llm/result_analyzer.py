from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .gemini import GeminiClient


_SYSTEM_PROMPTS = {
    "zh": r"""
你是 FixedEnzymeAgent 的结果分析模块。请只根据用户给出的结构化预测和确定性点突变描述进行分析，不得发明任何数值。

模型与描述符含义：
- PHENOM: activity_log10 is predicted log10(kcat / s^-1) for the supplied ligand. sigma_log10 is nominal uncertainty, not a calibrated confidence interval.
- EpHod: predicted optimum pH，是计算预测。
- UniStab: predicted ddG，是计算预测；好坏方向必须依据 stability_direction。
- BLOSUM62、疏水性/体积/电荷变化：仅描述氨基酸替换本身，不能单独证明突变有益或有害。
- pareto_optimal：表示该候选在归一化 activity/pH/stability 三目标上未被另一个合格候选同时支配。

必须遵守：
1. 不把预测描述成实验验证结果。
2. 不修改、补造或反推输入里没有的数据。
3. 分析 activity_log10、pH、ddG、BLOSUM62、理化性质变化和 Pareto 状态之间的互补信息与 trade-off。
4. 解释 final_score 只是当前用户权重下的综合排序，不是生物学真值。
5. 推荐实验验证优先级，并说明最终候选仍需湿实验验证。
6. 使用中文纯文本；不要输出 Markdown 代码围栏。

建议结构：整体结论；优先候选；理化与 BLOSUM62 解读；多目标/Pareto 权衡；实验验证顺序；局限性。
""".strip(),
    "en": r"""
You are the result-analysis module of FixedEnzymeAgent. Analyze only the supplied structured predictions and deterministic point-mutation descriptors. Never invent values.

Meaning of the inputs:
- PHENOM activity_log10 is predicted log10(kcat / s^-1), with sigma_log10 as nominal model uncertainty; it is not experimental activity.
- EpHod predicted optimum pH is computational.
- UniStab predicted ddG is computational; interpret direction only according to stability_direction.
- BLOSUM62 and hydrophobicity/volume/charge deltas only describe the amino-acid substitution and do not prove benefit or harm by themselves.
- pareto_optimal means the candidate is not simultaneously dominated by another passing candidate across normalized activity/pH/stability objectives.

Rules:
1. Do not describe predictions as experimental validation.
2. Do not modify, fabricate or infer unavailable numeric values.
3. Discuss complementary evidence and trade-offs across activity_log10, pH, ddG, BLOSUM62, physicochemical deltas and Pareto status.
4. Explain that final_score is a ranking under the current user weights, not biological ground truth.
5. Recommend an experimental validation order and state that wet-lab validation is required.
6. Write plain English text; do not output a Markdown code fence.

Suggested structure: overall conclusion; priority candidates; physicochemical/BLOSUM62 interpretation; multi-objective/Pareto trade-offs; experimental order; limitations.
""".strip(),
}


def analyze_results(
    *,
    cfg: dict,
    ranked_csv: Path,
    wt_ph: float,
    client: GeminiClient,
) -> str:
    df = pd.read_csv(ranked_csv)
    # Never send raw WT/mutant protein sequences to the third-party LLM.
    allowed_columns = [
        "mutation",
        "position",
        "activity_log10",
        "ph_opt",
        "delta_ph_from_wt",
        "ddg",
        "activity_component",
        "ph_component",
        "stability_component",
        "final_score",
        "passed",
        "reasons",
        "wt_aa",
        "mutant_aa",
        "wt_residue_class",
        "mutant_residue_class",
        "residue_class_change",
        "delta_hydrophobicity",
        "delta_volume",
        "delta_charge",
        "blosum62",
        "conservative_substitution",
        "pareto_optimal",
    ]
    existing = [c for c in allowed_columns if c in df.columns]
    limit = int(cfg.get("llm", {}).get("analysis_top_n", 20))
    rows = json.loads(df[existing].head(max(1, limit)).to_json(orient="records"))

    language = str(cfg.get("report", {}).get("language", "zh"))
    if language not in _SYSTEM_PROMPTS:
        language = "zh"

    payload = {
        "report_language": language,
        "user_request": cfg.get("user_request"),
        "selection": cfg["selection"],
        "candidate_pool_size": int(len(df)),
        "wt_predicted_optimum_ph": float(wt_ph),
        "top_candidates": rows,
    }
    llm_cfg = cfg["llm"]
    instruction = (
        "请分析下面的 FixedEnzymeAgent JSON 结果。所有数值只能引用 JSON 中已有值：\n"
        if language == "zh"
        else "Analyze the following FixedEnzymeAgent JSON. You may only quote numeric values present in the JSON:\n"
    )
    return client.text(
        model=llm_cfg.get("analysis_model", "gemini-2.5-pro"),
        system_prompt=_SYSTEM_PROMPTS[language],
        user_prompt=instruction + json.dumps(payload, ensure_ascii=False, indent=2),
        max_tokens=int(llm_cfg.get("max_tokens", 4000)),
    )
