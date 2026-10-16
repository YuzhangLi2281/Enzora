from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


_TEXT = {
    "zh": {
        "title": "Enzora 分析报告",
        "goal": "用户目标",
        "no_goal": "未提供自然语言目标；使用配置文件中的筛选参数。",
        "selected": "最终候选",
        "no_table": "无法读取最终候选表。",
        "no_candidates": "没有候选通过当前筛选条件。",
        "visuals": "可视化结果",
        "cards": "候选结果卡",
        "gemini": "MiniMax 结果分析",
        "no_analysis": "未生成 MiniMax 分析。",
        "files": "结果文件",
        "limits": "解释边界",
        "rank": "排名",
        "pareto": "Pareto 最优",
        "yes": "是",
        "no": "否",
        "mutation": "突变",
        "position": "位置",
        "final_score": "综合分数",
        "activity": "PHENOM activity_log10",
        "ph": "EpHod 最适 pH",
        "ddg": "UniStab ddG",
        "blosum": "BLOSUM62",
        "hydro": "疏水性变化 Δ",
        "volume": "残基体积变化 Δ (Å³)",
        "charge": "名义侧链电荷变化 Δ",
        "class": "残基类别变化",
        "conservative": "BLOSUM62 非负替换",
        "reason": "过滤原因",
        "none": "无",
        "codon": "密码子优化",
        "codon_not_requested": "本次未请求密码子优化。",
    },
    "en": {
        "title": "Enzora Analysis Report",
        "goal": "User goal",
        "no_goal": "No natural-language goal was provided; selection parameters came from the config file.",
        "selected": "Selected candidates",
        "no_table": "Unable to read the selected-candidate table.",
        "no_candidates": "No candidate passed the current filters.",
        "visuals": "Visualizations",
        "cards": "Candidate cards",
        "gemini": "MiniMax result analysis",
        "no_analysis": "No MiniMax analysis was generated.",
        "files": "Output files",
        "limits": "Interpretation boundaries",
        "rank": "Rank",
        "pareto": "Pareto optimal",
        "yes": "Yes",
        "no": "No",
        "mutation": "Mutation",
        "position": "Position",
        "final_score": "Final score",
        "activity": "PHENOM activity_log10",
        "ph": "EpHod optimum pH",
        "ddg": "UniStab ddG",
        "blosum": "BLOSUM62",
        "hydro": "Hydrophobicity Δ",
        "volume": "Residue volume Δ (Å³)",
        "charge": "Nominal side-chain charge Δ",
        "class": "Residue-class change",
        "conservative": "Non-negative BLOSUM62 substitution",
        "reason": "Filter reason",
        "none": "None",
        "codon": "Codon optimization",
        "codon_not_requested": "Codon optimization was not requested for this run.",
    },
}

_FIGURE_CAPTIONS = {
    "zh": {
        "01_final_score": "最终候选综合评分",
        "02_activity_vs_stability": "活性代理与稳定性预测关系",
        "03_property_heatmap": "候选多指标热图",
        "04_mutation_sequence_map": "突变位点序列分布图",
        "05_pareto_front": "三目标 Pareto 前沿",
    },
    "en": {
        "01_final_score": "Final scores of selected candidates",
        "02_activity_vs_stability": "PHENOM log10 kcat versus stability prediction",
        "03_property_heatmap": "Multi-property heatmap",
        "04_mutation_sequence_map": "Mutation sequence map",
        "05_pareto_front": "Three-objective Pareto front",
    },
}


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _bool_text(value, language: str) -> str:
    text = _TEXT[language]
    if isinstance(value, str):
        flag = value.strip().lower() in {"true", "1", "yes"}
    else:
        flag = bool(value)
    return text["yes"] if flag else text["no"]


def _selected_table(selected_csv: Path, language: str) -> str:
    text = _TEXT[language]
    try:
        df = pd.read_csv(selected_csv)
    except Exception:
        return text["no_table"]
    if df.empty:
        return text["no_candidates"]

    columns = [
        ("mutation", text["mutation"]),
        ("activity_log10", "activity_log10"),
        ("ph_opt", "pH opt"),
        ("ddg", "ddG"),
        ("blosum62", "BLOSUM62"),
        ("pareto_optimal", text["pareto"]),
        ("final_score", text["final_score"]),
    ]
    columns = [(key, label) for key, label in columns if key in df.columns]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows: list[str] = []
    for _, row in df.iterrows():
        values: list[str] = []
        for key, _ in columns:
            if key == "pareto_optimal":
                values.append(_bool_text(row[key], language))
            elif key in {"activity_log10", "ph_opt", "ddg", "final_score"}:
                values.append(_fmt(row[key]))
            else:
                values.append(str(row[key]))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *rows])


def _candidate_cards(selected_csv: Path, language: str) -> str:
    text = _TEXT[language]
    try:
        df = pd.read_csv(selected_csv)
    except Exception:
        return text["no_table"]
    if df.empty:
        return text["no_candidates"]

    blocks: list[str] = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        mutation = str(row.get("mutation", f"candidate-{rank}"))
        pareto = _bool_text(row.get("pareto_optimal", False), language)
        blocks.extend([
            f"### #{rank} {mutation}",
            "",
            f"**{text['pareto']}**: {pareto}",
            "",
            f"| {text['mutation']} metric | Value |",
            "| --- | ---: |",
            f"| {text['position']} | {row.get('position', '')} |",
            f"| {text['final_score']} | {_fmt(row.get('final_score'))} |",
            f"| {text['activity']} | {_fmt(row.get('activity_log10'))} |",
            f"| {text['ph']} | {_fmt(row.get('ph_opt'))} |",
            f"| {text['ddg']} | {_fmt(row.get('ddg'))} |",
            f"| {text['blosum']} | {row.get('blosum62', '')} |",
            f"| {text['hydro']} | {_fmt(row.get('delta_hydrophobicity'))} |",
            f"| {text['volume']} | {_fmt(row.get('delta_volume'))} |",
            f"| {text['charge']} | {_fmt(row.get('delta_charge'))} |",
            f"| {text['class']} | {row.get('residue_class_change', '')} |",
            f"| {text['conservative']} | {_bool_text(row.get('conservative_substitution', False), language)} |",
            f"| {text['reason']} | {row.get('reasons', '') or text['none']} |",
            "",
        ])
    return "\n".join(blocks)


def build_markdown_report(
    *,
    run_dir: Path,
    analysis_text: str,
    figure_paths: list[Path],
    selected_csv: Path,
    user_request: str | None,
    language: str = "zh",
    codon_optimization: dict | None = None,
) -> Path:
    if language not in _TEXT:
        raise ValueError("report language must be 'zh' or 'en'")
    text = _TEXT[language]
    captions = _FIGURE_CAPTIONS[language]

    lines = [
        f"# {text['title']}",
        "",
        f"## {text['goal']}",
        "",
        user_request.strip() if user_request and user_request.strip() else text["no_goal"],
        "",
    ]
    balance_path = run_dir / 'balancing.json'
    if balance_path.exists():
        balance = json.loads(balance_path.read_text(encoding='utf-8'))
        lines.extend([
            '## 实际评分配比' if language == 'zh' else '## Applied score balance', '',
            '| Model | Preference | Confidence control | Effective weight |',
            '| --- | ---: | ---: | ---: |',
            *[f"| {model} | {balance['preferences'][key]:.4f} | {balance['confidence'][key]:.4f} | {balance['effective_weights'][key]:.6f} |"
              for key, model in [('activity', 'Phenom'), ('ph', 'EpHod'), ('stability', 'UniStab')]],
            '',
            '综合分使用最后一列的归一化权重；可信度控制不是校准概率。' if language == 'zh'
            else 'Composite scores use the normalized effective weights in the final column. Confidence controls are not calibrated probabilities.',
            '',
        ])
    lines.extend([
        f"## {text['selected']}",
        "",
        _selected_table(selected_csv, language),
        "",
        f"## {text['cards']}",
        "",
        _candidate_cards(selected_csv, language),
        "",
        f"## {text['visuals']}",
        "",
    ])

    for image in figure_paths:
        relative = image.relative_to(run_dir).as_posix()
        caption = captions.get(image.stem, image.stem)
        lines.extend([
            f"### {caption}",
            "",
            f"![{caption}]({relative})",
            "",
        ])

    lines.extend([f"## {text['codon']}", ""])
    codon = codon_optimization or {}
    if codon.get("enabled"):
        hosts = codon.get("resolved_hosts", []) or []
        failed_hosts = codon.get("failed_hosts", []) or []
        if language == "zh":
            lines.extend([
                f"- 状态：`{codon.get('status', 'unknown')}`",
                f"- 成功宿主数：**{len(hosts)}**；每个宿主输出 **{codon.get('sequences_per_host', 0)}** 条蛋白对应 CDS。",
                "- 方法：按目标宿主密码子使用表，为每个氨基酸确定性选择最高频同义密码子；属于基础密码子偏好优化。",
                "- 11 个常用宿主使用内置 Kazusa 来源表；其他拉丁学名/TaxID 通过 Kazusa 动态查询并缓存。",
                "- FASTA 输出包含终止密码子；CSV 同时提供不含和包含终止密码子的 CDS。",
            ])
            if hosts:
                lines.extend(["", "目标宿主："])
                for host in hosts:
                    label = host.get("display_name_zh") or host.get("scientific_name")
                    lines.append(
                        f"- **{label}** (`{host.get('scientific_name')}`, NCBI TaxID {host.get('taxid')})；来源 `{host.get('codon_source')}`"
                    )
            for warning in codon.get("warnings", []) or []:
                lines.append(f"- 注意：{warning}")
            for failed in failed_hosts:
                lines.append(f"- 未完成宿主 `{failed.get('query')}`：{failed.get('error')}")
        else:
            lines.extend([
                f"- Status: `{codon.get('status', 'unknown')}`",
                f"- Successfully resolved hosts: **{len(hosts)}**; **{codon.get('sequences_per_host', 0)}** protein-derived CDS records per host.",
                "- Method: deterministic back-translation using the most frequent synonymous codon for each amino acid in the target host codon-usage table.",
                "- Eleven common hosts use built-in Kazusa-derived tables; other Latin scientific names/TaxIDs are queried dynamically from Kazusa and cached.",
                "- FASTA outputs include a stop codon; the CSV contains CDS versions both with and without the stop codon.",
            ])
            if hosts:
                lines.extend(["", "Expression hosts:"])
                for host in hosts:
                    lines.append(
                        f"- **{host.get('scientific_name')}** (NCBI TaxID {host.get('taxid')}); source `{host.get('codon_source')}`"
                    )
            for warning in codon.get("warnings", []) or []:
                lines.append(f"- Note: {warning}")
            for failed in failed_hosts:
                lines.append(f"- Unresolved host `{failed.get('query')}`: {failed.get('error')}")
        lines.append("")
    else:
        lines.extend([text["codon_not_requested"], ""])

    lines.extend([
        f"## {text['gemini']}",
        "",
        analysis_text.strip() if analysis_text.strip() else text["no_analysis"],
        "",
        f"## {text['files']}",
        "",
    ])

    if language == "zh":
        lines.extend([
            "- `00_resolved_request.json`：自然语言需求与最终生效参数",
            "- `01_mutation_library.csv`: VESM mutation-library manifest",
            "- `02_candidate_pool_ranked.csv`：候选池综合排序与点突变属性",
            "- `03_selected_topk.csv`：最终 Top-K",
            "- `04_mutation_features.csv`：理化性质、BLOSUM62 与 Pareto 标记",
            "- `05_codon_optimized_sequences.csv/.fasta`：多宿主合并输出；仅在明确请求密码子优化时生成",
            "- `analysis.txt`：MiniMax 纯文本分析",
            "- `figures/`：统计图、序列突变图与 Pareto 图",
            "- `result.json`：结构化运行报告",
            "",
            f"## {text['limits']}",
            "",
            "- `activity_log10`: PHENOM predicted log10(kcat / s^-1); sigma_log10 is nominal uncertainty, not a calibrated interval.",
            "- EpHod pH 与 UniStab ddG 均为计算预测。",
            "- BLOSUM62 与理化性质变化只用于描述替换本身，不能单独证明突变有益。",
            "- Pareto 最优表示候选在 activity/pH/stability 三个归一化目标上未被其他合格候选同时支配。",
            "- `final_score` 取决于当前用户权重与过滤规则，不代表生物学真值。",
            "- 当前密码子优化仅基于宿主高频同义密码子；动态 Kazusa 兜底仅支持标准遗传密码表 1，且不处理限制性内切酶位点、重复序列、mRNA 二级结构等合成约束。",
            "- 最终候选必须经过实验验证。",
            "",
        ])
    else:
        lines.extend([
            "- `00_resolved_request.json`: natural-language request and resolved parameters",
            "- `01_mutation_library.csv`: VESM mutation-library manifest",
            "- `02_candidate_pool_ranked.csv`: ranked pool with point-mutation descriptors",
            "- `03_selected_topk.csv`: final Top-K",
            "- `04_mutation_features.csv`: physicochemical changes, BLOSUM62 and Pareto flags",
            "- `05_codon_optimized_sequences.csv/.fasta`: combined multi-host output generated only when codon optimization is explicitly requested",
            "- `analysis.txt`: plain-text MiniMax analysis",
            "- `figures/`: score plots, mutation sequence map and Pareto visualization",
            "- `result.json`: structured run report",
            "",
            f"## {text['limits']}",
            "",
            "- `activity_log10` is PHENOM predicted log10(kcat / s^-1), not measured enzyme activity.",
            "- EpHod optimum pH and UniStab ddG are computational predictions.",
            "- BLOSUM62 and physicochemical deltas describe the substitution and do not by themselves prove benefit.",
            "- Pareto optimal means the candidate is not simultaneously dominated by another passing candidate across normalized activity/pH/stability objectives.",
            "- `final_score` depends on the active filters and user weights and is not biological ground truth.",
            "- Current codon optimization uses host-preferred synonymous codons only; dynamic Kazusa fallback assumes Standard genetic code 1 and does not optimize restriction sites, repeats, mRNA secondary structure, or other synthesis constraints.",
            "- Final candidates require experimental validation.",
            "",
        ])

    filename = "report_zh.md" if language == "zh" else "report_en.md"
    path = run_dir / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
