"""Use the scientific scoring implementation for all interactive re-ranking."""
from __future__ import annotations

from dataclasses import fields
from typing import Any

from Enzora_Agent.core.screening import rank_candidates
from Enzora_Agent.core.types import Candidate


def rerank_candidates(rows: list[dict[str, Any]], selection: dict) -> list[dict[str, Any]]:
    numeric = {'activity_log10', 'activity_sigma_log10', 'activity_kcat', 'library_score', 'ph_opt', 'ddg', 'delta_ph_from_wt', 'ph_uncertainty', 'stability_uncertainty'}
    allowed = {field.name for field in fields(Candidate)}
    candidates = []
    for row in rows:
        item = {k: v for k, v in row.items() if k in allowed}
        for key in numeric & item.keys():
            item[key] = None if item[key] in ('', None) else float(item[key])
        candidates.append(Candidate(**item))
    return [c.to_dict() for c in rank_candidates(candidates, selection)]
