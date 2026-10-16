"""External sequence lookup (summary.md 5.1): enzyme name -> candidates -> sequence."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..security import get_key

from ..config import load_settings
from ..proxy import uniprot_fetch_sequence, uniprot_search

router = APIRouter(prefix="/api", tags=["lookup"])

# Deterministic Chinese -> English alias table for common enzymes. UniProt and
# NCBI do not index Chinese names, so these mappings run before the search.
# Additional Chinese queries can be normalized by Gemini when a session key
# is present (optional, never required).
_ZH_EN_ALIASES: dict[str, str] = {
    "碳酸酐酶": "carbonic anhydrase",
    "溶菌酶": "lysozyme",
    "脂肪酶": "lipase",
    "蛋白酶": "protease",
    "淀粉酶": "amylase",
    "纤维素酶": "cellulase",
    "过氧化氢酶": "catalase",
    "过氧化物酶": "peroxidase",
    "超氧化物歧化酶": "superoxide dismutase",
    "葡萄糖氧化酶": "glucose oxidase",
    "漆酶": "laccase",
    "酪氨酸酶": "tyrosinase",
    "胆固醇氧化酶": "cholesterol oxidase",
    "木聚糖酶": "xylanase",
    "植酸酶": "phytase",
    "果胶酶": "pectinase",
    "琼脂糖酶": "agarase",
    "酯酶": "esterase",
    "转氨酶": "aminotransferase",
    "脱氢酶": "dehydrogenase",
    "聚合酶": "polymerase",
    "半乳糖苷酶": "galactosidase",
    "限制性内切酶": "restriction endonuclease",
    "核糖核酸酶": "ribonuclease",
    "肌酸激酶": "creatine kinase",
    "环糊精糖基转移酶": "cyclodextrin glycosyltransferase",
}

_HAN_RE = re.compile(r"[一-鿿]")


class LookupBody(BaseModel):
    query: str
    session_id: str | None = None
    api_key: str | None = None


class ResolveBody(BaseModel):
    accession: str
    source: str = "uniprot"


def _cache_dir() -> Path:
    return Path(load_settings()["runtime"]["cache_dir"])


def _normalize_query(query: str, api_key: str | None) -> tuple[str, list[str]]:
    """Chinese alias substitution + optional LLM normalization for other Han text."""
    notes: list[str] = []
    for zh, en in _ZH_EN_ALIASES.items():
        if zh in query:
            query = query.replace(zh, en)
            notes.append(f"translated '{zh}' -> '{en}'")
    if _HAN_RE.search(query):
        notes.append("query still contains Chinese characters (external databases index English names)")
        if api_key:
            try:
                from Enzora_Agent.llm.gemini import GeminiClient

                client = GeminiClient(
                    api_key=api_key,
                    base_url=load_settings()["llm"].get("base_url", "https://generativelanguage.googleapis.com/v1beta"),
                )
                translated = client.text(
                    model=load_settings()["llm"]["parameter_model"],
                    system_prompt=(
                        "You translate enzyme names from Chinese to English. "
                        "Return ONLY the English protein name or EC number, nothing else."
                    ),
                    user_prompt=query,
                    max_tokens=64,
                        ).strip()
                if translated:
                    query = translated
                    notes.append(f"LLM-normalized query -> '{translated}'")
            except Exception as exc:
                notes.append(f"LLM normalization unavailable: {type(exc).__name__}")
    return query, notes


@router.post("/lookup")
def lookup(payload: LookupBody) -> dict[str, Any]:
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    query, notes = _normalize_query(payload.query.strip(), payload.api_key or get_key(payload.session_id))
    try:
        candidates = uniprot_search(query, _cache_dir())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"external search failed: {type(exc).__name__}: {exc}") from exc
    warning = None
    if not candidates:
        warning = "no sequence found; try an English name, UniProt ID, or EC number"
    result: dict[str, Any] = {"candidates": candidates}
    if warning:
        result["warning"] = warning
    if notes:
        result["notes"] = notes
    return result


@router.post("/lookup/resolve")
def lookup_resolve(payload: ResolveBody) -> dict[str, Any]:
    if payload.source == "uniprot":
        try:
            data = uniprot_fetch_sequence(payload.accession, _cache_dir())
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"fetch failed: {type(exc).__name__}: {exc}") from exc
    else:
        try:
            # NCBI fallback: efetch protein FASTA via eutils (proxy-aware).
            from ..proxy import _get

            response = _get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "protein", "id": payload.accession, "rettype": "fasta", "retmode": "text"},
                timeout=30,
            )
            lines = response.text.strip().splitlines()
            header = lines[0] if lines else ""
            data = {
                "accession": payload.accession,
                "name": header[1:].split(" [")[0] if header.startswith(">") else None,
                "organism": None,
                "taxid": None,
                "sequence": "".join(lines[1:]),
                "url": f"https://www.ncbi.nlm.nih.gov/protein/{payload.accession}",
                "license": "public domain (NCBI)",
            }
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"NCBI fetch failed: {type(exc).__name__}: {exc}") from exc
    return data
