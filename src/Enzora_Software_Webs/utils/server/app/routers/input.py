"""Smart input routing endpoints (summary.md 5.4): classify, translate, codon-optimize."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..classify import translate_nucleotide
from ..config import load_settings
from ..input_files import parse_file, read_bioartifact, export_bioartifact
from ..security import get_key
from ..semantic_input import classify_text, validate_sequence

router = APIRouter(prefix="/api", tags=["input"])


class ClassifyBody(BaseModel):
    text: str
    session_id: str | None = None
    api_key: str | None = None
    type_override: str | None = None
    context: dict[str, Any] | None = None


class InputFileBody(BaseModel):
    filename: str
    content: str
    session_id: str | None = None
    api_key: str | None = None
    type_override: str | None = None


class TranslateBody(BaseModel):
    sequence: str
    sequence_type: str = "dna"


class CodonOptimizeBody(BaseModel):
    protein_sequence: str
    organism_keys: list[str]


@router.post("/input/classify")
@router.post("/classify", include_in_schema=False)
def classify(payload: ClassifyBody) -> dict[str, Any]:
    try:
        client, model = _input_client(payload.api_key, payload.session_id, payload.type_override)
        return classify_text(payload.text, client=client, model=model, type_override=payload.type_override,
                             context=payload.context)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="MiniMax 输入识别失败，请检查服务端 API 配置、网络及模型名称后重试。") from exc


def _input_client(api_key: str | None, session_id: str | None, type_override: str | None = None):
    settings = load_settings()
    model = settings["llm"].get("parameter_model", "MiniMax-M2.5")
    key = (api_key or "").strip() or get_key(session_id)
    if not key or type_override:
        return None, model
    class LazyClient:
        def json_object(self, **kwargs):
            from .llm import _client
            return _client(key).json_object(**kwargs)

    return LazyClient(), model


@router.post("/input/file")
def input_file(payload: InputFileBody) -> dict[str, Any]:
    try:
        client, model = _input_client(payload.api_key, payload.session_id, payload.type_override)
        return parse_file(payload.filename, payload.content, client=client, model=model,
                          type_override=payload.type_override)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="文件识别失败，请核对格式、输入类型和 MiniMax 配置后重试。") from exc


@router.post("/translate")
def translate(payload: TranslateBody) -> dict[str, Any]:
    try:
        sequence = validate_sequence(payload.sequence, payload.sequence_type)
        if payload.sequence_type not in {"dna", "rna"}:
            raise ValueError("翻译仅支持 DNA 或 RNA。")
        return translate_nucleotide(sequence)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/codon-optimize")
def codon_optimize(payload: CodonOptimizeBody) -> dict[str, Any]:
    """Branch A of the input router: direct protein -> host-preferred CDS.

    Reuses the package's public offline function; no models involved.
    """
    from Enzora_Agent.outputs.codon_optimization import HOSTS, optimize_protein_to_cds, supported_hosts

    known = {host["key"] for host in supported_hosts()}
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for key in payload.organism_keys:
        if key not in known:
            errors.append({"organism_key": key, "error": "unknown built-in host"})
            continue
        try:
            optimized = optimize_protein_to_cds(payload.protein_sequence, key)
        except ValueError as exc:
            errors.append({"organism_key": key, "error": str(exc)})
            continue
        host = HOSTS[key]
        results.append({
            "organism_key": key,
            "scientific_name": host.scientific_name,
            "display_name_zh": host.display_name_zh,
            "taxid": host.taxid,
            "codon_source": "builtin_kazusa",
            **{k: optimized[k] for k in (
                "optimized_cds", "optimized_cds_with_stop", "stop_codon",
                "protein_length", "cds_length", "gc_fraction")},
        })
    return {"results": results, "errors": errors, "method": "deterministic_most_frequent_synonymous_codon"}


class ArtifactExportBody(BaseModel):
    bioartifact: dict[str, Any]
    format: str


@router.post('/input/export')
def export_input(payload: ArtifactExportBody) -> dict[str, Any]:
    from Enzora_Agent.core.types import BioArtifact
    try:
        artifact = BioArtifact.from_dict(payload.bioartifact)
        content, warnings = export_bioartifact(artifact, payload.format)
        return {'content': content, 'format': payload.format, 'warnings': warnings}
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
