"""External data proxies. The frontend never talks to third-party APIs directly;
all requests go through this module with caching and license metadata.

Sources:
  - UniProt REST (https://rest.uniprot.org) — search + FASTA, CC BY 4.0
  - NCBI eutils — fallback search
  - AlphaFold DB (https://alphafold.ebi.ac.uk/api) — predicted mmCIF, CC BY 4.0
  - RCSB PDB (https://files.rcsb.org) — experimental mmCIF, CC0
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import httpx

USER_AGENT = "Enzora/0.2 (enzyme design service)"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{accession}"
NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

_DEFAULT_TIMEOUT = 30.0


def _cached_json(cache_dir: Path, cache_key: str, ttl: float, fetch: Callable[[], Any]) -> Any:
    path = cache_dir / f"{cache_key}.json"
    if path.is_file() and time.time() - path.stat().st_mtime < ttl:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def _request(method: str, url: str, *, params: dict[str, Any] | None = None,
             json: Any = None, timeout: float = _DEFAULT_TIMEOUT) -> httpx.Response:
    """Proxy-aware request with one direct-connection retry.

    Some local proxies (common on China networks) break TLS to certain hosts
    (observed: NCBI eutils SSL EOF through the proxy while direct works).
    Try the environment proxy first (attempt 0), then retry directly (attempt 1).
    """
    last_exc: httpx.HTTPError | None = None
    for attempt in range(2):
        try:
            with httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
                trust_env=(attempt == 0),
            ) as client:
                response = client.request(method, url, params=params, json=json)
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            last_exc = exc
    assert last_exc is not None
    raise last_exc


def _get(url: str, *, params: dict[str, Any] | None = None, timeout: float = _DEFAULT_TIMEOUT) -> httpx.Response:
    return _request("GET", url, params=params, timeout=timeout)


def _post(url: str, *, json: Any = None, timeout: float = _DEFAULT_TIMEOUT) -> httpx.Response:
    return _request("POST", url, json=json, timeout=timeout)


def uniprot_search(query: str, cache_dir: Path, *, limit: int = 10) -> list[dict[str, Any]]:
    """Search UniProtKB by name/alias/EC number/gene; NCBI eutils fallback."""
    query = str(query).strip()

    def fetch() -> list[dict[str, Any]]:
        fields = "accession,protein_name,gene_names,organism_name,lineage,length,ec,reviewed"
        try:
            response = _get(
                UNIPROT_SEARCH_URL,
                params={"query": query, "format": "json", "size": limit, "fields": fields},
            )
            payload = response.json()
            results = []
            for item in payload.get("results", []):
                desc = item.get("proteinDescription", {}) or {}
                rec = desc.get("recommendedName") or {}
                if not rec and desc.get("submittedNames"):
                    rec = (desc.get("submittedNames") or [{}])[0] or {}
                genes = item.get("genes", []) or []
                results.append({
                    "source": "uniprot",
                    "accession": item.get("primaryAccession"),
                    "name": (rec.get("fullName", {}) or {}).get("value"),
                    "gene": genes[0].get("geneName", {}).get("value") if genes else None,
                    "organism": (item.get("organism", {}) or {}).get("scientificName"),
                    "taxid": (item.get("organism", {}) or {}).get("taxonId"),
                    "length": (item.get("length") or {}).get("value"),
                    "ec_numbers": [e.get("value") for e in (rec.get("ecNumbers") or [])],
                    "reviewed": item.get("entryType") == "UniProtKB reviewed (Swiss-Prot)",
                    "url": f"https://www.uniprot.org/uniprotkb/{item.get('primaryAccession')}/entry",
                    "license": "CC BY 4.0",
                })
            return results
        except httpx.HTTPError:
            return _ncbi_search(query, limit)

    key = "".join(ch for ch in query.casefold() if ch.isalnum())[:40] or "query"
    return _cached_json(cache_dir / "sequence_lookup", f"search_{key}", 24 * 3600, fetch)


def _ncbi_search(query: str, limit: int) -> list[dict[str, Any]]:
    try:
        esearch = _get(NCBI_ESEARCH_URL, params={
            "db": "protein", "term": query, "retmode": "json", "retmax": limit,
        }).json()
        ids = esearch.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        esummary = _get(NCBI_ESUMMARY_URL, params={
            "db": "protein", "id": ",".join(ids), "retmode": "json",
        }).json()
        results = []
        for uid in esummary.get("result", {}).get("uids", []):
            info = esummary["result"].get(uid, {})
            results.append({
                "source": "ncbi",
                "accession": info.get("accessionversion") or info.get("caption"),
                "name": info.get("title", "").split(" [")[0][:200],
                "gene": None,
                "organism": info.get("organism"),
                "taxid": info.get("taxid"),
                "length": info.get("slen"),
                "ec_numbers": [],
                "reviewed": False,
                "url": f"https://www.ncbi.nlm.nih.gov/protein/{info.get('accessionversion') or uid}",
                "license": "public domain (NCBI)",
            })
        return results
    except httpx.HTTPError:
        return []


def uniprot_fetch_sequence(accession: str, cache_dir: Path) -> dict[str, Any]:
    """Fetch FASTA + metadata for one accession (metadata from cached search when possible)."""
    accession = str(accession).strip()

    def fetch() -> dict[str, Any]:
        response = _get(f"{UNIPROT_ENTRY_URL.format(accession=accession)}.fasta")
        text = response.text
        lines = text.strip().splitlines()
        header = lines[0] if lines else ""
        sequence = "".join(lines[1:]).replace(" ", "")
        # header format: >sp|ACCESSION|NAME OS=Organism OX=TaxID GN=Gene PE=...
        parts = header[1:].split("|") if header.startswith(">") else ["", accession, header]
        name = parts[2].split(" OS=")[0].strip() if len(parts) > 2 else None
        try:
            organism = header.split("OS=")[1].split(" OX=")[0].strip()
        except IndexError:
            organism = None
        try:
            taxid = int(header.split("OX=")[1].split(" ")[0].strip())
        except (IndexError, ValueError):
            taxid = None
        return {
            "accession": accession,
            "name": name,
            "organism": organism,
            "taxid": taxid,
            "sequence": sequence,
            "url": f"https://www.uniprot.org/uniprotkb/{accession}/entry",
            "license": "CC BY 4.0",
        }

    key = "".join(ch for ch in accession.casefold() if ch.isalnum())[:40]
    return _cached_json(cache_dir / "sequence_lookup", f"entry_{key}", 30 * 24 * 3600, fetch)
