from __future__ import annotations

import csv
import html as html_lib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..core.types import Candidate


# Common expression hosts are normalized to species-level profiles. Human cell
# lines are represented as Homo sapiens; bacteriophages are omitted because they
# are not one species. Bacillus subtilis is retained explicitly for this project.
#
# Codons below are the most frequently observed synonymous codon for each amino
# acid in the cited Kazusa codon-usage table. The optimization is therefore a
# deterministic preferred-codon back-translation, not a full synthesis optimizer.


@dataclass(frozen=True)
class HostSpec:
    key: str
    scientific_name: str
    display_name_zh: str
    taxid: str
    kazusa_species: str
    aliases: tuple[str, ...]
    preferred_codons: dict[str, str]
    preferred_stop: str


_HOSTS: tuple[HostSpec, ...] = (
    HostSpec(
        key="e_coli",
        scientific_name="Escherichia coli K-12 W3110",
        display_name_zh="大肠杆菌",
        taxid="316407",
        kazusa_species="316407",
        aliases=(
            "大肠杆菌", "escherichia coli", "e. coli", "e coli", "ecoli",
            "e.coli", "k-12", "k12", "w3110", "bl21", "dh5a", "dh5α",
        ),
        preferred_codons={
            "A": "GCG", "C": "TGC", "D": "GAT", "E": "GAA", "F": "TTT",
            "G": "GGC", "H": "CAT", "I": "ATT", "K": "AAA", "L": "CTG",
            "M": "ATG", "N": "AAC", "P": "CCG", "Q": "CAG", "R": "CGC",
            "S": "AGC", "T": "ACC", "V": "GTG", "W": "TGG", "Y": "TAT",
        },
        preferred_stop="TAA",
    ),
    HostSpec(
        key="s_cerevisiae",
        scientific_name="Saccharomyces cerevisiae",
        display_name_zh="酿酒酵母",
        taxid="4932",
        kazusa_species="4932",
        aliases=(
            "酿酒酵母", "saccharomyces cerevisiae", "s. cerevisiae",
            "s cerevisiae", "baker's yeast", "bakers yeast", "brewer's yeast",
        ),
        preferred_codons={
            "A": "GCT", "C": "TGT", "D": "GAT", "E": "GAA", "F": "TTT",
            "G": "GGT", "H": "CAT", "I": "ATT", "K": "AAA", "L": "TTG",
            "M": "ATG", "N": "AAT", "P": "CCA", "Q": "CAA", "R": "AGA",
            "S": "TCT", "T": "ACT", "V": "GTT", "W": "TGG", "Y": "TAT",
        },
        preferred_stop="TAA",
    ),
    HostSpec(
        key="h_sapiens",
        scientific_name="Homo sapiens",
        display_name_zh="人源细胞",
        taxid="9606",
        kazusa_species="9606",
        aliases=(
            "homo sapiens", "human", "human cell", "human cells", "人源",
            "人类", "人细胞", "hek293", "hek293t", "hela", "hela cells",
        ),
        preferred_codons={
            "A": "GCC", "C": "TGC", "D": "GAC", "E": "GAG", "F": "TTC",
            "G": "GGC", "H": "CAC", "I": "ATC", "K": "AAG", "L": "CTG",
            "M": "ATG", "N": "AAC", "P": "CCC", "Q": "CAG", "R": "AGA",
            "S": "AGC", "T": "ACC", "V": "GTG", "W": "TGG", "Y": "TAC",
        },
        preferred_stop="TGA",
    ),
    HostSpec(
        key="p_putida",
        scientific_name="Pseudomonas putida KT2440",
        display_name_zh="恶臭假单胞菌",
        taxid="160488",
        kazusa_species="160488",
        aliases=(
            "恶臭假单胞菌", "pseudomonas putida", "p. putida", "p putida", "kt2440",
        ),
        preferred_codons={
            "A": "GCC", "C": "TGC", "D": "GAC", "E": "GAG", "F": "TTC",
            "G": "GGC", "H": "CAC", "I": "ATC", "K": "AAG", "L": "CTG",
            "M": "ATG", "N": "AAC", "P": "CCG", "Q": "CAG", "R": "CGC",
            "S": "AGC", "T": "ACC", "V": "GTG", "W": "TGG", "Y": "TAC",
        },
        preferred_stop="TGA",
    ),
    HostSpec(
        key="k_pastoris",
        scientific_name="Komagataella pastoris (Pichia pastoris)",
        display_name_zh="毕赤酵母",
        taxid="4922",
        kazusa_species="4922",
        aliases=(
            "毕赤酵母", "pichia pastoris", "komagataella pastoris", "k. pastoris",
            "k pastoris", "p. pastoris", "p pastoris",
        ),
        preferred_codons={
            "A": "GCT", "C": "TGT", "D": "GAT", "E": "GAA", "F": "TTT",
            "G": "GGT", "H": "CAT", "I": "ATT", "K": "AAG", "L": "TTG",
            "M": "ATG", "N": "AAC", "P": "CCA", "Q": "CAA", "R": "AGA",
            "S": "TCT", "T": "ACT", "V": "GTT", "W": "TGG", "Y": "TAC",
        },
        preferred_stop="TAA",
    ),
    HostSpec(
        key="p_aeruginosa",
        scientific_name="Pseudomonas aeruginosa PAO1",
        display_name_zh="铜绿假单胞菌",
        taxid="208964",
        kazusa_species="208964",
        aliases=(
            "铜绿假单胞菌", "绿脓杆菌", "pseudomonas aeruginosa", "p. aeruginosa",
            "p aeruginosa", "pao1",
        ),
        preferred_codons={
            "A": "GCC", "C": "TGC", "D": "GAC", "E": "GAG", "F": "TTC",
            "G": "GGC", "H": "CAC", "I": "ATC", "K": "AAG", "L": "CTG",
            "M": "ATG", "N": "AAC", "P": "CCG", "Q": "CAG", "R": "CGC",
            "S": "AGC", "T": "ACC", "V": "GTG", "W": "TGG", "Y": "TAC",
        },
        preferred_stop="TGA",
    ),
    HostSpec(
        key="y_lipolytica",
        scientific_name="Yarrowia lipolytica CLIB122",
        display_name_zh="解脂耶氏酵母",
        taxid="284591",
        kazusa_species="284591",
        aliases=(
            "解脂耶氏酵母", "yarrowia lipolytica", "y. lipolytica", "y lipolytica", "clib122",
        ),
        preferred_codons={
            "A": "GCC", "C": "TGC", "D": "GAC", "E": "GAG", "F": "TTC",
            "G": "GGC", "H": "CAC", "I": "ATC", "K": "AAG", "L": "CTG",
            "M": "ATG", "N": "AAC", "P": "CCC", "Q": "CAG", "R": "CGA",
            "S": "TCT", "T": "ACC", "V": "GTG", "W": "TGG", "Y": "TAC",
        },
        preferred_stop="TAA",
    ),
    HostSpec(
        key="synechocystis_pcc6803",
        scientific_name="Synechocystis sp. PCC 6803",
        display_name_zh="集胞藻 PCC 6803",
        taxid="1148",
        kazusa_species="1148",
        aliases=(
            "集胞藻", "synechocystis", "synechocystis sp. pcc 6803", "pcc 6803", "pcc6803",
        ),
        preferred_codons={
            "A": "GCC", "C": "TGT", "D": "GAT", "E": "GAA", "F": "TTT",
            "G": "GGC", "H": "CAT", "I": "ATT", "K": "AAA", "L": "TTG",
            "M": "ATG", "N": "AAT", "P": "CCC", "Q": "CAA", "R": "CGG",
            "S": "TCC", "T": "ACC", "V": "GTG", "W": "TGG", "Y": "TAT",
        },
        preferred_stop="TAA",
    ),
    HostSpec(
        key="c_reinhardtii",
        scientific_name="Chlamydomonas reinhardtii",
        display_name_zh="莱茵衣藻",
        taxid="3055",
        kazusa_species="3055",
        aliases=(
            "莱茵衣藻", "chlamydomonas reinhardtii", "c. reinhardtii", "c reinhardtii",
        ),
        preferred_codons={
            "A": "GCC", "C": "TGC", "D": "GAC", "E": "GAG", "F": "TTC",
            "G": "GGC", "H": "CAC", "I": "ATC", "K": "AAG", "L": "CTG",
            "M": "ATG", "N": "AAC", "P": "CCC", "Q": "CAG", "R": "CGC",
            "S": "AGC", "T": "ACC", "V": "GTG", "W": "TGG", "Y": "TAC",
        },
        preferred_stop="TAA",
    ),
    HostSpec(
        key="a_thaliana",
        scientific_name="Arabidopsis thaliana",
        display_name_zh="拟南芥",
        taxid="3702",
        kazusa_species="3702",
        aliases=(
            "拟南芥", "arabidopsis thaliana", "a. thaliana", "a thaliana", "thale cress",
        ),
        preferred_codons={
            "A": "GCT", "C": "TGT", "D": "GAT", "E": "GAA", "F": "TTT",
            "G": "GGA", "H": "CAT", "I": "ATT", "K": "AAG", "L": "CTT",
            "M": "ATG", "N": "AAT", "P": "CCT", "Q": "CAA", "R": "AGA",
            "S": "TCT", "T": "ACT", "V": "GTT", "W": "TGG", "Y": "TAT",
        },
        preferred_stop="TGA",
    ),
    HostSpec(
        key="b_subtilis",
        scientific_name="Bacillus subtilis",
        display_name_zh="枯草芽孢杆菌",
        taxid="1423",
        kazusa_species="1423",
        aliases=(
            "枯草芽孢杆菌", "bacillus subtilis", "b. subtilis", "b subtilis", "wb800n",
        ),
        preferred_codons={
            "A": "GCA", "C": "TGC", "D": "GAT", "E": "GAA", "F": "TTT",
            "G": "GGC", "H": "CAT", "I": "ATT", "K": "AAA", "L": "CTG",
            "M": "ATG", "N": "AAT", "P": "CCG", "Q": "CAA", "R": "AGA",
            "S": "TCA", "T": "ACA", "V": "GTT", "W": "TGG", "Y": "TAT",
        },
        preferred_stop="TAA",
    ),
)

HOSTS = {host.key: host for host in _HOSTS}

# Public support set shown to the user. B. subtilis is listed separately because the
# project explicitly requested it as an additional host even though it is already a
# common expression chassis in the source report.
COMMON_HOST_KEYS = (
    "e_coli",
    "s_cerevisiae",
    "h_sapiens",
    "p_putida",
    "k_pastoris",
    "p_aeruginosa",
    "y_lipolytica",
    "synechocystis_pcc6803",
    "c_reinhardtii",
    "a_thaliana",
)
EXTRA_HOST_KEYS = ("b_subtilis",)

_CODON_TRIGGER = re.compile(
    r"密码子\s*(?:优化|偏好|适配)|codon\s*(?:optimization|optimisation|optimi[sz](?:e|ation|ed|ing))"
    r"|optimi[sz](?:e|ed|ing)\s+(?:the\s+)?codons?\b",
    re.IGNORECASE,
)



KAZUSA_SEARCH_URL = "https://www.kazusa.or.jp/codon/cgi-bin/spsearch.cgi"
KAZUSA_TABLE_URL = "https://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi"
DEFAULT_HOST_KEY = "e_coli"

_LATIN_BINOMIAL_RE = re.compile(r"\b([A-Z][a-z]{2,}\s+[a-z][a-z0-9_-]{2,})\b")
_TAXID_RE = re.compile(r"\b(?:NCBI\s*)?(?:TaxID|taxid|taxonomy\s*id)\s*[:=#]?\s*(\d{2,9})\b", re.IGNORECASE)
_DYNAMIC_GENUS_STOPWORDS = {
    "Please", "Codon", "Perform", "Optimize", "Optimization", "Optimise",
    "Expression", "Protein", "Sequence", "Final", "Target", "Host",
    "Organism", "Species", "Return", "Using", "Use", "Generate", "Include",
}


def supported_hosts() -> list[dict[str, str]]:
    """Return the 11 offline built-in hosts."""
    result = []
    for key in (*COMMON_HOST_KEYS, *EXTRA_HOST_KEYS):
        host = HOSTS[key]
        result.append({
            "key": host.key,
            "scientific_name": host.scientific_name,
            "display_name_zh": host.display_name_zh,
            "taxid": host.taxid,
            "kazusa_species": host.kazusa_species,
            "source": "builtin_kazusa",
        })
    return result


def _alias_pattern(alias: str) -> re.Pattern[str]:
    # Latin/common names should not match as a substring of a longer ASCII token.
    escaped = re.escape(alias)
    if re.search(r"[A-Za-z0-9]", alias):
        return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
    return re.compile(escaped, re.IGNORECASE)


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _builtin_request(host: HostSpec, *, source: str, position: int = 0) -> dict:
    return {
        "kind": "builtin",
        "query": host.scientific_name,
        "organism_key": host.key,
        "scientific_name": host.scientific_name,
        "display_name_zh": host.display_name_zh,
        "taxid": host.taxid,
        "selection_source": source,
        "position": position,
    }


def _find_builtin_requests(request: str) -> list[dict]:
    found: list[dict] = []
    for host in _HOSTS:
        positions: list[int] = []
        for alias in host.aliases:
            match = _alias_pattern(alias).search(request)
            if match:
                positions.append(match.start())
        if positions:
            found.append(_builtin_request(host, source="explicit_builtin", position=min(positions)))
    return sorted(found, key=lambda item: int(item["position"]))


def _extract_dynamic_requests(request: str, builtin_requests: list[dict]) -> list[dict]:
    builtin_taxids = {str(item["taxid"]) for item in builtin_requests}
    builtin_names: set[str] = set()
    for host in _HOSTS:
        builtin_names.add(_normalize_name(host.scientific_name))
        builtin_names.update(_normalize_name(alias) for alias in host.aliases)

    dynamic: list[dict] = []
    for match in _TAXID_RE.finditer(request):
        taxid = match.group(1)
        if taxid in builtin_taxids:
            continue
        dynamic.append({
            "kind": "kazusa_taxid",
            "query": taxid,
            "organism_key": f"kazusa_{taxid}",
            "scientific_name": None,
            "display_name_zh": None,
            "taxid": taxid,
            "selection_source": "explicit_kazusa_taxid",
            "position": match.start(),
        })

    for match in _LATIN_BINOMIAL_RE.finditer(request):
        name = re.sub(r"\s+", " ", match.group(1)).strip()
        genus = name.split()[0]
        if genus in _DYNAMIC_GENUS_STOPWORDS:
            continue
        if _normalize_name(name) in builtin_names:
            continue
        dynamic.append({
            "kind": "kazusa_name",
            "query": name,
            "organism_key": None,
            "scientific_name": name,
            "display_name_zh": None,
            "taxid": None,
            "selection_source": "explicit_kazusa_name",
            "position": match.start(),
        })

    # Dedupe before any network lookup. TaxID and name duplicates are deduped again after resolution.
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for item in sorted(dynamic, key=lambda x: int(x["position"])):
        marker = (str(item["kind"]), _normalize_name(str(item["query"])))
        if marker not in seen:
            seen.add(marker)
            unique.append(item)
    return unique


def resolve_codon_optimization_request(user_request: str | None) -> dict:
    """Parse whether codon optimization is requested and which host(s) were named.

    The parser is intentionally deterministic and offline. The 11 built-in hosts are
    resolved immediately. Other Latin scientific names or explicit NCBI TaxIDs are
    recorded as Kazusa fallback requests and resolved only when outputs are written.
    """
    request = str(user_request or "").strip()
    if not request or not _CODON_TRIGGER.search(request):
        return {
            "enabled": False,
            "hosts": [],
            "host_count": 0,
            "selection_source": "not_requested",
            "warning": None,
            # backward-compatible singular keys
            "organism_key": None,
            "scientific_name": None,
            "display_name_zh": None,
            "taxid": None,
        }

    builtins = _find_builtin_requests(request)
    dynamic = _extract_dynamic_requests(request, builtins)
    host_requests = sorted([*builtins, *dynamic], key=lambda x: int(x.get("position", 0)))

    if not host_requests:
        default = _builtin_request(HOSTS[DEFAULT_HOST_KEY], source="default_e_coli")
        host_requests = [default]
        warning = "检测到密码子优化请求，但未识别到目标宿主；按规则默认使用大肠杆菌。"
        source = "default_e_coli"
    else:
        warning = None
        source = "explicit_hosts"

    # Strip parser-only character positions from serialized output.
    hosts = [{k: v for k, v in item.items() if k != "position"} for item in host_requests]
    result = {
        "enabled": True,
        "hosts": hosts,
        "host_count": len(hosts),
        "selection_source": source,
        "warning": warning,
    }
    if len(hosts) == 1:
        result.update({
            "organism_key": hosts[0].get("organism_key"),
            "scientific_name": hosts[0].get("scientific_name"),
            "display_name_zh": hosts[0].get("display_name_zh"),
            "taxid": hosts[0].get("taxid"),
        })
    else:
        result.update({
            "organism_key": None,
            "scientific_name": None,
            "display_name_zh": None,
            "taxid": None,
        })
    return result


def _default_download_text(url: str, timeout: float) -> tuple[str, str]:
    request = Request(url, headers={"User-Agent": "FixedEnzymeAgent/0.4.1"})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
        final_url = response.geturl()
    return final_url, raw


def _strip_html(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html_lib.unescape(value)).strip()


def search_kazusa_species(
    scientific_name: str,
    *,
    timeout: float = 10.0,
    downloader: Callable[[str, float], tuple[str, str]] | None = None,
) -> dict[str, str]:
    """Resolve a Latin scientific name to a Kazusa/NCBI TaxID.

    Kazusa's organism search is regular-expression based. We therefore prefer an
    exact normalized name; if there is no exact match, a unique result is accepted.
    Ambiguous searches raise instead of silently selecting the first organism.
    """
    downloader = downloader or _default_download_text
    query = re.sub(r"\s+", " ", scientific_name.strip())
    url = f"{KAZUSA_SEARCH_URL}?{urlencode({'species': query, 'c': 'i'})}"
    final_url, raw = downloader(url, timeout)

    direct = re.search(r"showcodon\.cgi\?[^#\s]*species=(\d+)", final_url, re.IGNORECASE)
    if direct:
        return {"taxid": direct.group(1), "scientific_name": query}

    link_re = re.compile(
        r"href=[\"'][^\"']*showcodon\.cgi\?[^\"']*species=(\d+)[^\"']*[\"'][^>]*>(.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )
    hits: list[dict[str, str]] = []
    for taxid, label_html in link_re.findall(raw):
        label = _strip_html(label_html)
        name = re.sub(r"\s*\[gb[a-z]+\].*$", "", label, flags=re.IGNORECASE).strip()
        if not name:
            name = label
        hits.append({"taxid": taxid, "scientific_name": name})

    # Some server responses may expose the selected species ID in another link/form.
    if not hits:
        ids = re.findall(r"showcodon\.cgi\?[^\"'<>\s]*species=(\d+)", raw, re.IGNORECASE)
        ids = list(dict.fromkeys(ids))
        if len(ids) == 1:
            return {"taxid": ids[0], "scientific_name": query}
        raise ValueError(f"Kazusa 中未找到宿主: {query}")

    # Dedupe identical TaxIDs while preserving order.
    deduped: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for hit in hits:
        if hit["taxid"] not in seen_ids:
            seen_ids.add(hit["taxid"])
            deduped.append(hit)
    hits = deduped

    qnorm = _normalize_name(query)
    exact = [hit for hit in hits if _normalize_name(hit["scientific_name"]) == qnorm]
    if len(exact) == 1:
        return exact[0]
    if len(hits) == 1:
        return hits[0]

    preview = ", ".join(f"{h['scientific_name']} (TaxID {h['taxid']})" for h in hits[:8])
    raise ValueError(
        f"Kazusa 宿主查询存在多个匹配: {query} -> {preview}。"
        "请在请求中提供更完整的拉丁学名或直接写 NCBI TaxID。"
    )


def _cache_path(cache_dir: str | Path | None, taxid: str) -> Path | None:
    if cache_dir is None:
        return None
    return Path(cache_dir) / "codon_tables" / f"kazusa_{taxid}.json"


def fetch_kazusa_codon_profile(
    taxid: str,
    *,
    scientific_name_hint: str | None = None,
    timeout: float = 10.0,
    cache_dir: str | Path | None = None,
    downloader: Callable[[str, float], tuple[str, str]] | None = None,
) -> dict:
    """Fetch one codon table from Kazusa and convert it to preferred codons.

    The URL explicitly requests amino-acid labels with NCBI genetic code 1
    (Standard). Dynamic fallback is therefore intended for ordinary expression
    hosts using the standard code; organelles/non-standard codes should not be
    selected without an explicit future genetic-code option.
    """
    taxid = str(taxid).strip()
    if not taxid.isdigit():
        raise ValueError(f"无效 TaxID: {taxid}")

    path = _cache_path(cache_dir, taxid)
    if path and path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("taxid") == taxid and cached.get("preferred_codons"):
            cached["codon_source"] = "kazusa_cache"
            return cached

    downloader = downloader or _default_download_text
    url = f"{KAZUSA_TABLE_URL}?{urlencode({'species': taxid, 'aa': '1', 'style': 'N'})}"
    _, raw = downloader(url, timeout)
    text = _strip_html(raw)
    if "not found" in text.casefold():
        raise ValueError(f"Kazusa 中没有 TaxID {taxid} 的密码子表")

    codon_re = re.compile(
        r"\b([AUCGT]{3})\s+([A-Z*])\s+([0-9]+(?:\.[0-9]+)?)\s+"
        r"([0-9]+(?:\.[0-9]+)?)\s*\(\s*([0-9]+)\s*\)"
    )
    rows = []
    for codon, aa, fraction, per_thousand, count in codon_re.findall(text):
        rows.append({
            "codon": codon.replace("U", "T"),
            "aa": aa,
            "fraction": float(fraction),
            "per_thousand": float(per_thousand),
            "count": int(count),
        })
    if len(rows) < 60:
        raise RuntimeError(f"Kazusa TaxID {taxid} 密码子表解析失败，仅识别到 {len(rows)} 个 codon")

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["aa"]), []).append(row)
    missing = sorted(set("ACDEFGHIKLMNPQRSTVWY*") - set(grouped))
    if missing:
        raise RuntimeError(f"Kazusa TaxID {taxid} 密码子表缺少氨基酸/终止项: {missing}")

    def best(items: list[dict]) -> str:
        chosen = max(items, key=lambda item: (float(item["fraction"]), int(item["count"]), str(item["codon"])))
        return str(chosen["codon"])

    preferred_codons = {aa: best(grouped[aa]) for aa in "ACDEFGHIKLMNPQRSTVWY"}
    preferred_stop = best(grouped["*"])

    name_match = re.search(r"(.+?)\s*\[gb[a-z]+\]\s*:\s*(\d+)\s+CDS", text, re.IGNORECASE)
    if name_match:
        scientific_name = name_match.group(1).strip()
        cds_count = int(name_match.group(2))
    else:
        scientific_name = scientific_name_hint or f"NCBI TaxID {taxid}"
        cds_count = None

    code_match = re.search(r"Genetic code\s+(\d+)\s*:\s*([A-Za-z0-9 _-]+)", text, re.IGNORECASE)
    genetic_code = int(code_match.group(1)) if code_match else 1
    genetic_code_name = code_match.group(2).strip() if code_match else "Standard"
    if genetic_code != 1:
        raise ValueError(
            f"Kazusa TaxID {taxid} 返回 genetic code {genetic_code} ({genetic_code_name})；"
            "当前动态密码子优化仅支持 Standard genetic code 1。"
        )

    profile = {
        "organism_key": f"kazusa_{taxid}",
        "scientific_name": scientific_name,
        "display_name_zh": None,
        "taxid": taxid,
        "preferred_codons": preferred_codons,
        "preferred_stop": preferred_stop,
        "codon_source": "kazusa_dynamic",
        "kazusa_url": url,
        "cds_count": cds_count,
        "genetic_code": genetic_code,
        "genetic_code_name": genetic_code_name,
    }
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def resolve_host_profile(
    host_request: dict,
    *,
    timeout: float = 10.0,
    cache_dir: str | Path | None = None,
    downloader: Callable[[str, float], tuple[str, str]] | None = None,
) -> dict:
    kind = str(host_request.get("kind"))
    if kind == "builtin":
        key = str(host_request["organism_key"])
        host = HOSTS[key]
        return {
            "organism_key": host.key,
            "scientific_name": host.scientific_name,
            "display_name_zh": host.display_name_zh,
            "taxid": host.taxid,
            "preferred_codons": host.preferred_codons,
            "preferred_stop": host.preferred_stop,
            "codon_source": "builtin_kazusa",
            "kazusa_url": f"{KAZUSA_TABLE_URL}?species={host.kazusa_species}&aa=1&style=N",
            "selection_source": host_request.get("selection_source"),
            "query": host_request.get("query"),
            "genetic_code": 1,
            "genetic_code_name": "Standard",
        }

    if kind == "kazusa_name":
        hit = search_kazusa_species(
            str(host_request["query"]), timeout=timeout, downloader=downloader
        )
        taxid = hit["taxid"]
        name_hint = hit["scientific_name"]
    elif kind == "kazusa_taxid":
        taxid = str(host_request["taxid"])
        name_hint = host_request.get("scientific_name")
    else:
        raise ValueError(f"未知密码子宿主请求类型: {kind}")

    profile = fetch_kazusa_codon_profile(
        taxid,
        scientific_name_hint=str(name_hint) if name_hint else None,
        timeout=timeout,
        cache_dir=cache_dir,
        downloader=downloader,
    )
    profile["selection_source"] = host_request.get("selection_source")
    profile["query"] = host_request.get("query")
    return profile


def _optimize_with_profile(protein_sequence: str, profile: dict) -> dict[str, str | float | int]:
    protein = "".join(str(protein_sequence).split()).upper()
    if not protein:
        raise ValueError("蛋白质序列不能为空")
    preferred_codons = profile["preferred_codons"]
    invalid = sorted(set(protein) - set(preferred_codons))
    if invalid:
        raise ValueError(f"密码子优化只支持 20 种标准氨基酸；发现: {invalid}")

    cds = "".join(preferred_codons[aa] for aa in protein)
    stop = str(profile["preferred_stop"])
    cds_with_stop = cds + stop
    gc = (cds.count("G") + cds.count("C")) / len(cds) if cds else 0.0
    return {
        "optimized_cds": cds,
        "stop_codon": stop,
        "optimized_cds_with_stop": cds_with_stop,
        "protein_length": len(protein),
        "cds_length": len(cds),
        "gc_fraction": gc,
    }


def optimize_protein_to_cds(protein_sequence: str, organism_key: str) -> dict[str, str | float | int]:
    """Offline optimization for one of the 11 built-in hosts."""
    if organism_key not in HOSTS:
        raise ValueError(f"不支持的内置密码子优化宿主: {organism_key}")
    host = HOSTS[organism_key]
    return _optimize_with_profile(
        protein_sequence,
        {"preferred_codons": host.preferred_codons, "preferred_stop": host.preferred_stop},
    )


def _wrap_fasta(sequence: str, width: int = 80) -> Iterable[str]:
    for start in range(0, len(sequence), width):
        yield sequence[start : start + width]


def _safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    return value or "host"


def write_codon_optimization_outputs(
    *,
    run_dir: Path,
    sequence_name: str,
    wt_sequence: str,
    selected: list[Candidate],
    resolved_request: dict,
    cache_dir: str | Path | None = None,
    kazusa_timeout: float = 10.0,
    downloader: Callable[[str, float], tuple[str, str]] | None = None,
) -> dict:
    if not resolved_request.get("enabled"):
        return {
            **resolved_request,
            "status": "not_requested",
            "count": 0,
            "csv": None,
            "fasta": None,
            "per_host_fastas": {},
            "method": None,
            "resolved_hosts": [],
            "failed_hosts": [],
        }

    resolved_hosts: list[dict] = []
    failed_hosts: list[dict] = []
    seen_taxids: set[str] = set()
    for host_request in resolved_request.get("hosts", []):
        try:
            profile = resolve_host_profile(
                host_request,
                timeout=kazusa_timeout,
                cache_dir=cache_dir,
                downloader=downloader,
            )
            taxid = str(profile["taxid"])
            if taxid in seen_taxids:
                continue
            seen_taxids.add(taxid)
            resolved_hosts.append(profile)
        except Exception as exc:
            failed_hosts.append({
                "query": host_request.get("query"),
                "kind": host_request.get("kind"),
                "error": f"{type(exc).__name__}: {exc}",
            })

    if not resolved_hosts:
        warnings = [resolved_request.get("warning")] if resolved_request.get("warning") else []
        if failed_hosts:
            warnings.append("所有请求的动态宿主都未能从 Kazusa 解析/下载，因此未生成密码子优化序列。")
        return {
            **resolved_request,
            "status": "failed",
            "host_count": 0,
            "count": 0,
            "csv": None,
            "fasta": None,
            "per_host_fastas": {},
            "method": "deterministic_most_frequent_synonymous_codon",
            "resolved_hosts": [],
            "failed_hosts": failed_hosts,
            "warnings": [w for w in warnings if w],
        }

    csv_path = run_dir / "05_codon_optimized_sequences.csv"
    fasta_path = run_dir / "05_codon_optimized_sequences.fasta"

    records: list[tuple[str, str, str]] = [("WT", "WT", wt_sequence)]
    records.extend(
        (candidate.mutation, candidate.mutation, candidate.mutant_sequence or "")
        for candidate in selected
    )

    rows: list[dict] = []
    combined_fasta: list[str] = []
    per_host_lines: dict[str, list[str]] = {}
    for host_index, profile in enumerate(resolved_hosts, start=1):
        key = str(profile["organism_key"])
        per_host_lines[key] = []
        for label, mutation, protein in records:
            optimized = _optimize_with_profile(protein, profile)
            row = {
                "name": f"{sequence_name}_{label}",
                "mutation": mutation,
                "host_index": host_index,
                "organism_key": key,
                "organism": profile["scientific_name"],
                "taxid": profile["taxid"],
                "codon_source": profile["codon_source"],
                "protein_length": optimized["protein_length"],
                "cds_length": optimized["cds_length"],
                "gc_fraction": optimized["gc_fraction"],
                "stop_codon": optimized["stop_codon"],
                "optimized_cds": optimized["optimized_cds"],
                "optimized_cds_with_stop": optimized["optimized_cds_with_stop"],
            }
            rows.append(row)
            header = (
                f">{sequence_name}_{label}__{key} mutation={mutation} "
                f"host={profile['scientific_name']} taxid={profile['taxid']} "
                f"source={profile['codon_source']} stop=included"
            )
            seq_lines = list(_wrap_fasta(str(optimized["optimized_cds_with_stop"])))
            combined_fasta.extend([header, *seq_lines])
            per_host_lines[key].extend([header, *seq_lines])

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    fasta_path.write_text("\n".join(combined_fasta) + "\n", encoding="utf-8")

    per_host_fastas: dict[str, str] = {}
    for profile in resolved_hosts:
        key = str(profile["organism_key"])
        path = run_dir / f"05_codon_optimized_{_safe_filename(key)}.fasta"
        path.write_text("\n".join(per_host_lines[key]) + "\n", encoding="utf-8")
        per_host_fastas[key] = str(path)

    warnings: list[str] = []
    if resolved_request.get("warning"):
        warnings.append(str(resolved_request["warning"]))
    if failed_hosts:
        warnings.append(f"{len(failed_hosts)} 个宿主无法从 Kazusa 解析/下载；其余宿主仍已输出。")

    public_hosts = [
        {
            "organism_key": p["organism_key"],
            "scientific_name": p["scientific_name"],
            "display_name_zh": p.get("display_name_zh"),
            "taxid": p["taxid"],
            "codon_source": p["codon_source"],
            "selection_source": p.get("selection_source"),
            "query": p.get("query"),
            "kazusa_url": p.get("kazusa_url"),
            "cds_count": p.get("cds_count"),
            "genetic_code": p.get("genetic_code", 1),
        }
        for p in resolved_hosts
    ]

    result = {
        **resolved_request,
        "status": "partial" if failed_hosts else "success",
        "host_count": len(public_hosts),
        "count": len(rows),
        "csv": str(csv_path),
        "fasta": str(fasta_path),
        "per_host_fastas": per_host_fastas,
        "method": "deterministic_most_frequent_synonymous_codon",
        "resolved_hosts": public_hosts,
        "failed_hosts": failed_hosts,
        "warnings": warnings,
        "includes_wt": True,
        "includes_stop_codon_in_fasta": True,
        "sequences_per_host": len(records),
    }
    if len(public_hosts) == 1:
        host = public_hosts[0]
        result.update({
            "organism_key": host["organism_key"],
            "scientific_name": host["scientific_name"],
            "display_name_zh": host.get("display_name_zh"),
            "taxid": host["taxid"],
        })
    else:
        result.update({
            "organism_key": None,
            "scientific_name": None,
            "display_name_zh": None,
            "taxid": None,
        })
    return result
