"""Deterministic nucleotide translation; input recognition lives in semantic_input."""
from __future__ import annotations

from typing import Any

STOP_CODONS = frozenset(("TAA", "TAG", "TGA"))

CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "CGT": "R", "CGC": "R",
    "CGA": "R", "CGG": "R", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

def translate_nucleotide(sequence: str) -> dict[str, Any]:
    """Translate a DNA/RNA sequence with the standard genetic code.

    ORF = first ATG through the first in-frame stop codon. Deterministic and
    offline; warnings describe any deviation from a clean CDS.
    """
    dna = str(sequence).replace("U", "T").upper()
    dna = "".join(dna.split())
    warnings: list[str] = []
    if not dna:
        raise ValueError("empty nucleotide sequence")

    start = dna.find("ATG")
    if start < 0:
        start = 0
        warnings.append("no ATG start codon found; translated from position 1")
    body = dna[start:]
    if len(body) % 3:
        warnings.append("length not a multiple of 3; trailing bases trimmed")
        body = body[: len(body) - len(body) % 3]

    amino: list[str] = []
    stop_index: int | None = None
    for i in range(0, len(body), 3):
        codon = body[i : i + 3]
        if codon in STOP_CODONS:
            stop_index = i // 3
            break
        amino.append(CODON_TABLE.get(codon, "X"))
    if stop_index is None:
        warnings.append("no in-frame stop codon; translated to end of input")

    protein = "".join(amino)
    orf_start = start + 1
    orf_end = start + (stop_index * 3 if stop_index is not None else len(body))
    return {
        "protein_sequence": protein,
        "orf_start_1based": orf_start,
        "orf_end_1based": orf_end,
        "cds_length": len(body),
        "warnings": warnings,
    }
