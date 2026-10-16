"""Sequence translation plug-in (summary.md 5.4, branch A/B of the input router).

Translates DNA/RNA to protein with the standard genetic code (code 1), from the
first ATG through the first in-frame stop codon. Deterministic and offline; the
LLM is never involved. Enables DAG plans such as:

    sequence_translate -> codon_optimizer
"""
from Enzora_Agent.tools import AgentTool, ToolContext, ToolSpec

CODON_TABLE = {
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
STOP_CODONS = ("TAA", "TAG", "TGA")


class SequenceTranslateTool(AgentTool):
    def __init__(self, manifest):
        self.spec = ToolSpec(
            name=manifest["name"],
            version=manifest["version"],
            description=manifest["description"],
            capabilities=tuple(manifest["capabilities"]),
            inputs=tuple(manifest["inputs"]),
            outputs=tuple(manifest["outputs"]),
            properties={"offline": True, "batch": True, "gpu": False},
        )

    def run(self, context: ToolContext):
        raw = str(context.artifacts["nucleotide_sequence"])
        sequence_type = str(context.artifacts.get("sequence_type", "dna")).lower()
        if sequence_type not in {"dna", "rna"}:
            raise ValueError("sequence_type must be 'dna' or 'rna'")
        dna = raw.replace("U", "T").upper()
        dna = "".join(dna.split())
        if not dna:
            raise ValueError("empty nucleotide sequence")

        warnings = []
        start = dna.find("ATG")
        if start < 0:
            start = 0
            warnings.append("no ATG start codon; translated from position 1")
        body = dna[start:]
        if len(body) % 3:
            warnings.append("length not a multiple of 3; trailing bases trimmed")
            body = body[: len(body) - len(body) % 3]

        amino = []
        stop_index = None
        for i in range(0, len(body), 3):
            codon = body[i : i + 3]
            if codon in STOP_CODONS:
                stop_index = i // 3
                break
            amino.append(CODON_TABLE.get(codon, "X"))
        if stop_index is None:
            warnings.append("no in-frame stop codon; translated to end")

        protein = "".join(amino)
        metadata = {
            "sequence_type": sequence_type,
            "orf_start_1based": start + 1,
            "orf_end_1based": start + (stop_index * 3 if stop_index is not None else len(body)),
            "cds_length": len(body),
            "protein_length": len(protein),
            "warnings": warnings,
        }
        return {"protein_sequence": protein, "translation_metadata": metadata}


def create_tool(manifest):
    return SequenceTranslateTool(manifest)
