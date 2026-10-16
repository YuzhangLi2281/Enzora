from __future__ import annotations

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def normalize_sequence(sequence: str) -> str:
    seq = "".join(sequence.split()).upper()
    if not seq:
        raise ValueError("sequence 不能为空")
    bad = sorted(set(seq) - set(AMINO_ACIDS))
    if bad:
        raise ValueError(f"只支持 20 种标准氨基酸；发现非法字符: {bad}")
    # EpHod warns/truncates above 1022, and PHENOM point-mutation code skips >1024.
    if len(seq) > 1000:
        raise ValueError("当前固定链路要求序列长度 <= 1000 aa，以同时满足 EpHod/PHENOM 推理限制")
    return seq


def generate_single_substitutions(
    sequence: str,
    allowed_positions: list[int] | None = None,
    excluded_positions: list[int] | None = None,
) -> list[str]:
    seq = normalize_sequence(sequence)
    allowed = set(allowed_positions) if allowed_positions is not None else None
    excluded = set(excluded_positions or [])
    n = len(seq)
    for collection, label in [(allowed or set(), "allowed_positions"), (excluded, "excluded_positions")]:
        bad = sorted(p for p in collection if p < 1 or p > n)
        if bad:
            raise ValueError(f"{label} 含越界位置: {bad}")

    out: list[str] = []
    for i, wt in enumerate(seq, start=1):
        if allowed is not None and i not in allowed:
            continue
        if i in excluded:
            continue
        for mt in AMINO_ACIDS:
            if mt != wt:
                out.append(f"{wt}{i}{mt}")
    return out


def parse_mutation(mutation: str) -> tuple[str, int, str]:
    if len(mutation) < 3:
        raise ValueError(f"非法突变格式: {mutation}")
    wt = mutation[0].upper()
    mt = mutation[-1].upper()
    if wt not in AMINO_ACIDS or mt not in AMINO_ACIDS:
        raise ValueError(f"非法氨基酸突变: {mutation}")
    try:
        pos = int(mutation[1:-1])
    except ValueError as exc:
        raise ValueError(f"非法突变位置: {mutation}") from exc
    return wt, pos, mt


def apply_mutation(sequence: str, mutation: str) -> str:
    seq = normalize_sequence(sequence)
    wt, pos, mt = parse_mutation(mutation)
    if not 1 <= pos <= len(seq):
        raise ValueError(f"突变位置越界: {mutation}")
    if seq[pos - 1] != wt:
        raise ValueError(f"突变 {mutation} 的 WT 与输入序列不一致：序列位置 {pos} 是 {seq[pos-1]}")
    return seq[: pos - 1] + mt + seq[pos:]
