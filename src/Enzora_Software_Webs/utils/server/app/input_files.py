"""Read sequence-bearing files without interpreting coordinate files as a viewer."""
from __future__ import annotations

from collections import OrderedDict
from io import StringIO
from pathlib import Path
from typing import Any

from .semantic_input import MAX_RECORDS, check_size, classify_records, classify_text, record, result

FASTA_EXTENSIONS = {".fa", ".fasta", ".fas", ".faa", ".fna", ".ffn", ".frn"}
KNOWN_TYPES = {".faa": "protein", ".fna": "dna", ".ffn": "dna", ".frn": "rna"}
RESIDUES = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    "ACDEFGHIKLMNPQRSTVWY"))
RESIDUES.update({"MSE": "M", "SEC": "U", "PYL": "O", "UNK": "X", "ASX": "B", "GLX": "Z"})
NUCLEOTIDES = {"DA": "A", "DC": "C", "DG": "G", "DT": "T", "DI": "N",
               "A": "A", "C": "C", "G": "G", "U": "U", "I": "N"}


def parse_fasta(content: str, *, type_override: str | None = None) -> tuple[list[dict[str, str]], list[str]]:
    records: list[dict[str, str]] = []
    name: str | None = None
    sequence: list[str] = []

    def finish() -> None:
        if name is None:
            return
        seq = "".join(sequence).upper()
        if not seq:
            raise ValueError(f"FASTA 记录 {name} 没有序列。")
        entry = {"id": str(len(records) + 1), "name": name, "sequence": seq}
        records.append(record(seq, type_override, identifier=entry["id"], name=name) if type_override else entry)
        if len(records) > MAX_RECORDS:
            raise ValueError(f"单个文件最多支持 {MAX_RECORDS} 条记录。")

    for line in content.lstrip("\ufeff").splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            finish()
            name = line[1:].strip() or f"sequence_{len(records) + 1}"
            sequence = []
        else:
            if name is None:
                raise ValueError("FASTA 文件必须以 > 标题行开始。")
            if any(ch.isspace() for ch in line) or not all(ch.isascii() and (ch.isalpha() or ch == "*") for ch in line):
                raise ValueError(f"FASTA 记录 {name} 中包含说明文字、空格或不支持的字符。")
            sequence.append(line)
    finish()
    if not records:
        raise ValueError("文件中没有 FASTA 记录。")
    return records, []


def _chains(chains: dict[str, list[str]], *, source: str) -> tuple[list[dict[str, str]], list[str]]:
    records: list[dict[str, str]] = []
    warnings: list[str] = []
    for chain, residues in chains.items():
        if all(r in NUCLEOTIDES for r in residues):
            kind = "dna" if any(r.startswith("D") for r in residues) else "rna"
            seq = "".join(NUCLEOTIDES[r] for r in residues)
        elif all(r in RESIDUES for r in residues):
            kind = "protein"
            seq = "".join(RESIDUES[r] for r in residues)
        else:
            unknown = sorted({r for r in residues if r not in RESIDUES and r not in NUCLEOTIDES})
            raise ValueError(f"链 {chain or '_'} 包含无法转换的残基或混合聚合物：{', '.join(unknown) or '蛋白/核酸混合'}。请导出对应 FASTA 后上传。")
        records.append(record(seq, kind, identifier=chain or "_", name=f"Chain {chain or '_'} ({source})"))
        if "MSE" in residues:
            warnings.append(f"链 {chain or '_'} 的 MSE 按其对应氨基酸 M 读取。")
    if not records:
        raise ValueError("结构文件中未找到可读取的蛋白或核酸链。")
    return records, warnings


def parse_pdb(content: str) -> tuple[list[dict[str, str]], list[str]]:
    seqres: dict[str, list[str]] = OrderedDict()
    for line in content.splitlines():
        if line.startswith("SEQRES"):
            seqres.setdefault(line[11:12].strip(), []).extend(line[19:70].split())
    if seqres:
        return _chains(seqres, source="SEQRES")
    atoms: dict[str, list[str]] = OrderedDict()
    seen: set[tuple[str, str]] = set()
    residue_numbers: dict[str, list[int]] = {}
    for line in content.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        residue, chain = line[17:20].strip(), line[21:22].strip()
        if line.startswith("HETATM") and residue not in RESIDUES:
            continue
        key = (chain, line[22:27])
        if key in seen:
            continue
        seen.add(key)
        atoms.setdefault(chain, []).append(residue)
        try:
            residue_numbers.setdefault(chain, []).append(int(line[22:26]))
        except ValueError:
            pass
    records, warnings = _chains(atoms, source="ATOM")
    warnings.append("PDB 未提供 SEQRES，已从第一个模型的已解析残基读取；缺失残基不会补齐，请确认是否为完整序列。")
    for chain, numbers in residue_numbers.items():
        if any(b - a > 1 for a, b in zip(numbers, numbers[1:])):
            warnings.append(f"链 {chain or '_'} 的坐标残基编号不连续，序列可能不完整。")
    return records, warnings


def parse_cif(content: str) -> tuple[list[dict[str, str]], list[str]]:
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    data = MMCIF2Dict(StringIO(content))

    def values(key: str) -> list[str]:
        value = data.get(key, [])
        return [value] if isinstance(value, str) else value

    entity_ids = values("_entity_poly_seq.entity_id")
    monomers = values("_entity_poly_seq.mon_id")
    sequences: dict[str, list[str]] = OrderedDict()
    for entity, residue in zip(entity_ids, monomers):
        sequences.setdefault(entity, []).append(residue.upper())
    if sequences:
        asym_ids, asym_entities = values("_struct_asym.id"), values("_struct_asym.entity_id")
        chains = OrderedDict((chain, sequences[entity]) for chain, entity in zip(asym_ids, asym_entities)
                             if entity in sequences)
        return _chains(chains or sequences, source="mmCIF polymer sequence")
    # Coordinate-only files retain explicit incompleteness warnings.
    chains: dict[str, list[str]] = OrderedDict()
    seen: set[tuple[str, str]] = set()
    identifiers = values("_atom_site.auth_asym_id") or values("_atom_site.label_asym_id")
    numbers = values("_atom_site.auth_seq_id") or values("_atom_site.label_seq_id")
    residue_names = values("_atom_site.label_comp_id")
    models = values("_atom_site.pdbx_PDB_model_num") or ["1"] * len(identifiers)
    insertions = values("_atom_site.pdbx_PDB_ins_code") or ["?"] * len(identifiers)
    groups = values("_atom_site.group_PDB") or ["ATOM"] * len(identifiers)
    for chain, number, residue, model, insertion, group in zip(identifiers, numbers, residue_names, models, insertions, groups):
        if model != models[0] or (group == "HETATM" and residue not in RESIDUES):
            continue
        key = (chain, number + insertion)
        if key not in seen:
            seen.add(key)
            chains.setdefault(chain, []).append(residue.upper())
    records, warnings = _chains(chains, source="mmCIF coordinates")
    warnings.append("mmCIF 未提供完整聚合物序列，已读取第一个模型的坐标残基；缺失残基不会补齐，请确认完整性。")
    return records, warnings


def _bio_records(content: str, format_name: str, override: str | None) -> tuple[list[dict[str, str]], list[str]]:
    from Bio import SeqIO

    entries: list[dict[str, str]] = []
    for item in SeqIO.parse(StringIO(content), format_name):
        molecule = str(item.annotations.get("molecule_type", "")).lower()
        kind = override or ("protein" if "protein" in molecule else "rna" if "rna" in molecule else "dna")
        entries.append(record(str(item.seq), kind, identifier=str(len(entries) + 1),
                              name=item.description or item.id))
        if len(entries) > MAX_RECORDS:
            raise ValueError(f"单个文件最多支持 {MAX_RECORDS} 条记录。")
    if not entries:
        raise ValueError("文件中没有序列记录。")
    return entries, []


def _parse_sequence_file(filename: str, content: str, *, client: Any = None, model: str = "MiniMax-M2.5",
               type_override: str | None = None) -> dict[str, Any]:
    check_size(content)
    suffix = Path(filename.lower()).suffix
    warnings: list[str] = []
    extra: dict[str, Any] = {}
    if suffix in FASTA_EXTENSIONS:
        entries, warnings = parse_fasta(content, type_override=type_override or KNOWN_TYPES.get(suffix))
        if entries and "type" not in entries[0]:
            entries, extra = classify_records(entries, client=client, model=model)
    elif suffix in {".pdb", ".ent"}:
        entries, warnings = parse_pdb(content)
    elif suffix in {".cif", ".mmcif"}:
        entries, warnings = parse_cif(content)
    elif suffix in {".gb", ".gbk", ".genbank"}:
        entries, warnings = _bio_records(content, "genbank", type_override)
    elif suffix in {".fq", ".fastq"}:
        entries, warnings = _bio_records(content, "fastq", type_override)
    elif suffix in {".txt", ".seq"}:
        if content.lstrip("\ufeff \r\n\t").startswith(">"):
            entries, warnings = parse_fasta(content, type_override=type_override)
            if "type" not in entries[0]:
                entries, extra = classify_records(entries, client=client, model=model)
        else:
            return {**classify_text(content, client=client, model=model, type_override=type_override), "filename": filename}
    else:
        raise ValueError("支持 FASTA/FA/FAA/FNA/FFN/FRN、TXT/SEQ、FASTQ/FQ、GenBank/GB/GBK、PDB/ENT、CIF/mmCIF。")
    if type_override and any(r["type"] != type_override for r in entries):
        raise ValueError("手动选择的类型与文件中声明的聚合物类型不一致，请核对文件或取消类型覆盖。")
    warnings.extend(str(w) for w in extra.get("warnings", []) if isinstance(w, str))
    if len(entries) > 1:
        warnings.append("文件含多条记录或链，请选择一条进行本次运行；不会自动拼接。")
    return {**result(entries, interpretation=str(extra.get("interpretation") or
                    f"从 {filename} 读取到 {len(entries)} 条序列，请核对并选择。"), warnings=warnings,
                    ambiguous=bool(extra.get("ambiguous"))), "filename": filename}


def read_bioartifact(filename: str, content: str, *, type_override: str | None = None,
                     client: Any = None, model: str = 'MiniMax-M2.5'):
    """Read supported sequence formats into the one BioArtifact schema.

    SBOL support covers explicit Sequence objects, not assembly/circuit semantics.
    External references are never fetched. FASTA retains sequence identity/type;
    structured annotations require JSON, CSV, GenBank or sequence-only SBOL.
    """
    import csv
    import json
    import re
    import xml.etree.ElementTree as ET
    from Enzora_Agent.core.types import BioArtifact

    check_size(content)
    suffix = Path(filename.lower()).suffix
    if suffix == '.json':
        artifact = BioArtifact.from_dict(json.loads(content))
    elif suffix == '.csv':
        reader = csv.DictReader(StringIO(content))
        required = {'id', 'name', 'type', 'sequence'}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError('CSV 必须包含 id,name,type,sequence 四列。')
        entries, envelope = [], None
        for row in reader:
            if None in row or any(row[k] is None for k in required):
                raise ValueError('CSV 行字段数量不一致。')
            extras = {k: json.loads(row[k]) for k in ('predictions', 'metadata', 'data') if row.get(k)}
            if envelope is not None and extras != envelope:
                raise ValueError('CSV 各行的 BioArtifact 说明与预测必须一致。')
            envelope = extras
            entries.append({k: row[k] for k in required})
        artifact = BioArtifact(records=entries, **(envelope or {}))
    elif suffix in {'.sbol', '.xml', '.rdf'}:
        if '<!DOCTYPE' in content.upper() or '<!ENTITY' in content.upper():
            raise ValueError('SBOL 不接受 DTD 或外部实体。')
        try:
            root = ET.fromstring(content)
        except ET.ParseError as exc:
            raise ValueError('SBOL XML 语法无效。') from exc
        rdf = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
        en = 'https://enzora.org/bioartifact#'
        entries, envelope = [], None
        for element in root:
            if element.tag not in {'{http://sbols.org/v2#}Sequence', '{http://sbols.org/v3#}Sequence'}:
                continue
            namespace = element.tag[1:].split('}')[0]
            sequence = element.findtext('{'+namespace+'}elements')
            encoding = element.find('{'+namespace+'}encoding')
            if not sequence or encoding is None:
                raise ValueError('SBOL Sequence 必须包含 elements 和 encoding。')
            uri = encoding.get('{'+rdf+'}resource')
            protein = uri in {'http://www.chem.qmul.ac.uk/iupac/AminoAcid/', 'https://identifiers.org/edam:format_1208'}
            nucleic = uri in {'http://www.chem.qmul.ac.uk/iubmb/misc/naseq.html', 'https://identifiers.org/edam:format_1207'}
            if not protein and not nucleic:
                raise ValueError(f'不支持的 SBOL 序列编码: {uri}')
            kind = element.findtext('{'+en+'}moleculeType') or type_override
            if kind is None:
                if protein:
                    kind = 'protein'
                elif 'U' in sequence.upper():
                    kind = 'rna'
                elif 'T' in sequence.upper():
                    kind = 'dna'
                else:
                    raise ValueError('SBOL 核酸编码不能区分 DNA/RNA，请指定输入类型。')
            if (protein and kind != 'protein') or (nucleic and kind not in {'dna', 'rna'}):
                raise ValueError('SBOL molecule type 与 encoding 不一致。')
            identifier = element.findtext('{'+en+'}recordId') or element.get('{'+rdf+'}about')
            if not identifier:
                raise ValueError('SBOL Sequence 缺少标识。')
            name = element.findtext('{'+namespace+'}name') or element.findtext('{http://purl.org/dc/terms/}title') or identifier
            entries.append(record(sequence, kind, identifier=identifier, name=name))
            encoded = element.findtext('{'+en+'}envelope')
            if encoded:
                value = json.loads(encoded)
                if envelope is not None and envelope != value:
                    raise ValueError('SBOL 说明对象不一致。')
                envelope = value
        artifact = BioArtifact(records=entries, **(envelope or {}))
    elif suffix in {'.gb', '.gbk', '.genbank'}:
        from Bio import SeqIO
        entries, envelope = [], None
        for item in SeqIO.parse(StringIO(content), 'genbank'):
            molecule = str(item.annotations.get('molecule_type', '')).lower()
            kind = type_override or ('protein' if 'protein' in molecule else 'rna' if 'rna' in molecule else 'dna')
            entries.append(record(str(item.seq), kind, identifier=item.id, name=item.description))
            notes = item.annotations.get('structured_comment', {}).get('Enzora', {})
            if 'Envelope' in notes:
                value = json.loads(notes['Envelope'])
                if envelope is not None and envelope != value:
                    raise ValueError('GenBank 说明对象不一致。')
                envelope = value
        artifact = BioArtifact(records=entries, **(envelope or {}))
        if envelope is None:
            artifact.metadata['warnings'] = ['GenBank 仅导入序列身份与类型，其他 feature/annotation 不进入模型输入。']
    else:
        headers = [line[1:] for line in content.splitlines() if line.startswith('>')]
        typed = [re.fullmatch(r'(\S+)\s+(.*?)\s+\[enzora_type=(protein|dna|rna)\]', h) for h in headers]
        if headers and all(typed) and suffix in FASTA_EXTENSIONS:
            entries, _ = parse_fasta(content, type_override='protein' if all(m[3] == 'protein' for m in typed) else None)
            artifact = BioArtifact(records=[record(e['sequence'], m[3], identifier=m[1], name=m[2])
                                            for e, m in zip(entries, typed)])
        else:
            parsed = _parse_sequence_file(filename, content, type_override=type_override, client=client, model=model)
            artifact = BioArtifact(records=parsed['records'], metadata={'warnings': parsed['warnings']})
    if not artifact.records or len(artifact.records) > MAX_RECORDS:
        raise ValueError(f'文件必须包含 1–{MAX_RECORDS} 条序列记录。')
    if type_override and any(r['type'] != type_override for r in artifact.records):
        raise ValueError('手动输入类型与文件中的类型不一致。')
    return artifact


def export_bioartifact(artifact, format_name: str) -> tuple[str, list[str]]:
    """Serialize a BioArtifact; explicitly report annotations lost by a format."""
    import csv
    import hashlib
    import json
    import re
    import xml.etree.ElementTree as ET
    from urllib.parse import quote
    from Enzora_Agent.core.types import BioArtifact

    if not isinstance(artifact, BioArtifact):
        raise ValueError('Expected BioArtifact')
    artifact.validate_records()
    if not artifact.records or len(artifact.records) > MAX_RECORDS:
        raise ValueError(f'导出必须包含 1–{MAX_RECORDS} 条记录。')
    value = artifact.to_dict()
    # Also rejects non-JSON model tensors instead of silently stringifying them.
    json.dumps(value, ensure_ascii=False, allow_nan=False)
    envelope = {k: value[k] for k in ('predictions', 'metadata', 'data')}
    fmt = format_name.lower()
    if fmt == 'json':
        return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), []
    if fmt == 'csv':
        stream = StringIO()
        writer = csv.DictWriter(stream, fieldnames=['id', 'name', 'type', 'sequence', 'predictions', 'metadata', 'data'])
        writer.writeheader()
        for item in artifact.records:
            writer.writerow({**{k: item[k] for k in ('id','name','type','sequence')},
                             **{k: json.dumps(v, ensure_ascii=False, allow_nan=False) for k,v in envelope.items()}})
        return stream.getvalue(), []
    if fmt in {'fasta', 'fa'}:
        lines = []
        for item in artifact.records:
            if re.search(r'\s', item['id']) or any(c in item['name'] for c in '\r\n'):
                raise ValueError('FASTA 导出要求 id 无空白、name 无换行。请改用 JSON。')
            lines.append(f">{item['id']} {item['name']} [enzora_type={item['type']}]")
            lines.extend(item['sequence'][i:i+80] for i in range(0, len(item['sequence']), 80))
        warnings = ['FASTA 仅保留序列、标识、名称与类型，预测和说明请同时导出 JSON。'] if any(envelope.values()) else []
        return '\n'.join(lines)+'\n', warnings
    if fmt in {'genbank', 'gb', 'gbk'}:
        from Bio import SeqIO
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        records = []
        for item in artifact.records:
            if any(c.isspace() for c in item['id']) or '\n' in item['name']:
                raise ValueError('GenBank 导出要求 id 无空白、name 无换行。')
            row = SeqRecord(Seq(item['sequence']), id=item['id'], name=item['id'], description=item['name'])
            row.annotations['molecule_type'] = {'protein':'protein','dna':'DNA','rna':'RNA'}[item['type']]
            row.annotations['structured_comment'] = {'Enzora': {'Envelope': json.dumps(envelope, ensure_ascii=False, allow_nan=False)}}
            records.append(row)
        stream = StringIO(); SeqIO.write(records, stream, 'genbank')
        return stream.getvalue(), []
    if fmt == 'sbol':
        rdf, sbol, en = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#', 'http://sbols.org/v3#', 'https://enzora.org/bioartifact#'
        for prefix, namespace in [('rdf',rdf),('sbol',sbol),('enzora',en)]:
            ET.register_namespace(prefix,namespace)
        root = ET.Element('{'+rdf+'}RDF')
        for i,item in enumerate(artifact.records):
            identity = 'https://enzora.org/sequences/'+hashlib.sha256(item['sequence'].encode()).hexdigest()[:16]+'/'+quote(item['id'],safe='')
            node = ET.SubElement(root,'{'+sbol+'}Sequence', {'{'+rdf+'}about':identity})
            for tag,text in [('name',item['name']),('elements',item['sequence'])]:
                ET.SubElement(node,'{'+sbol+'}'+tag).text=text
            ET.SubElement(node,'{'+sbol+'}encoding', {'{'+rdf+'}resource':'https://identifiers.org/edam:format_'+('1208' if item['type']=='protein' else '1207')})
            ET.SubElement(node,'{'+en+'}recordId').text=item['id']
            ET.SubElement(node,'{'+en+'}moleculeType').text=item['type']
            if i == 0:
                ET.SubElement(node,'{'+en+'}envelope').text=json.dumps(envelope,ensure_ascii=False,allow_nan=False)
        return ET.tostring(root,encoding='unicode',xml_declaration=True), []
    raise ValueError('导出支持 fasta、csv、json、genbank、sbol。')


def parse_file(filename: str, content: str, *, client: Any = None, model: str = 'MiniMax-M2.5',
               type_override: str | None = None) -> dict[str, Any]:
    from Enzora_Agent.core.types import BioArtifact
    suffix = Path(filename.lower()).suffix
    typed_fasta = '[enzora_type=' in content and suffix in FASTA_EXTENSIONS
    if suffix not in {'.json', '.csv', '.sbol', '.xml', '.rdf', '.gb', '.gbk', '.genbank'} and not typed_fasta:
        parsed = _parse_sequence_file(filename, content, client=client, model=model, type_override=type_override)
        artifact = BioArtifact(records=parsed['records'], metadata={
            k: parsed[k] for k in ('warnings', 'request_text', 'interpretation', 'ambiguous') if k in parsed})
        return {**parsed, 'bioartifact': artifact.to_dict()}
    artifact = read_bioartifact(filename, content, client=client, model=model, type_override=type_override)
    return {**result(artifact.records, warnings=artifact.metadata.get('warnings', []),
                    interpretation=f'从 {filename} 读取 {len(artifact.records)} 条序列，请核对并选择。'),
            'filename': filename, 'bioartifact': artifact.to_dict()}
