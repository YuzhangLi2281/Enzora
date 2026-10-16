from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections.abc import MutableMapping
from typing import Optional


@dataclass
class Candidate:
    mutation: str
    wt_sequence: str
    mutant_sequence: Optional[str] = None

    # External model predictions.
    activity_log10: Optional[float] = None
    activity_sigma_log10: Optional[float] = None
    activity_kcat: Optional[float] = None
    library_score: Optional[float] = None
    ph_uncertainty: Optional[float] = None
    stability_uncertainty: Optional[float] = None
    score_normalization: Optional[str] = None
    ph_opt: Optional[float] = None
    delta_ph_from_wt: Optional[float] = None
    ddg: Optional[float] = None

    # Deterministic scoring components.
    activity_component: Optional[float] = None
    ph_component: Optional[float] = None
    stability_component: Optional[float] = None
    final_score: Optional[float] = None
    score_evidence: Optional[dict] = None
    passed: bool = True
    reasons: list[str] = field(default_factory=list)

    substitutions: list[dict] = field(default_factory=list)

    # Deterministic point-mutation descriptors.
    position: Optional[int] = None
    wt_aa: Optional[str] = None
    mutant_aa: Optional[str] = None
    wt_residue_class: Optional[str] = None
    mutant_residue_class: Optional[str] = None
    residue_class_change: Optional[str] = None
    wt_hydrophobicity: Optional[float] = None
    mutant_hydrophobicity: Optional[float] = None
    delta_hydrophobicity: Optional[float] = None
    wt_volume: Optional[float] = None
    mutant_volume: Optional[float] = None
    delta_volume: Optional[float] = None
    wt_charge: Optional[float] = None
    mutant_charge: Optional[float] = None
    delta_charge: Optional[float] = None
    blosum62: Optional[int] = None
    conservative_substitution: Optional[bool] = None
    pareto_optimal: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BioArtifact(MutableMapping):
    """Shared tool envelope. Sequence records, predictions and provenance travel together.

    ``data`` holds tool-declared artifacts; mapping access is the tool contract.
    JSON is the full-fidelity interchange format, including the declared outputs.
    """

    data: dict = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)
    predictions: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        from copy import deepcopy
        self.data = deepcopy(self.data)
        self.records = deepcopy(self.records)
        self.predictions = deepcopy(self.predictions)
        self.metadata = deepcopy(self.metadata)
        self.validate_records()
        for key, value in tuple(self.data.items()):
            self[key] = value

    def validate_records(self):
        alphabets = {'protein': set('ACDEFGHIKLMNPQRSTVWYXBZJUO*'),
                     'dna': set('ACGTRYSWKMBDHVN'), 'rna': set('ACGURYSWKMBDHVN')}
        identifiers = set()
        for item in self.records:
            if not isinstance(item, dict) or set(item) != {'id', 'name', 'type', 'sequence'}:
                raise ValueError('BioArtifact records require id, name, type and sequence')
            if not isinstance(item['id'], str) or not item['id'] or item['id'] in identifiers:
                raise ValueError('BioArtifact record IDs must be unique nonempty strings')
            identifiers.add(item['id'])
            kind, sequence = item['type'], item['sequence']
            if kind not in alphabets or not isinstance(sequence, str) or not sequence or set(sequence) - alphabets[kind]:
                raise ValueError(f'Invalid BioArtifact sequence or molecule type: {item["id"]}')
            if not isinstance(item['name'], str):
                raise ValueError('BioArtifact record names must be strings')

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        if key == 'protein_sequence':
            existing = next((r for r in self.records if r['id'] == 'WT'), None)
            wt = existing if existing and existing['sequence'] == value and existing['type'] == 'protein' else {
                'id': 'WT', 'name': 'WT', 'type': 'protein', 'sequence': value}
            previous = self.records
            self.records = [r for r in self.records if r['id'] != 'WT'] + [wt]
            try:
                self.validate_records()
            except ValueError:
                self.records = previous
                raise
        if key in {'protein_sequences', 'mutant_sequences'}:
            if not isinstance(value, dict):
                raise ValueError(f'{key} must map record IDs to protein sequences')
            records = {r['id']: r for r in self.records}
            for identifier, sequence in value.items():
                identifier = str(identifier)
                existing = records.get(identifier)
                if existing and existing['sequence'] == sequence and existing['type'] == 'protein':
                    continue
                records[identifier] = {'id': identifier, 'name': identifier,
                                       'type': 'protein', 'sequence': sequence}
            previous = self.records
            self.records = list(records.values())
            try:
                self.validate_records()
            except ValueError:
                self.records = previous
                raise
        if key.endswith('_predictions'):
            self.predictions[key] = value
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]
        self.predictions.pop(key, None)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def merge(self, other):
        from copy import deepcopy
        if isinstance(other, BioArtifact):
            records = {r['id']: deepcopy(r) for r in self.records}
            records.update({r['id']: deepcopy(r) for r in other.records})
            self.records = list(records.values())
            self.predictions.update(deepcopy(other.predictions))
            self.metadata.update(deepcopy(other.metadata))
        self.update(other)
        self.validate_records()

    def to_dict(self):
        from dataclasses import is_dataclass
        def encode(value):
            if is_dataclass(value):
                return encode(asdict(value))
            if isinstance(value, dict):
                return {str(k): encode(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [encode(v) for v in value]
            return value
        return {'schema': 'enzora.bioartifact/1', 'records': encode(self.records),
                'predictions': encode(self.predictions), 'metadata': encode(self.metadata),
                'data': encode(self.data)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get('schema') != 'enzora.bioartifact/1':
            raise ValueError('Expected enzora.bioartifact/1 JSON object')
        if set(value) != {'schema', 'records', 'predictions', 'metadata', 'data'}:
            raise ValueError('BioArtifact JSON fields do not match the schema')
        if not isinstance(value['records'], list) or any(not isinstance(value[k], dict) for k in ('predictions', 'metadata', 'data')):
            raise ValueError('Invalid BioArtifact envelope types')
        import json
        json.dumps(value, allow_nan=False)
        records = {r['id']: r for r in value['records'] if isinstance(r, dict) and 'id' in r}
        for key in ('protein_sequences', 'mutant_sequences'):
            sequences = value['data'].get(key)
            if sequences is not None:
                if not isinstance(sequences, dict):
                    raise ValueError(f'{key} must be an object')
                for identifier, sequence in sequences.items():
                    if identifier in records and (records[identifier]['sequence'] != sequence or records[identifier]['type'] != 'protein'):
                        raise ValueError('BioArtifact data and sequence records disagree')
        for key, predictions in value['predictions'].items():
            if key in value['data'] and value['data'][key] != predictions:
                raise ValueError('BioArtifact prediction fields disagree')
        return cls(**{k: value[k] for k in ('data', 'records', 'predictions', 'metadata')})
