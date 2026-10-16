from __future__ import annotations

from pathlib import Path
import csv

from ..adapters import PhenomAdapter, EpHodAdapter, UniStabAdapter
from ..outputs.codon_optimization import resolve_codon_optimization_request, write_codon_optimization_outputs
from ..core.runtime import build_env
from ..core.types import Candidate
from .base import AgentTool, ToolContext, ToolHealth, ToolSpec


def _runtime(context: ToolContext) -> tuple[dict[str, str], int, bool]:
    runtime = context.config["runtime"]
    force_cpu = bool(context.config.get("force_cpu", False))
    env = build_env(
        force_cpu=force_cpu,
        gpu_id=runtime["gpu_id"],
        cache_dir=runtime["cache_dir"],
    )
    return env, runtime["timeout_seconds"], force_cpu


class VESMLibraryTool(AgentTool):
    spec = ToolSpec('vesm', '1.0', 'Construct a quota-controlled mutation library with VESM ensemble',
        ('mutation_library_construction',), ('protein_sequence',),
        ('mutation_list', 'protein_sequences', 'mutant_sequences', 'library_manifest'),
        properties={'gpu': True, 'offline': True})

    def __init__(self, config: dict):
        from ..adapters.vesm import VESMAdapter
        self.config = config
        self.adapter = VESMAdapter(config['mutation'])

    def healthcheck(self) -> ToolHealth:
        errors = self.adapter.preflight()
        return ToolHealth(not errors, tuple(errors))

    def run(self, context: ToolContext):
        env, timeout, force_cpu = _runtime(context)
        rows = self.adapter.build(context.artifacts['protein_sequence'], self.config['candidate'],
            work_dir=context.run_dir / context.step_id, env=env, timeout=timeout, force_cpu=force_cpu)
        sequences = {row['mutations']: row['sequence'] for row in rows}
        return {'mutation_list': list(sequences), 'protein_sequences': sequences,
                'mutant_sequences': sequences, 'library_manifest': rows}


class PhenomTool(AgentTool):
    spec = ToolSpec('phenom', '2.0', 'PHENOM ligand-conditioned kcat prediction',
        ('activity_prediction', 'activity_ranking'), ('protein_sequences', 'ligand_smiles'),
        ('activity_predictions',), properties={'batch': True, 'gpu': True, 'offline': True})

    def __init__(self, model_config: dict):
        self.adapter = PhenomAdapter(model_config)

    def healthcheck(self) -> ToolHealth:
        errors = self.adapter.preflight()
        return ToolHealth(not errors, tuple(errors))

    def run(self, context: ToolContext):
        env, timeout, force_cpu = _runtime(context)
        predictions = self.adapter.predict(context.artifacts['protein_sequences'],
            ligand_smiles=context.artifacts['ligand_smiles'], work_dir=context.run_dir / context.step_id,
            env=env, timeout=timeout, force_cpu=force_cpu)
        return {'activity_predictions': predictions}


class BalanceScoresTool(AgentTool):
    spec = ToolSpec('balance_scores', '1.0', 'Balance explicit preference weights by reliability controls',
        ('score_balancing',), ('score_preferences', 'scorer_confidence'), ('effective_weights',),
        properties={'gpu': False, 'offline': True})

    def run(self, context: ToolContext):
        from ..core.screening import balance_weights
        return {'effective_weights': balance_weights(context.artifacts['score_preferences'],
                                                     context.artifacts['scorer_confidence'])}


class EpHodTool(AgentTool):
    spec = ToolSpec(
        "ephod", "1.0", "Protein optimum pH prediction provider",
        ("ph_prediction",), ("protein_sequences",), ("ph_predictions",),
        properties={"batch": True, "gpu": False, "offline": True},
    )

    def __init__(self, model_config: dict):
        self.adapter = EpHodAdapter(model_config["repo"], model_config["python"], model_config["weights_dir"])

    def healthcheck(self) -> ToolHealth:
        errors = self.adapter.preflight()
        return ToolHealth(not errors, tuple(errors))

    def run(self, context: ToolContext):
        env, timeout, _ = _runtime(context)
        values = self.adapter.predict_with_uncertainty(
            context.artifacts["protein_sequences"],
            work_dir=context.run_dir / context.step_id, env=env, timeout=timeout,
        )
        return {"ph_predictions": values}


class UniStabTool(AgentTool):
    spec = ToolSpec(
        "unistab", "1.0", "Mutation stability prediction provider",
        ("stability_prediction",), ("protein_sequence", "mutant_sequences"),
        ("stability_predictions",),
        properties={"batch": True, "gpu": True, "offline": True},
    )

    def __init__(self, model_config: dict):
        self.adapter = UniStabAdapter(
            model_config["repo"], model_config["python"], model_config["checkpoint"], model_config["weights_dir"]
        )

    def healthcheck(self) -> ToolHealth:
        errors = self.adapter.preflight()
        return ToolHealth(not errors, tuple(errors))

    def run(self, context: ToolContext):
        env, timeout, force_cpu = _runtime(context)
        values = self.adapter.predict_ddg_with_uncertainty(
            context.artifacts["protein_sequence"], context.artifacts["mutant_sequences"],
            work_dir=context.run_dir / context.step_id, env=env, timeout=timeout,
            force_cpu=force_cpu,
        )
        return {"stability_predictions": values}


class CodonOptimizationTool(AgentTool):
    spec = ToolSpec(
        "codon_optimizer", "1.0", "Multi-host synonymous codon optimization provider",
        ("codon_optimization",),
        ("protein_sequence", "selected_candidates", "host_organisms"),
        ("optimized_cds_set", "codon_optimization_result"),
        properties={"batch": True, "gpu": False, "network": "optional", "offline_builtin_hosts": True},
    )

    @staticmethod
    def _candidate(value, wt_sequence: str) -> Candidate:
        if isinstance(value, Candidate):
            return value
        if not isinstance(value, dict) or not value.get("mutation"):
            raise ValueError("selected_candidates must contain Candidate objects or mappings")
        candidate = Candidate(mutation=str(value["mutation"]), wt_sequence=wt_sequence)
        candidate.mutant_sequence = value.get("mutant_sequence")
        return candidate

    def run(self, context: ToolContext):
        wt_sequence = str(context.artifacts["protein_sequence"])
        hosts = context.artifacts["host_organisms"]
        if isinstance(hosts, str):
            hosts = [hosts]
        if not isinstance(hosts, (list, tuple)) or not hosts:
            raise ValueError("host_organisms must be a non-empty string array")
        request = resolve_codon_optimization_request(
            "Please perform codon optimization for " + " and ".join(map(str, hosts))
        )
        selected = [self._candidate(value, wt_sequence) for value in context.artifacts["selected_candidates"]]
        result = write_codon_optimization_outputs(
            run_dir=context.run_dir,
            sequence_name=str(context.config.get("sequence_name", "enzyme")),
            wt_sequence=wt_sequence,
            selected=selected,
            resolved_request=request,
            cache_dir=context.config.get("runtime", {}).get("cache_dir"),
            kazusa_timeout=float(context.config.get("runtime", {}).get("kazusa_timeout", 10.0)),
        )
        optimized = []
        if result.get("csv"):
            with Path(result["csv"]).open("r", encoding="utf-8", newline="") as handle:
                optimized = list(csv.DictReader(handle))
        return {"optimized_cds_set": optimized, "codon_optimization_result": result}


def register_configured_tools(registry, config: dict) -> list[str]:
    """Register every configured built-in provider without coupling the planner to it."""
    factories = {"phenom": PhenomTool, "ephod": EpHodTool, "unistab": UniStabTool}
    registry.register(BalanceScoresTool())
    registered = ['balance_scores']
    if 'mutation' in config:
        registry.register(VESMLibraryTool(config))
        registered.append('vesm')
    for name, model_config in config.get("models", {}).items():
        factory = factories.get(name)
        if factory is not None and model_config.get("enabled", True):
            registry.register(factory(model_config))
            registered.append(name)
    if config.get("tools", {}).get("codon_optimizer", {}).get("enabled", True):
        registry.register(CodonOptimizationTool())
        registered.append("codon_optimizer")
    return registered


BALANCE_TOOL_SCHEMA = {'type': 'function', 'function': {
    'name': 'balance_scores', 'description': 'Apply user preferences and evidence-informed reliability controls. Every value must be in [0,1]; preference weights must sum to 1. Does not advance phases.',
    'parameters': {'type':'object','additionalProperties':False,'required':['weights','confidence'],
        'properties':{name:{'type':'object','additionalProperties':False,'required':['activity','ph','stability'],
            'properties':{key:{'type':'number','minimum':0,'maximum':1}
                          for key in ('activity','ph','stability')}} for name in ('weights','confidence')}}}}


def execute_sequence_tool(name: str, arguments: dict, sequence: str, config: dict) -> dict:
    """Shared execution for Assistant native calls and the explicit Tools form."""
    import json,math,os
    from ..core.sequence import normalize_sequence
    if normalize_sequence(sequence) != sequence or not 1 <= len(sequence) <= 1000:
        raise ValueError('Provide 1–1000 canonical amino-acid residues')
    if name == 'prepare_mature_protein':
        start,end=arguments.get('start'),arguments.get('end')
        if set(arguments)!={'start','end'} or type(start) is not int or type(end) is not int or not 1<=start<=end<=len(sequence):
            raise ValueError('Provide a valid inclusive 1-based residue interval')
        return {'prepared_sequence':{'start':start,'end':end,'sequence':sequence[start-1:end],
                'source':'user_specified_region','active_run_unchanged':True}}
    if arguments:
        raise ValueError('Structure tools accept no arguments')
    if name=='inspect_structure':
        from Bio.PDB import MMCIFParser, PDBParser
        from Bio.SeqUtils import seq1
        path = Path(config['mutation']['structure_path'])
        parser = MMCIFParser(QUIET=True) if path.suffix.lower() in {'.cif','.mmcif'} else PDBParser(QUIET=True)
        model = parser.get_structure('provided',path)[0]
        chain = model[config['mutation']['chain_id']]
        residues = [r for r in chain if r.id[0]==' ']
        if ''.join(seq1(r.resname) for r in residues) != sequence:
            raise ValueError('The supplied structure does not map exactly to this run sequence')
        points = [{'residue':i,'x':float(r['CA'].coord[0]),'y':float(r['CA'].coord[1]),'z':float(r['CA'].coord[2])}
                  for i,r in enumerate(residues,1) if 'CA' in r]
        from Bio.PDB import PDBIO
        from io import StringIO
        stream=StringIO(); writer=PDBIO();writer.set_structure(chain);writer.save(stream)
        structure = {'source':'provided WT backbone','sequence':sequence,'points':points,'pdb':stream.getvalue()}
        return {'structure':structure,'source':'provided WT backbone','resolved_ca_atoms':len(points),'prediction_run':False}
    if name=='predict_structure':
        if not os.environ.get('BIOHUB_API_TOKEN'):
            raise ValueError('ESMFold2 requires the administrator BIOHUB_API_TOKEN')
        import urllib.request
        request = urllib.request.Request('https://biohub.ai/api/v1/fold',
            data=json.dumps({'model':'esmfold2-fast-2026-05','sequence':sequence,'msa':None}).encode(),
            headers={'Authorization':'Bearer '+os.environ['BIOHUB_API_TOKEN'],'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=90) as stream:
            data=json.load(stream)
        data=data.get('data',data)
        coordinates=data['coordinates']
        if len(coordinates)!=len(sequence):
            raise ValueError('ESMFold2 coordinate count does not match the sequence')
        points=[]
        import math
        for i,atoms in enumerate(coordinates,1):
            ca=atoms[1]  # Official atom37 order: N, CA, C, CB, O, ...
            if ca is not None and len(ca)==3 and all(type(v) in (int,float) and math.isfinite(v) for v in ca):
                points.append({'residue':i,'x':ca[0],'y':ca[1],'z':ca[2]})
        if not points:
            raise ValueError('ESMFold2 returned no finite CA coordinates')
        structure={'source':'Biohub ESMFold2 prediction','sequence':sequence,'points':points}
        return {'structure':structure,'source':'Biohub ESMFold2 prediction','resolved_ca_atoms':len(points),'prediction_run':True}
    raise ValueError('Unknown sequence tool')


def query_esm_atlas(query: str, *, similar: bool = False) -> dict:
    """Read official Atlas Alpha records; lookup never triggers fold-on-miss."""
    import hashlib,json,re,urllib.request,urllib.error
    from urllib.parse import urlencode,quote
    base='https://biohub.ai/esm/protein/api/v1alpha1'
    def request(path,params=None):
        url=base+path+('?' + urlencode(params) if params else '')
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Enzora/1.0','Accept':'application/json'})
            with urllib.request.urlopen(req,timeout=35) as stream:
                raw=stream.read(8*1024*1024+1)
            if len(raw)>8*1024*1024:
                raise ValueError('Atlas response exceeds the single-protein limit')
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            if exc.code==404:
                raise ValueError('No matching stored ESM Atlas record was found. Try a UniProt accession or similarity search.') from None
            raise RuntimeError(f'ESM Atlas HTTP {exc.code}') from None
    if not isinstance(query,str) or not query.strip() or len(query)>2048:
        raise ValueError('Use a UniProt accession, Atlas MD5 hash or protein sequence up to 2048 residues')
    query=query.strip()
    extra={}
    is_sequence=len(query)>20 and set(query)<=set('ACDEFGHIKLMNPQRSTVWY')
    if similar:
        if not is_sequence:
            raise ValueError('Atlas similarity search requires a protein sequence')
        data=request('/similarity-search',{'sequence':query,'topk_results':5,'topk_features':5,'include_cluster_info':'true'})
        return {'source':'ESM Atlas Alpha','query_kind':'SAE feature similarity','query_sequence':query,
            'similar_proteins':data['similar_proteins'],'source_url':'https://biohub.ai/esm/protein/atlas',
            'interpretation':'SAE feature cosine similarity is not sequence identity or experimentally validated function.'}
    if re.fullmatch('[0-9a-fA-F]{32}',query):
        protein_hash=query.lower()
    elif is_sequence:
        protein_hash=hashlib.md5(query.encode()).hexdigest()
    elif re.fullmatch('[A-Za-z0-9_-]{3,32}',query):
        extra=request('/uniprot/'+quote(query,safe=''))
        protein_hash=extra['protein_hash']
    else:
        raise ValueError('Enter an exact UniProt accession, Atlas hash or canonical sequence')
    if not isinstance(protein_hash,str) or not re.fullmatch('[0-9a-f]{32}',protein_hash):
        raise ValueError('Atlas did not return a valid protein hash')
    protein=request('/proteins/'+protein_hash,{'topk_features':5,'fold_on_miss':'false'})
    sequence=protein['sequence']
    if hashlib.md5(sequence.encode()).hexdigest()!=protein_hash:
        raise ValueError('Atlas sequence identity does not match its protein hash')
    fields=('protein_hash','accession','source','sequence','sequence_length','protein_name','gene','organism','function',
            'ptm','mean_plddt','residues_plddt','sae_features','cluster_rep_protein_hash','folded_on_demand')
    metadata={key:protein[key] if key in protein else extra[key] for key in fields if key in protein or key in extra}
    value={'source':'ESM Atlas Alpha','metadata':metadata,'source_url':base+'/proteins/'+protein_hash+'?fold_on_miss=false',
           'lookup_url':base+'/proteins/'+protein_hash+'?fold_on_miss=false','fold_on_miss':False}
    if protein.get('pdb'):
        from Bio.PDB import PDBParser
        from io import StringIO
        import math
        pdb=protein['pdb']
        structure=PDBParser(QUIET=True).get_structure('atlas',StringIO(pdb))
        points=[{'residue':r.id[1],'x':float(r['CA'].coord[0]),'y':float(r['CA'].coord[1]),'z':float(r['CA'].coord[2])}
                for r in structure[0].get_residues() if 'CA' in r and all(math.isfinite(float(x)) for x in r['CA'].coord)]
        value['structure']={'source':'ESM Atlas stored predicted structure','sequence':sequence,'pdb':pdb,'points':points}
    return value
