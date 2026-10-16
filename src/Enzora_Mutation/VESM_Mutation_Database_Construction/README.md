# VESM enzyme mutation-library construction

This directory contains a single-sequence inference and library-construction
pipeline for ProteinDPO, EnzGFM, and VESM3. Model weights are intentionally
configured as external paths so the code can be checked locally and run on a
GPU server later.

## Scoring and scale alignment

For every eligible residue `i` and non-WT amino acid `a`, each model
produces the raw log-likelihood ratio

```text
raw_llr(model, i, a) = log P_model(a at i) - log P_model(WT at i)
```

ProteinDPO is structure-conditioned and autoregressive; EnzGFM and VESM3 are
sequence models. Their raw log-probability scales are therefore not directly
exchangeable. The script independently maps each model's complete eligible
non-WT LLR distribution through its empirical CDF and the inverse standard
normal CDF. This monotonic, unbounded transformation preserves each model's
ranking while placing all three on the same reference distribution.

VESM aggregation is performed only across models for the identical mutation:

```text
ensemble(i, a) = min(
    gaussian_llr_ProteinDPO(i, a),
    gaussian_llr_EnzGFM(i, a),
    gaussian_llr_VESM3(i, a),
)
```

No minimum is ever taken across different residues or amino acids. Higher
ensemble scores are then preferred during weighted library sampling. Raw
canonical-AA logits, raw LLRs, calibrated scores, and the ensemble matrix are
all retained in the output NPZ file.

## Mutation space and library design

The currently released `vesm_enzyme_mutation_library_full_length` was generated
over all 544 sequence positions, **without** a pLDDT exclusion. It contains
10,336 single substitutions and a 1,000,000-sequence total allocation:

- 10,336 single mutants
- 339,664 double mutants
- 300,000 triple mutants
- 200,000 quadruple mutants
- 100,000 quintuple mutants
- 50,000 sextuple mutants

The separately run PHENOM-Agent three-round single-point follow-up applies
`pLDDT > 70` to this table at candidate selection time. Structure model 0 has
470 qualifying positions, or 8,930 qualifying single substitutions. This
filter does **not** retroactively describe the full-length VESM library.

The WT is excluded, mutation positions cannot repeat within a sequence, and all
mutation sets are unique. Multi-site mutants are reproducibly sampled from the
configured top score fraction using softmax weights. Because additive
single-mutation LLRs do not model epistasis, high-order designs require
independent downstream evaluation.

## Configuration and execution

`vesm_enzyme_data_config.yaml` is now the authoritative source for mutation
scope, per-order counts and output directory; validation no longer silently
overwrites these values. The checked-in configuration reproduces the released
full-length million-sequence library. To construct a pLDDT-filtered library,
set `mutation_space.scope: plddt`, update the single-mutant quota to 8,930,
choose the desired higher-order quotas and use a distinct output directory.
ProteinDPO needs both the vanilla ESM-IF1 architecture checkpoint and its DPO
fine-tuned state dict.

Validate sequence/structure alignment and full-length counts without weights:

```bash
python vesm_mutation_database_construct.py --validate-only
```

Run all models and construct the library:

```bash
python vesm_mutation_database_construct.py \
  --config vesm_enzyme_data_config.yaml
```

The configured output directory receives:

- `vesm_enzyme_mutate_database.fasta`
- `library_manifest.csv`
- `single_mutation_scores.csv`
- `model_logits_llrs_and_calibrated_scores.npz`
- `mutation_count_distribution.jpg`
- `library_llr_distribution.jpg`
- `run_metadata.json`

## Current Enzora request interface (2026-10-15)

The runtime template has moved to `Enzora/config/mutation.yaml`. Default allocation is now `{1: 1000}`. The historical million-sequence allocation documented above is a previous release example, not an enforced quota.

`library.counts` accepts positive integer mutation orders and quotas, including mixed orders. Singles use the requested number of highest-ranked eligible substitutions; multi-site sequences use the existing weighted unique sampling. Quotas are never reduced silently. Position restrictions apply in full-sequence mode as well as pLDDT mode. A structure/sequence mismatch, impossible quota, empty mutation space or nonfinite model score is an error.

The agent writes a per-run mutation YAML with absolute resource paths and explicit candidate restrictions. Generation outputs keep ensemble scores separate from PHENOM activity predictions. Live progress reports model completion and the number of sequence records actually flushed to disk. Generation weights are supplied separately on the server, never downloaded by the agent's local checks.
