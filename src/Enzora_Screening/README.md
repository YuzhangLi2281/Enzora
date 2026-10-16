# Screening providers

Exactly three scorers are exposed by the agent configuration:

- `phenom`: `PheNom/inference/inference.py`; ligand-conditioned predicted log10 kcat, nominal sigma_log10 and kcat in s^-1. ESMC protein graphs and MolT5 ligand features are prepared by the existing scripts in `PheNom/train` in the configured subprocess environment.
- `ephod`: `EpHod/ephod/run.py`; predicted optimum pH. Weights are read from the explicitly configured local directory.
- `unistab`: `UniStab/src/inference.py`; predicted ddG with the configured ranking direction. Every requested candidate must receive a finite result. Missing/failed batches are errors, never skipped predictions.

PHENOM retains the released checkpoint architecture and tensor schema: 1152 ESMC channels, 33 protein input channels, 61 ligand atom channels, 30 ligand edge channels, task outputs kcat/Km. Its internal `enzgfm_adapter` name is a checkpoint state key, not an EnzGFM activity model. No training-table compatibility layer was added; training data and the 92-field table were not changed.

Generation-model code and configuration belong under Enzora_Mutation and config/mutation.yaml. No EnzGFM activity scorer is registered here.

See ../../config/server.yaml (from Enzora root: config/server.yaml) and the root README for local-resource requirements. Real model execution has not been validated until the server resources are transferred.
