# Deployment preparation

2026-10-15: the two-hop SSH connection was verified with read-only checks. No files, model weights or environments were deployed. The proposed home-directory project location does not yet exist. The root volume has about 15 GB free; the available larger mounts are not writable by the current account at their roots. A writable storage location must be established before transferring model resources. Connection endpoints and credentials must remain outside public site files.

Runtime configuration is in `Enzora/config/server.yaml`, `mutation.yaml` and `request.example.yaml`. See the repository README for launch commands and scientific contracts. Configure each scorer's Python environment separately. Do not copy workstation virtual environments or stale node_modules to the GPU server.

Required resources, to be arranged on the server in the next step:

- VESM: ProteinDPO base/fine-tuned checkpoints, generation-only EnzGFM weights/tokenizer, ESM3 base and VESM3 checkpoints, matching enzyme CIF and chain.
- PHENOM: released model.pt/degree.pt/config.json, local ESMC weights and the compatible `esm.models.esmc.EsmcModel` implementation, local MolT5 weights/tokenizer, torch-geometric/RDKit dependencies.
- EpHod: ESM1v model, SVR and RLAT weights under the configured weights directory.
- UniStab: trained checkpoint plus ESMFold / ESM2 and contact-regression weights in its configured local torch hub cache.
- Gemini: private GEMINI_API_KEY environment variable; use an official supported model name in YAML.

After transfer, run path checks, environment import checks, one-sequence real inference per scorer, a small library, then the full 1000-sequence three-phase workflow. The local orchestration tests use isolated model responses and do not establish GPU inference correctness or model quality.

Publish only the built software frontend to its designated software path in the existing Pages repository. Use a public HTTPS backend hostname through the chosen gateway. GitHub Pages does not host Python inference. Do not embed internal server addresses or credentials. Preserve every other wiki page. No publication has been performed in this task stage.
