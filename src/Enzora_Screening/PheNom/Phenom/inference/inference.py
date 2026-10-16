from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_DIR = SCRIPT_DIR.parent
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))

from phenome_normal_prediction import format_prediction, prepare_inference_frame
from models.net import net
from utils.dataloader import DataLoader
from utils.dataset import ProteinMoleculeDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PHENOME enzyme predictions.")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("trained_weights/PHENOME"))
    parser.add_argument("--protein-cache-dir", type=Path, required=True)
    parser.add_argument("--ligand-cache-dir", type=Path, required=True)
    parser.add_argument("--task", choices=["kcat", "km", "kcat_km", "ph", "esp", "mutation", "all"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_folder.mkdir(parents=True, exist_ok=True)
    args.ligand_cache_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.input_file)
    frame = prepare_inference_frame(source, args.task)
    model = _load_model(args.model_dir, args.device)
    dataset = ProteinMoleculeDataset(
        frame,
        molecule_folder=args.ligand_cache_dir,
        protein_folder=args.protein_cache_dir,
        protein_embedding_dim=model.enzgfm_adapter.in_features,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    columns: dict[str, list] = {}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            primary = batch["primary"].to(args.device)
            mu, sigma, _, _, _ = model(primary, protein_only=batch["protein_only"])
            wt_mu = wt_sigma = None
            if batch["reference"] is not None:
                reference = batch["reference"].to(args.device)
                wt_mu, wt_sigma, _, _, _ = model(reference, protein_only=batch["protein_only"])
            batch_columns = format_prediction(args.task, mu, sigma, wt_mu, wt_sigma)
            for name, values in batch_columns.items():
                columns.setdefault(name, []).extend(values.tolist())

    result = source.copy()
    for name, values in columns.items():
        result[name] = values
    output = args.output_folder / "inference_results.csv"
    result.to_csv(output, index=False)
    print(f"Saved PHENOME predictions to {output}")


def _load_model(model_dir: Path, device: str) -> net:
    with (model_dir / "config.json").open() as handle:
        config = json.load(handle)
    degree = torch.load(model_dir / "degree.pt", map_location=device)
    state = torch.load(model_dir / "model.pt", map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    params = config["params"]
    params["prot_evo_channels"] = state["enzgfm_adapter.weight"].shape[1]
    model = net(
        degree["ligand_deg"], degree["protein_deg"],
        **{key: params[key] for key in (
            "mol_in_channels", "mol_edge_channels", "clique_pe_walk_length", "prot_in_channels",
            "prot_evo_channels", "hidden_channels", "pre_layers", "post_layers", "aggregators", "scalers",
            "total_layer", "K", "heads", "dropout", "dropout_attn_score",
        )},
        device=device,
    ).to(device)
    model.load_state_dict(state)
    return model


if __name__ == "__main__":
    main()
