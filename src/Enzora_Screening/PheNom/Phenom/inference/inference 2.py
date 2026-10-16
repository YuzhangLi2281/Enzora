import argparse
import os
import torch
import copy
import pandas as pd
import json
from models.net import net
from utils.dataset import ProteinMoleculeDataset
from utils import protein_init, protein_init_with_keys
from utils.utils import virtual_screening, DataLoader
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to output folder')
    parser.add_argument('--model_dir', type=str, default='trained_weights/PSICHIC_plus', help='Directory with config.json, model.pt, degree.pt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_interpret', action='store_true', help='Save interpretability outputs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    device = args.device
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Load config and model
    with open(os.path.join(args.model_dir, 'config.json')) as f:
        config = json.load(f)
    degree_dict = torch.load(os.path.join(args.model_dir, 'degree.pt'))
    mol_deg, clique_deg, prot_deg = degree_dict['ligand_deg'], degree_dict['clique_deg'], degree_dict['protein_deg']

    model = net(
        mol_deg, prot_deg,
        mol_in_channels=config['params']['mol_in_channels'],
        mol_edge_channels=config['params']['mol_edge_channels'],
        clique_pe_walk_length=config['params']['clique_pe_walk_length'],
        prot_in_channels=config['params']['prot_in_channels'],
        prot_evo_channels=config['params']['prot_evo_channels'],
        hidden_channels=config['params']['hidden_channels'],
        pre_layers=config['params']['pre_layers'],
        post_layers=config['params']['post_layers'],
        aggregators=config['params']['aggregators'],
        scalers=config['params']['scalers'],
        total_layer=config['params']['total_layer'],
        K=config['params']['K'],
        heads=config['params']['heads'],
        dropout=config['params']['dropout'],
        dropout_attn_score=config['params']['dropout_attn_score'],
        device=device
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pt'), map_location=device), strict=False)
    model.eval()

    # Read input CSV
    df = pd.read_csv(args.input_file)

    # Ensure ID column is present and filled
    if 'ID' in df.columns:
        id_series = df['ID'].astype(str)
        id_series = id_series.replace(['', 'nan', 'NaN', 'None'], pd.NA)
        default_ids = "Row_" + (df.index + 1).astype(str)
        ids = id_series.fillna(default_ids).tolist()
    else:
        ids = "Row_" + (df.index + 1).astype(str)
        ids = ids.tolist()
    df['ID'] = ids

    # Prepare protein_dict and ligand_path
    if 'Protein_Sequence' in df.columns:
        all_proteins = df['Protein'].tolist()
        all_seqs = df['Protein_Sequence'].tolist()

        protein_seqs = dict(zip(all_proteins, all_seqs))

        all_proteins = []
        all_seqs = []
        for protein, seq in protein_seqs.items():
            all_proteins.append(protein)
            all_seqs.append(seq)
    else:
        all_proteins = df['Protein'].unique().tolist()
        all_seqs = copy.deepcopy(all_proteins)
    protein_dict = protein_init_with_keys(all_proteins, all_seqs)


    ligand_path = os.path.join(args.output_folder, 'ligands')
    if not os.path.exists(ligand_path):
        os.makedirs(ligand_path)

    # Create dataset and DataLoader
    dataset = ProteinMoleculeDataset(
        df, molecule_dict={}, molecule_folder=ligand_path, protein_dict=protein_dict,
        protein_folder=None, result_path=args.output_folder, dataset_tag='inference', device=device
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                follow_batch=['mol_x', 'clique_x', 'prot_node_aa'],
                                num_workers=8, pin_memory=True) # Added pin_memory


    # Run virtual screening
    result_df = virtual_screening(
        df, model, loader, args.output_folder + "/interpretations",
        save_interpret=args.save_interpret,
        ligand_path=ligand_path,
        protein_dict=protein_dict,
        device=device,
        save_cluster=False
    )

    output_file = os.path.join(args.output_folder, 'inference_results.csv')
    result_df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

    # --- Cleanup: Remove ligands folder if empty ---
    if os.path.isdir(ligand_path):
        shutil.rmtree(ligand_path)

    # --- Cleanup: Remove log file if empty ---
    log_file = os.path.join(args.output_folder, 'inference_dataset_errors.log')
    if os.path.isfile(log_file) and os.path.getsize(log_file) == 0:
        os.remove(log_file)

if __name__ == '__main__':
    main()