import torch
from rdkit.Chem import AllChem, RemoveAllHs
from rdkit import Chem
from tqdm import tqdm
import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import math
import os
import hashlib
import threading
import logging

from .mol_to_graph import from_rdmol
from .randomwalk_pe import add_rw_pe
from .tree_decomposition import tree_decomposition

# def normalize_smiles(smiles):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         mol = Chem.RemoveHs(mol, sanitize=True)
#         mol = AllChem.RemoveAllHs(mol)

#         normalized_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
#         if normalized_smiles:
#             return normalized_smiles
#         else:
#             return None
#     except:
#         return None
    


def smiles_to_graph(smiles, one_hot=True):
    graph_dict = {}

    mol = Chem.MolFromSmiles(smiles)
    atom_ids, atomic_nums, atom_feat, edge_index, edge_attr = from_rdmol(mol, one_hot)
    
    ## tree decomposition
    tree_edge_index, atom2clique_index, num_cliques, x_clique = tree_decomposition(mol,return_vocab=True)
    ## if weird compounds => each assign the separate cluster
    if atom2clique_index.nelement() == 0:
        num_cliques = len(mol.GetAtoms())
        x_clique = torch.tensor([3]*num_cliques)
        atom2clique_index = torch.stack([torch.arange(num_cliques),
                                            torch.arange(num_cliques)])

    graph_dict['smiles'] = smiles
    graph_dict['atom_idx'] =  atom_ids
    graph_dict['atom_types'] =  atomic_nums
    graph_dict['atom_feature'] =  atom_feat
    graph_dict['atom_edge_index'] = edge_index
    graph_dict['atom_edge_attr'] = edge_attr
    graph_dict['atom_num_nodes'] = atom_ids.shape[0]
    

    graph_dict['tree_edge_index'] = tree_edge_index.long()
    graph_dict['atom2clique_index'] = atom2clique_index.long()
    graph_dict['x_clique'] = x_clique.long().view(-1, 1)
    graph_dict['clique_num_nodes'] = num_cliques
    if num_cliques == 1:
        graph_dict['clique_pe'] = torch.zeros(1, 20).float()
    else:
        graph_dict['clique_pe'] = add_rw_pe(graph_dict['tree_edge_index'], num_cliques, edge_weight=None, walk_length=20, device = 'cpu')

    
    return graph_dict


def safe_filename(smiles):
    return hashlib.sha256(smiles.encode()).hexdigest()[:32]

def process_molecule_file(smiles, ligand_path):
    # try:
    mol_filename = safe_filename(smiles)
    mol_dict_path = os.path.join(ligand_path, f"{mol_filename}.pt")
    
    if not os.path.exists(mol_dict_path):
        mol_graph = smiles_to_graph(smiles)
        if mol_graph is None:
            raise Exception('Molecule is None')
        torch.save(mol_graph, mol_dict_path)
    else:
        mol_graph = torch.load(mol_dict_path)

    return mol_graph

def process_molecule_file_safe(smiles, ligand_path):
    try:
        return process_molecule_file(smiles, ligand_path)
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {str(e)}")
        return None

def preprocess_ligands_parallel(smiles_list, ligand_path, log_file_path, max_workers=None):
    # Configure logging with a specified file path
    logging.basicConfig(filename=log_file_path, level=logging.ERROR, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    def run_in_background():
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_molecule_file_safe, smiles, ligand_path): smiles for smiles in smiles_list}
            for future in futures:
                try:
                    future.result()  # This will raise an exception if the function failed
                except Exception as e:
                    # Log the error with the SMILES string
                    logging.error(f"Error processing SMILES {futures[future]}: {str(e)}")

        return futures

    # Start the function in a new thread
    thread = threading.Thread(target=run_in_background)
    thread.start()

    # Optionally, you can return the thread object if you want to join it later
    return thread

def check_preprocessing_status(futures):
    completed = 0
    errors = 0
    for future in as_completed(futures):
        smiles = futures[future]
        try:
            result = future.result()
            if result is not None:
                completed += 1
            else:
                errors += 1
        except Exception as e:
            print(f"Unexpected error for SMILES {smiles}: {str(e)}")
            errors += 1
        
        if (completed + errors) % 100 == 0:  # Print status every 100 processed
            print(f"Progress: {completed} completed, {errors} errors, {len(futures) - completed - errors} remaining")

    print(f"Final status: {completed} completed, {errors} errors")
    return completed, errors

def ligand_init(smiles_list, one_hot=True):
    ligand_dict = {}
    for smiles in tqdm(smiles_list):
        graph = smiles_to_graph(smiles, one_hot)
        if graph:
            ligand_dict[smiles] = graph

    return ligand_dict


def ligand_init_with_existing(smiles_list, ligand_dict, one_hot=True):

    for smiles in tqdm(smiles_list):
        if smiles in ligand_dict:
            continue
        graph = smiles_to_graph(smiles, one_hot)
        if graph:
            ligand_dict[smiles] = graph

    return ligand_dict

# Top-level function definition
def process_smiles(smiles, one_hot=True):
    graph = smiles_to_graph(smiles, one_hot)
    if graph:
        return (smiles, graph)
    return None

def ligand_init_concurrent(smiles_list, max_workers=None, chunk_size=100):
    ligand_dict = {}

    total_chunks = math.ceil(len(smiles_list) / chunk_size)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, len(smiles_list), chunk_size):
            chunk = smiles_list[i:i+chunk_size]
            futures.extend(executor.submit(process_smiles, smiles) for smiles in chunk)

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                if result:
                    smiles, graph = result
                    ligand_dict[smiles] = graph
            except Exception as e:
                # Handle exception (e.g., log the error)
                print(f"Error processing SMILES: {e}")

    return ligand_dict