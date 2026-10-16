import os
import numpy as np
import sys
import hashlib

# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

    
from itertools import repeat
import pandas as pd 
from .dataloader import DataLoader

import torch
from torch_geometric.utils import degree

class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

def create_custom_loader(type='epoch'):
    if type == 'epoch':
        return DataLoader
    elif type =='infinite':
        return InfiniteDataLoader
    else:
        raise Exception('Not Implemented')
        
class CustomWeightedRandomSampler(torch.utils.data.WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

def sampler_from_weights(weights):
    sampler = CustomWeightedRandomSampler(weights, len(weights), replacement=True)
    
    return sampler 

def create_custom_sampler(class_list, specified_weight={}):
    assert isinstance(specified_weight,dict)
    class_list = np.array(class_list)
    class_weight = {
         t: 1./len(np.where(class_list == t)[0]) for t in np.unique(class_list)
    }
    
    samples_weight = np.array([class_weight[t] for t in class_list])
    
    if specified_weight:
        specified_weight = np.array([specified_weight[i] for i in class_list])
        samples_weight *= specified_weight
        
    sampler = CustomWeightedRandomSampler(samples_weight, len(samples_weight))
    
    return sampler 

def compute_pna_degrees(train_loader):
    mol_max_degree = -1
    clique_max_degree = -1
    prot_max_degree = -1
    
    for data in tqdm(train_loader):
        # mol
        mol_d = degree(data.mol_edge_index[1], num_nodes=data.mol_x.shape[0], dtype=torch.long)
        mol_max_degree = max(mol_max_degree, int(mol_d.max()))
        # clique
        try:
            clique_d = degree(data.clique_edge_index[1], num_nodes=data.clique_x.shape[0], dtype=torch.long)
        except RuntimeError:
            print(data.clique_edge_index[1])
            print(data.clique_x)
            print('clique shape',data.clique_x.shape)
            print('atom shape',data.mol_x.shape[0])
            break
        clique_max_degree = max(clique_max_degree, int(clique_d.max()))
        # protein
        prot_d = degree(data.prot_edge_index[1], num_nodes=data.prot_node_aa.shape[0], dtype=torch.long)
        prot_max_degree = max(prot_max_degree, int(prot_d.max()))

    # Compute the in-degree histogram tensor
    mol_deg = torch.zeros(mol_max_degree + 1, dtype=torch.long)
    clique_deg = torch.zeros(clique_max_degree + 1, dtype=torch.long)
    prot_deg = torch.zeros(prot_max_degree + 1, dtype=torch.long)

    for data in tqdm(train_loader):
        # mol
        mol_d = degree(data.mol_edge_index[1], num_nodes=data.mol_x.shape[0], dtype=torch.long)
        mol_deg += torch.bincount(mol_d, minlength=mol_deg.numel())

        # clique
        clique_d = degree(data.clique_edge_index[1], num_nodes=data.clique_x.shape[0], dtype=torch.long)
        clique_deg += torch.bincount(clique_d, minlength=clique_deg.numel())

        # Protein
        prot_d = degree(data.prot_edge_index[1], num_nodes=data.prot_node_aa.shape[0], dtype=torch.long)
        prot_deg += torch.bincount(prot_d, minlength=prot_deg.numel())

    return mol_deg, clique_deg, prot_deg


def unbatch(src, batch, dim: int = 0):
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)

    :rtype: :class:`List[Tensor]`

    Example:

        >>> src = torch.arange(7)
        >>> batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
        >>> unbatch(src, batch)
        (tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


def unbatch_nodes(data_tensor, index_tensor):
    """
    Unbatch a data tensor based on an index tensor.

    Args:
    data_tensor (torch.Tensor): The tensor to be unbatched.
    index_tensor (torch.Tensor): A tensor of the same length as data_tensor's first dimension, 
                                 indicating the batch index for each element in data_tensor.

    Returns:
    list[torch.Tensor]: A list of tensors, where each tensor corresponds to a separate batch.
    """
    return [data_tensor[index_tensor == i] for i in index_tensor.unique()]


def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

def printline(line):
    sys.stdout.write(line + "\x1b[K\r")
    sys.stdout.flush()

def protein_degree_from_path(protein_path):
    protein_max_degree = -1 
    protein_files = os.listdir(protein_path)
    for pfile in protein_files:
        v = torch.load(os.path.join(protein_path, pfile))
        node_num = len(v['seq'])
        edge_index = v['edge_index']
        protein_degree = degree(edge_index[1], num_nodes=node_num, dtype=torch.long)
        protein_max_degree = max(protein_max_degree, protein_degree.max())

    protein_deg = torch.zeros(protein_max_degree + 1, dtype=torch.long)
    for pfile in protein_files:
        v = torch.load(os.path.join(protein_path, pfile))
        node_num = len(v['seq'])
        edge_index = v['edge_index']
        protein_degree = degree(edge_index[1], num_nodes=node_num, dtype=torch.long)
        protein_deg += torch.bincount(protein_degree, minlength=protein_deg.numel())   

    return protein_deg

def protein_degree_from_dict(protein_dict):
    protein_max_degree = -1 
    for k, v in protein_dict.items():
        node_num = len(v['seq'])
        edge_index = v['edge_index']
        protein_degree = degree(edge_index[1], num_nodes=node_num, dtype=torch.long)
        protein_max_degree = max(protein_max_degree, protein_degree.max())

    protein_deg = torch.zeros(protein_max_degree + 1, dtype=torch.long)
    for k, v in protein_dict.items():
        node_num = len(v['seq'])
        edge_index = v['edge_index']
        protein_degree = degree(edge_index[1], num_nodes=node_num, dtype=torch.long)
        protein_deg += torch.bincount(protein_degree, minlength=protein_deg.numel())   

    return protein_deg


def ligand_degree_from_dict(ligand_dict):
    mol_max_degree = -1 
    clique_max_degree = -1 
    
    for k, v in tqdm(ligand_dict.items()):
         # mol
        mol_x = v['atom_feature']
        mol_edge_index = v['atom_edge_index']
        mol_d = degree(mol_edge_index[1], num_nodes=mol_x.shape[0], dtype=torch.long)
        mol_max_degree = max(mol_max_degree, int(mol_d.max()))
        # clique
        clique_x = v['x_clique']
        clique_edge_index = v['tree_edge_index'].long()
        clique_d = degree(clique_edge_index[1], num_nodes=clique_x.shape[0], dtype=torch.long)
        clique_max_degree = max(clique_max_degree, int(clique_d.max()))
    
    mol_deg = torch.zeros(mol_max_degree + 1, dtype=torch.long)
    clique_deg = torch.zeros(clique_max_degree + 1, dtype=torch.long)

    for k, v in tqdm(ligand_dict.items()):
        # mol
        mol_x = v['atom_feature']
        mol_edge_index = v['atom_edge_index']
        mol_d = degree(mol_edge_index[1], num_nodes=mol_x.shape[0], dtype=torch.long)
        
        mol_deg += torch.bincount(mol_d, minlength=mol_deg.numel())
        # clique
        clique_x = v['x_clique']
        clique_edge_index = v['tree_edge_index'].long()
        clique_d = degree(clique_edge_index[1], num_nodes=clique_x.shape[0], dtype=torch.long)
        clique_deg += torch.bincount(clique_d, minlength=clique_deg.numel())
       
    return mol_deg, clique_deg


def minmax_norm(arr):
    return (arr - arr.min())/(arr.max() - arr.min())

def percentile_rank(arr):
    return np.argsort(np.argsort(arr)) / (len(arr)-1) 

from rdkit import Chem
from rdkit.Chem import PropertyPickleOptions
import pickle
        
def store_ligand_score(ligand_smiles, atom_types, atom_scores, ligand_path):

    # Create a molecule from a SMILES string
    mol = Chem.MolFromSmiles(ligand_smiles)
    # Add an atom-level property to the first atom
    for i, atom in enumerate(mol.GetAtoms()):
        
        if atom_types[i] == atom.GetAtomicNum():
            atom.SetProp("PHENOME_Atom_Score", str(atom_scores[i]))
        else:
            return False
    # Configure RDKit to pickle all properties
    Chem.SetDefaultPickleProperties(PropertyPickleOptions.AllProps)

    # Serialize molecule to a pickle file
    with open(ligand_path, 'wb') as f:
        pickle.dump(mol, f)

    return True
    
def safe_smiles_filename(smiles):
    return hashlib.sha256(smiles.encode()).hexdigest()[:32]


def load_ligand_from_path(ligand_path, ligand_key):
    ligand_filename = safe_smiles_filename(ligand_key)
    ligand_file = os.path.join(ligand_path, f"{ligand_filename}.pt")
    if not os.path.exists(ligand_file):
        raise FileNotFoundError(f"Ligand file not found: {ligand_file}")

    ligand_data = torch.load(ligand_file)
    return ligand_data
