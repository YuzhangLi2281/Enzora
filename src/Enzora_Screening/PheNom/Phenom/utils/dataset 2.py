import torch.utils.data
import random
from torch_geometric.data import Dataset
# from torch.utils.data import Dataset
import torch
import pandas as pd
from torch_geometric.data import Data
import pickle
import torch.utils.data
from copy import deepcopy
import numpy as np
import os
from utils.protein_init import *
from mol_utils import smiles_to_graph, process_molecule_file, MoleculeNormalizer
import hashlib
import logging


class ProteinMoleculeDataset(Dataset):
    def __init__(self, sequence_data, molecule_dict={}, molecule_folder='', 
                 protein_dict={}, protein_folder='', result_path='', dataset_tag='', 
                 source_data_column=None, device='cpu', 
                 molecule_error_log='molecule_error_log.txt', standardize=True, timeout=10):
        super(ProteinMoleculeDataset, self).__init__()

        if isinstance(sequence_data,pd.core.frame.DataFrame):
        
        # Ensure 'Ligand' and 'Protein' columns exist
            assert 'Ligand' in sequence_data.columns and 'Protein' in sequence_data.columns, \
                "DataFrame must contain 'Ligand' and 'Protein' columns"

            # Handle ID column
            if 'ID' in sequence_data.columns:
                id_series = sequence_data['ID'].astype(str)
                # Replace empty strings and 'nan' (from astype(str) on NaN) with NaN
                id_series = id_series.replace(['', 'nan', 'NaN', 'None'], pd.NA)
                # Fill missing with Row_{i+1} using vectorized pandas operation
                default_ids = pd.Series("Row_" + (sequence_data.index.astype(int) + 1).astype(str), index=sequence_data.index)
                ids = id_series.fillna(default_ids).tolist()
            else:
                ids = "Row_" + (sequence_data.index.astype(int) + 1).astype(str)
                ids = ids.tolist()
            self.ids = ids

            # Extract Ligand and Protein columns
            ligands = sequence_data['Ligand'].to_numpy()
            proteins = sequence_data['Protein'].to_numpy()
            
            # Extract or create label columns
            reg_labels = sequence_data.get('regression_label', np.full_like(ligands, np.nan, dtype=float))
            cls_labels = sequence_data.get('classification_label', np.full_like(ligands, np.nan, dtype=float))
            # --- Fetch individual functional effect labels ---
            oi_labels = sequence_data.get('orthosteric_inhibitor', np.full_like(ligands, -1.0, dtype=float))
            oa_labels = sequence_data.get('orthosteric_activator', np.full_like(ligands, -1.0, dtype=float))
            ai_labels = sequence_data.get('allosteric_inhibitor', np.full_like(ligands, -1.0, dtype=float))
            aa_labels = sequence_data.get('allosteric_activator', np.full_like(ligands, -1.0, dtype=float))
            
            # --- Combine into a single multi-label array ---
            # Shape will be (num_samples, 4)
            mcls_labels = np.column_stack((oi_labels, oa_labels, ai_labels, aa_labels))

            if source_data_column is not None:
                source_data = sequence_data.get(source_data_column, np.full_like(ligands, np.nan, dtype=float))
            else:
                source_data = np.full_like(ligands, np.nan, dtype=float)

             # --- Store pairs ---
            # Note: We store the mcls_labels array directly here.
            # np.column_stack handles combining arrays of different types by upcasting if needed.
            # We convert to specific types later in __getitem__.
            self.pairs = list(zip(ligands, proteins, reg_labels, cls_labels, source_data, ids)) # Add ids to pairs
            self.mcls_labels = torch.from_numpy(mcls_labels).float()

        else:
            raise Exception('Must be a pandas dataframe..')
        
        self.mol_dict = molecule_dict
        self.mol_folder = molecule_folder
        self.prot_dict = protein_dict
        self.prot_folder = protein_folder
        self.result_path = result_path
        self.dataset_tag = dataset_tag
        self.device = device
        
        
        self.molecule_error_log = molecule_error_log
        
        self.timeout = timeout
        self.standardize = standardize
        self.error_molecule = set()
        # Setup logging 
        self.setup_logging()
    
    def setup_logging(self):
        # Use dataset_tag to create the log file name
        log_file_name = f"{self.dataset_tag}_dataset_errors.log" if self.dataset_tag else 'dataset_errors.log'
        log_file_path = os.path.join(self.result_path, log_file_name)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Set up logging to write to a file
        logging.basicConfig(
            filename=log_file_path,  # File to write logs
            filemode='a',  # Append mode, so logs are not overwritten
            format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
            level=logging.ERROR  # Log level: capture errors and above
        )

    def safe_smiles_filename(self, smiles):
        return hashlib.sha256(smiles.encode()).hexdigest()[:32]

    def mol_graph(self, mol_key):
        ## mol in a dictionary
        if mol_key in self.mol_dict:
            return self.mol_dict[mol_key]
        ## mol in a folder
        else:
            mol_filename = self.safe_smiles_filename(mol_key)
            mol_dict_path = os.path.join(self.mol_folder, f"{mol_filename}.pt")
            ## mol already in the folder then load it
            if os.path.exists(mol_dict_path):
                mol_graph = torch.load(mol_dict_path)
                if mol_graph is None:
                    raise Exception('Molecule is None')
                return mol_graph
            
            ## mol not yet in the folder - process, save it for next time
            else:
                if mol_key in self.error_molecule:
                    raise Exception('Molecule is None as previously failed')
                ## Standard + Normalize, or just Normalize
                normalizer = MoleculeNormalizer(timeout_seconds=self.timeout)
                if self.standardize:
                    result = normalizer.standardize_normalize(mol_key)
                else:
                    result = normalizer.normalize(mol_key)
                norm_smi = result.smiles
                status = result.status
                error_message = result.error_message
                
                # catch error if the SMILES is None
                if norm_smi is None:
                    self.error_molecule.add(mol_key)

                    with open(self.molecule_error_log, 'a') as f:
                        f.write(f"||||| {mol_key} ||||| with status (failed to normalize because [{status}]); and error message - {error_message}\n")
                    raise Exception('Molecule SMILES is None')
                ## Graph Construction
                mol_graph = smiles_to_graph(norm_smi)
                # catch error if the graph is None
                if mol_graph is None:
                    self.error_molecule.add(mol_key)

                    with open(self.molecule_error_log, 'a') as f:
                        f.write(f"||||| {mol_key} ||||| with status (successful normalization but fail at graph construction); and error message - None\n")
                    raise Exception('Molecule is None')
                
                ## Save graph
                torch.save(mol_graph, mol_dict_path)

                return mol_graph
                
    
    def prot_graph(self, prot_key):
        if prot_key in self.prot_dict:
            return self.prot_dict[prot_key]

        prot_dict_path = os.path.join(self.prot_folder,f"{prot_key}.pt")
        if os.path.exists(prot_dict_path):
            prot = torch.load(prot_dict_path)
            return prot
        else:
            raise Exception('cannot find in dictionary or folder.')
        

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return self.__len__()

    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx):
        try:
            # Extract data
            mol_key, prot_key, reg_y, cls_y, source_y, id_val = self.pairs[idx]
            mcls_y = self.mcls_labels[idx].view(1,-1)
            
            reg_y = torch.tensor(reg_y if not pd.isna(reg_y) else float('nan')).float()
            cls_y = torch.tensor(cls_y if not pd.isna(cls_y) else float('nan')).float()
            
            mol = self.mol_graph(str(mol_key))
            prot = self.prot_graph(str(prot_key))

            # Create MultiGraphData for positive pair
            pl_pair = self.create_multigraph_data(mol, prot, reg_y, cls_y, mcls_y, str(mol_key), str(prot_key), str(source_y))
            pl_pair.id = id_val  # Attach ID to the data object

            return pl_pair

        except Exception as e:
            # Log the error with mol_key
            logging.error(f"Error processing mol_key: ||| {mol_key} |||, error: {str(e)}")
            return None 


    def create_multigraph_data(self, mol, prot, reg_y, cls_y, mcls_y, mol_key, prot_key, source_y):
        mol_x = mol['atom_idx']
        mol_x_pe = mol['ligand_pe'] if 'ligand_pe' in mol else None
        mol_x_feat = mol['atom_feature']
        mol_edge_index = mol['atom_edge_index']
        mol_edge_attr = mol['atom_edge_attr']
        mol_num_nodes = mol['atom_num_nodes']

        mol_x_clique = mol['x_clique']
        clique_x_pe = mol['clique_pe'] if 'clique_pe' in mol else None
        clique_num_nodes = mol['clique_num_nodes']
        clique_edge_index = mol['tree_edge_index']
        atom2clique_index = mol['atom2clique_index']

        prot_seq = prot['seq']
        prot_node_pe = prot['protein_pe'] if 'protein_pe' in prot else None
        prot_node_aa = prot['seq_feat'].float()
        prot_node_evo = prot['token_representation'].float()
        prot_num_nodes = prot['num_nodes']
        prot_edge_index = prot['edge_index'].long()
        prot_edge_weight = prot['edge_weight'].float()

        return MultiGraphData(
            mol_x=mol_x, mol_x_feat=mol_x_feat, mol_x_pe=mol_x_pe,
            mol_edge_index=mol_edge_index, mol_edge_attr=mol_edge_attr, mol_num_nodes=mol_num_nodes,
            clique_x=mol_x_clique, clique_x_pe=clique_x_pe, clique_edge_index=clique_edge_index, atom2clique_index=atom2clique_index, clique_num_nodes=clique_num_nodes,
            prot_node_aa=prot_node_aa, prot_node_evo=prot_node_evo, prot_node_pe=prot_node_pe,
            prot_seq=prot_seq, prot_edge_index=prot_edge_index, prot_edge_weight=prot_edge_weight, prot_num_nodes=prot_num_nodes,
            reg_y=reg_y, cls_y=cls_y, mcls_y=mcls_y,
            mol_key=mol_key, prot_key=prot_key, 
            source_y=source_y
        )

def maybe_num_nodes(index, num_nodes=None):
    # NOTE(WMF): I find out a problem here, 
    # index.max().item() -> int
    # num_nodes -> tensor
    # need type conversion.
    # return index.max().item() + 1 if num_nodes is None else num_nodes
    return index.max().item() + 1 if num_nodes is None else int(num_nodes)

def get_self_loop_attr(edge_index, edge_attr, num_nodes):
    r"""Returns the edge features or weights of self-loops
    :math:`(i, i)` of every node :math:`i \in \mathcal{V}` in the
    graph given by :attr:`edge_index`. Edge features of missing self-loops not
    present in :attr:`edge_index` will be filled with zeros. If
    :attr:`edge_attr` is not given, it will be the vector of ones.

    .. note::
        This operation is analogous to getting the diagonal elements of the
        dense adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_weight = torch.tensor([0.2, 0.3, 0.5])
        >>> get_self_loop_attr(edge_index, edge_weight)
        tensor([0.5000, 0.0000])

        >>> get_self_loop_attr(edge_index, edge_weight, num_nodes=4)
        tensor([0.5000, 0.0000, 0.0000, 0.0000])
    """
    loop_mask = edge_index[0] == edge_index[1]
    loop_index = edge_index[0][loop_mask]

    if edge_attr is not None:
        loop_attr = edge_attr[loop_mask]
    else:  # A vector of ones:
        loop_attr = torch.ones_like(loop_index, dtype=torch.float)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    full_loop_attr = loop_attr.new_zeros((num_nodes, ) + loop_attr.size()[1:])
    full_loop_attr[loop_index] = loop_attr

    return full_loop_attr



class MultiGraphData(Data):
    def __inc__(self, key, item, *args):
        if key == 'mol_edge_index':
            return self.mol_x.size(0)
        elif key == 'clique_edge_index':
            return self.clique_x.size(0)
        elif key == 'atom2clique_index':
            return torch.tensor([[self.mol_x.size(0)], [self.clique_x.size(0)]])
        elif key == 'prot_edge_index':
            return self.prot_node_aa.size(0)
        elif key == 'prot_struc_edge_index':
            return self.prot_node_aa.size(0)
        elif key == 'm2p_edge_index':
             return torch.tensor([[self.mol_x.size(0)], [self.prot_node_aa.size(0)]])
        # elif key == 'edge_index_p2m':
        #     return torch.tensor([[self.prot_node_s.size(0)],[self.mol_x.size(0)]])
        else:
            return super(MultiGraphData, self).__inc__(key, item, *args)

