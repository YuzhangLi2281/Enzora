from torch_geometric.data import Dataset
from torch_geometric.data import Data
from copy import deepcopy
from torch_geometric.loader import DataLoader


class DictDataset(Dataset):
    def __init__(self, entity_dict, entity_type='ligand'):
        super(DictDataset, self).__init__()
        self.entity_dict = entity_dict
        self.entity_keys = list(entity_dict.keys())
        self.entity_type = entity_type
        if entity_type == 'ligand':
            self.node_key = 'atom_idx'
            self.num_node_key = 'atom_num_nodes'
            self.edge_key = 'atom_edge_index'
            self.edge_feat_key = 'atom_edge_attr'

        elif entity_type == 'protein':
            self.node_key = 'token_representation'
            self.num_node_key = 'num_nodes'
            self.edge_key = 'edge_index'
            self.edge_feat_key = 'edge_weight'


        elif entity_type == 'clique':
            self.node_key = 'x_clique'
            self.num_node_key = 'clique_num_nodes'
            self.edge_key = 'tree_edge_index'
            self.edge_feat_key = None

        else:
            raise Exception("Entity type not implemented!")
    
    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return self.__len__()
    
    def __len__(self):
        return len(self.entity_keys)


    def __getitem__(self, idx):
        key = self.entity_keys[idx]
        entity = self.entity_dict[key]
        x = entity[self.node_key]
        num_nodes = entity[self.num_node_key]
        edge_index = entity[self.edge_key]
        
        edge_attr = entity[self.edge_feat_key] if self.edge_feat_key else None

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, key=key)
        

def create_dict_loader(entity_dict, entity_type, batch_size):
    dict_dataset = DictDataset(entity_dict, entity_type)

    dict_loader = DataLoader(dict_dataset, batch_size=batch_size, shuffle=False)

    return dict_loader



