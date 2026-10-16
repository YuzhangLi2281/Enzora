from torch import Tensor
import torch
from tqdm import tqdm
# from vs_edits.dict_loader import create_dict_loader
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
    to_scipy_sparse_matrix,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)

def add_rw_pe(edge_index, num_nodes, edge_weight=None, walk_length=20, device = 'cpu'):
    assert edge_index is not None
    num_edges = edge_index.size(1)
    edge_index = edge_index.to(device)
    
    row, col = edge_index
    N = num_nodes
    assert N is not None

    if edge_weight is None:
        value = torch.ones(num_edges, device=row.device)
    else:
        value = edge_weight

    value = torch.ones(num_edges, device=device)
    value = scatter(value, row, dim_size=N, reduce='sum').clamp(min=1)[row]
    value = 1.0 / value

    # if N <= 2_000:  # Dense code path for faster computation:
    adj = torch.zeros((N, N), device=device)
    adj[row, col] = value
    loop_index = torch.arange(N, device=device)
    # else:
    #     adj = to_torch_csr_tensor(edge_index, value, size=1)

    def get_pe(out: Tensor) -> Tensor:
        if is_torch_sparse_tensor(out):
            return get_self_loop_attr(*to_edge_index(out), num_nodes=N)
        return out[loop_index, loop_index]

    out = adj
    pe_list = [get_pe(out)]
    for _ in range(walk_length - 1):
        out = out @ adj
        pe_list.append(get_pe(out))

    pe = torch.stack(pe_list, dim=-1)

    return pe



def add_rwpe_to_dict(entity_dict, entity_type, batch_size, device, walk_length=20):
    dict_loader = create_dict_loader(entity_dict,entity_type,batch_size)
    pe_dict = {}

    for data in tqdm(dict_loader):
        entity_keys = data.key
        batch_rw_pe = add_rw_pe(data.edge_index, data.num_nodes, None, walk_length, device)
        batch_rw_pe = batch_rw_pe.cpu()
        unbatched_pe = unbatch_nodes(batch_rw_pe, data.batch)
    
        for k, v in zip(entity_keys, unbatched_pe):
            pe_dict[k] = v
    
    for k, pe in pe_dict.items():
        entity_dict[k][entity_type+'_pe'] = pe

    return entity_dict




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