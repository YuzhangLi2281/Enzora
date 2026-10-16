import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax

from torch_geometric.utils import scatter
from torch_geometric.nn import global_add_pool
from models.layers import MLP, dropout_node, LayerNorm
from torch_geometric.utils import to_dense_batch

class MotifPool(torch.nn.Module):
    def __init__(self, hidden_channels, heads, pe_walk_length=20, dropout_attn_score=0): 
        super().__init__()
        assert hidden_channels % heads == 0 

        self.pe_lin = torch.nn.Linear(pe_walk_length, hidden_channels, bias=False)
        self.pe_norm = LayerNorm(hidden_channels)

        self.atom_proj = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.atom_norm = LayerNorm(hidden_channels)

        self.attn = torch.nn.MultiheadAttention(
                hidden_channels,
                heads,
                batch_first=True
            )
        self.clique_norm = LayerNorm(hidden_channels)
        mh_hidden_channels = hidden_channels // heads
        
        self.score_proj = torch.nn.ModuleList()
        for _ in range(heads): 
            self.score_proj.append( MLP([ mh_hidden_channels, mh_hidden_channels*2, 1]) )
        
        self.heads = heads
        self.mh_hidden_channels = mh_hidden_channels
        self.dropout_attn_score = dropout_attn_score

    def reset_parameters(self):
        
        self.pe_lin.reset_parameters()
        self.pe_norm.reset_parameters()
        self.atom_proj.reset_parameters()
        self.atom_norm.reset_parameters()
        self.attn._reset_parameters()
        self.clique_norm.reset_parameters()
        for m in self.score_proj:
            m.reset_parameters()

    def forward(self, x, clique_x, clique_pe, atom2clique_index, clique_batch):
        row, col = atom2clique_index
        H = self.heads
        C = self.mh_hidden_channels

        clique_pe = self.pe_norm(self.pe_lin(clique_pe))
        clique_hx = scatter(x[row], col, dim=0, dim_size=clique_x.size(0), reduce='mean')
        clique_hx = self.atom_norm(self.atom_proj(clique_hx))
        clique_x = clique_x + clique_pe + clique_hx

        ## self attention
        h, mask = to_dense_batch(clique_x, clique_batch)
        h, _ = self.attn(h, h, h, key_padding_mask=~mask,
                             need_weights=False)
        h = h[mask]
        clique_x = clique_x + h
        clique_x = self.clique_norm(clique_x)
        ## GNN scoring
        score_clique = clique_x.view(-1, H, C)
        score = torch.cat([ mlp(score_clique[:, i]) for i, mlp in enumerate(self.score_proj) ], dim=-1)
        score = F.dropout(score, p=self.dropout_attn_score, training=self.training)
        alpha = softmax(score, clique_batch)    
        
        ## multihead aggregation of drug feature
        drug_feat = clique_x.view(-1, H, C) * alpha.view(-1, H, 1) 
        drug_feat = drug_feat.view(-1, H * C)
        drug_feat = global_add_pool(drug_feat, clique_batch)

        return drug_feat, clique_x, alpha
