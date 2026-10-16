import torch
from torch_geometric.nn import global_add_pool
from torch.nn import Embedding, Linear
from torch_geometric.utils import degree, to_scipy_sparse_matrix, segregate_self_loops
import torch.nn.functional as F
from torch_geometric.utils import scatter
import numpy as np
import scipy.sparse as sp
from models.layers import MLP, AtomEncoder, Drug_PNAConv, Protein_PNAConv, DrugProteinConv, PosLinear, GCNCluster, SAGECluster, SGCluster, APPNPCluster, dropout_edge, LayerNorm
from copy import deepcopy
## for drug pooling
from models.motif_pool import MotifPool
## for cluster
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_dense_batch, dropout_adj, degree, subgraph, softmax, add_remaining_self_loops
from models.protein_pool import dense_mincut_pool , dense_dmon_pool, simplify_pool
from models.enzyme_prediction import EnzymePredictionHead
## for cluster
from torch_geometric.nn.norm import GraphNorm
import torch_geometric
import math


EPS = 1e-15
import math

class net(torch.nn.Module):
    def __init__(self, mol_deg, prot_deg,
                 # MOLECULE
                 mol_in_channels=43,  mol_edge_channels=30,
                 clique_pe_walk_length=20,
                 prot_in_channels=33, prot_evo_channels=1024,
                 hidden_channels=200,
                 pre_layers=2, post_layers=1,
                 aggregators=['mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'linear'],
                 # interaction
                 total_layer=3,                
                 K = [5,10,20],
                 t = 1,
                 # training
                 heads=5, 
                 dropout=0,
                 dropout_attn_score=0.2,
                 drop_residue=0,
                 dropout_cluster_edge=0,
                 gaussian_noise=0,
                 prediction_head_layout='unified',
                 
                 device='cuda:0'):
        super(net, self).__init__()
        self.total_layer = total_layer
        self.hidden_channels = hidden_channels
        self.heads = heads

        if isinstance(K, int):
            K = [K]*total_layer 

        # MOLECULE IN FEAT
        self.atom_type_encoder = Embedding(20, hidden_channels)
        self.atom_feat_encoder = MLP([mol_in_channels, hidden_channels * 2, hidden_channels], out_norm=True) 

        self.clique_encoder = Embedding(4, hidden_channels)

        # PROTEIN IN FEAT
        self.enzgfm_adapter = Linear(prot_evo_channels, 1280, bias=False)
        self.prot_evo = MLP([1280, hidden_channels * 2, hidden_channels], out_norm=True)
        self.prot_aa = MLP([prot_in_channels, hidden_channels * 2, hidden_channels], out_norm=True) 
    
        ### MOLECULE and PROTEIN
        self.mol_convs = torch.nn.ModuleList()
        self.prot_convs = torch.nn.ModuleList()

        self.mol_gn2 = torch.nn.ModuleList()
        self.prot_gn2 = torch.nn.ModuleList()

        self.inter_convs = torch.nn.ModuleList()

        self.num_cluster = K
        self.t = t
        self.cluster = torch.nn.ModuleList()

        self.mol_pools = torch.nn.ModuleList()
        self.prot_norms = torch.nn.ModuleList()

        self.atom_lins = torch.nn.ModuleList() 
        self.residue_lins = torch.nn.ModuleList() 
        
        self.c2a_mlps = torch.nn.ModuleList()
        self.c2r_mlps = torch.nn.ModuleList()

        self.total_layer = total_layer
        self.prot_edge_dim = hidden_channels

        for idx in range(total_layer):
            self.mol_convs.append(Drug_PNAConv(
                mol_deg, hidden_channels,edge_channels=mol_edge_channels,
                pre_layers=pre_layers, post_layers=post_layers,
                aggregators=aggregators,
                scalers=scalers,
                num_towers=heads,
                dropout=dropout
            ))

            self.prot_convs.append(Protein_PNAConv(
                prot_deg, hidden_channels, edge_channels=hidden_channels, # None,
                pre_layers=pre_layers, post_layers=post_layers,
                aggregators=aggregators,
                scalers=scalers,
                num_towers=heads,
                dropout=dropout
            ))

            self.cluster.append(GCNCluster([hidden_channels, hidden_channels*2, self.num_cluster[idx]], in_norm=True))
            self.inter_convs.append(DrugProteinConv(
                atom_channels=hidden_channels,
                residue_channels=hidden_channels,
                heads=heads,
                t=t,
                dropout_attn_score=dropout_attn_score
            ))
            
            self.mol_pools.append(MotifPool(hidden_channels, heads, clique_pe_walk_length,dropout_attn_score))
            self.prot_norms.append(LayerNorm(hidden_channels))
            
            self.atom_lins.append( Linear(hidden_channels, hidden_channels, bias=False) )
            self.residue_lins.append( Linear(hidden_channels, hidden_channels, bias=False) )

            self.c2a_mlps.append(MLP([hidden_channels, hidden_channels * 2, hidden_channels], bias=False))
            self.c2r_mlps.append(MLP([hidden_channels, hidden_channels * 2, hidden_channels], bias=False))

            self.mol_gn2.append(GraphNorm(hidden_channels))
            self.prot_gn2.append(GraphNorm(hidden_channels))

        self.atom_attn_lin = PosLinear(heads * total_layer, 1, bias=False, init_value= 1 / heads) #(heads * total_layer))
        self.residue_attn_lin = PosLinear(heads * total_layer, 1, bias=False, init_value= 1 / heads) #(heads * total_layer))
        
        self.mol_out = MLP([hidden_channels, hidden_channels * 2, hidden_channels], out_norm=True)
        self.prot_out = MLP([hidden_channels, hidden_channels * 2, hidden_channels], out_norm=True)

        self.prediction_head = EnzymePredictionHead(
            hidden_channels * 2, hidden_channels, layout=prediction_head_layout
        )

        self.dropout = dropout
        self.drop_residue = drop_residue
        self.gaussian_noise = gaussian_noise
        self.dropout_cluster_edge = dropout_cluster_edge
        self.device = device


    def reset_parameters(self):
        self.atom_feat_encoder.reset_parameters()
        self.clique_encoder.reset_parameters()
        self.enzgfm_adapter.reset_parameters()
        self.prot_evo.reset_parameters()
        self.prot_aa.reset_parameters()
        
        for idx in range(self.total_layer):
            self.mol_convs[idx].reset_parameters()
            self.prot_convs[idx].reset_parameters()

            self.mol_gn2[idx].reset_parameters()
            self.prot_gn2[idx].reset_parameters()

            self.cluster[idx].reset_parameters()

            self.mol_pools[idx].reset_parameters()
            self.prot_norms[idx].reset_parameters()

            self.inter_convs[idx].reset_parameters()

            self.atom_lins[idx].reset_parameters()
            self.residue_lins[idx].reset_parameters()

            self.c2a_mlps[idx].reset_parameters()
            self.c2r_mlps[idx].reset_parameters()
    
        self.atom_attn_lin.reset_parameters()
        self.residue_attn_lin.reset_parameters()
        self.mol_out.reset_parameters()
        self.prot_out.reset_parameters()

        self.prediction_head.reset_parameters()

    def forward(self, data, save_cluster=False, protein_only=False, all_task_outputs=False):
        # Protein
        residue_x=data.prot_node_aa
        residue_evo_x=self.enzgfm_adapter(data.prot_node_evo)
        residue_edge_index=data.prot_edge_index
        residue_edge_weight=data.prot_edge_weight
        # Mol-Protein Interaction batch
        prot_batch=data.prot_node_aa_batch

        if not protein_only:
            mol_x=data.mol_x
            mol_x_feat=data.mol_x_feat
            bond_x=data.mol_edge_attr
            atom_edge_index=data.mol_edge_index
            clique_x=data.clique_x
            clique_x_pe=data.clique_x_pe
            atom2clique_index=data.atom2clique_index
            mol_batch=data.mol_x_batch
            clique_batch=data.clique_x_batch


        # Init variables        
        residue_edge_attr = _rbf(residue_edge_weight, D_max=1.0, D_count=self.prot_edge_dim, device=self.device)
        # residue_edge_attr = None
        # PROTEIN Featurize
        residue_x = self.prot_aa(residue_x) + self.prot_evo(residue_evo_x)

        if not protein_only:
            atom_x = self.atom_type_encoder(mol_x.view(-1)) + self.atom_feat_encoder(mol_x_feat)
            clique_x = self.clique_encoder(clique_x.view(-1))
        # clique_id = clique_x.squeeze()
        ortho_loss = torch.tensor(0.).to(self.device)
        cluster_loss = torch.tensor(0.).to(self.device)

        clique_scores = []
        residue_scores = []
        layer_s = {}
        # MOLECULE-PROTEIN Layers
        for idx in range(self.total_layer):
            residue_x = self.prot_convs[idx](residue_x, residue_edge_index, residue_edge_attr)

            if not protein_only:
                atom_x = self.mol_convs[idx](atom_x, bond_x, atom_edge_index)
                drug_x, clique_x, clique_score = self.mol_pools[idx](
                    atom_x, clique_x, clique_x_pe, atom2clique_index, clique_batch
                )
                clique_scores.append(clique_score)

                atom_s = self.cluster[idx](atom_x, atom_edge_index)
                atom_hx, atom_mask = to_dense_batch(atom_x, mol_batch)
                atom_s, _ = to_dense_batch(atom_s, mol_batch)
                atom_adj = to_dense_adj(atom_edge_index, mol_batch)
                _, _, _, mol_cl_loss, mol_o_loss = dense_mincut_pool(
                    atom_hx, atom_adj, atom_s, atom_mask
                )

            ## Cluster protein residues
            dropped_residue_edge_index, _ = dropout_edge(residue_edge_index, p=self.dropout_cluster_edge, 
                                                         force_undirected=True,training=self.training)
            s = self.cluster[idx](residue_x, dropped_residue_edge_index)
            residue_hx, residue_mask = to_dense_batch(residue_x, prot_batch)

            if save_cluster:
                layer_s[idx] = s 

            # cluster features
            s, _ = to_dense_batch(s, prot_batch)
            residue_adj = to_dense_adj(residue_edge_index, prot_batch)
            cluster_mask = residue_mask 

            cluster_drop_mask = None
            if self.drop_residue != 0 and self.training:
                _, _, residue_drop_mask = dropout_node(residue_edge_index, self.drop_residue, residue_x.size(0), prot_batch,
                                                       self.training)  # drop residue for regularization
                residue_drop_mask, _ = to_dense_batch(residue_drop_mask.reshape(-1,1), prot_batch) # drop residue for regularization
                residue_drop_mask = residue_drop_mask.squeeze()
                cluster_drop_mask = residue_mask * residue_drop_mask.squeeze()

            s, cluster_x, residue_adj, cl_loss, o_loss = dense_mincut_pool(residue_hx, residue_adj, s, cluster_mask, cluster_drop_mask)

            # spectral_loss += sp_loss
            if not protein_only:
                o_loss = 0.5 * (o_loss + mol_o_loss)
                cl_loss = 0.5 * (cl_loss + mol_cl_loss)
            ortho_loss += o_loss / self.total_layer
            cluster_loss += cl_loss / self.total_layer
            
            cluster_x = self.prot_norms[idx](cluster_x)
            
            batch_size = s.size(0)
            cluster_residue_batch = torch.arange(batch_size).repeat_interleave(self.num_cluster[idx]).to(self.device)
            cluster_x = cluster_x.reshape(batch_size*self.num_cluster[idx], -1)

            if not protein_only:
                p2m_edge_index = torch.stack([
                    torch.arange(batch_size*self.num_cluster[idx]),
                    torch.arange(batch_size).repeat_interleave(self.num_cluster[idx]),
                ]).to(self.device)
                clique_x, cluster_x, inter_attn = self.inter_convs[idx](
                    drug_x, clique_x, clique_batch, cluster_x, p2m_edge_index
                )
                inter_attn = inter_attn[1]
                row, col = atom2clique_index
                atom_x = atom_x + F.relu(
                    self.atom_lins[idx](scatter(clique_x[col], row, dim=0, dim_size=atom_x.size(0), reduce='mean'))
                )
                atom_x = atom_x + self.c2a_mlps[idx](atom_x)

            residue_hx, _ = to_dense_batch(cluster_x, cluster_residue_batch)
            residue_x = residue_x + F.relu( self.residue_lins[idx]((s @ residue_hx)[residue_mask]) ) # cluster -> residue
            residue_x = residue_x + self.c2r_mlps[idx](residue_x)
            
            ## Graph Normalization
            if not protein_only:
                atom_x = self.mol_gn2[idx](atom_x, mol_batch)
            residue_x = self.prot_gn2[idx](residue_x, prot_batch)

            if protein_only:
                residue_scores.append(residue_x.new_zeros((residue_x.size(0), self.heads)))
            else:
                inter_attn, _ = to_dense_batch(inter_attn, cluster_residue_batch)
                residue_scores.append((s @ inter_attn)[residue_mask])


        # Pool based on attn scores
        if protein_only:
            clique_scores = None
            atom_score = None
            mol_batch = None
            clique_batch = None
        else:
            row, col = atom2clique_index
            clique_scores = torch.cat(clique_scores, dim=-1)
            atom_scores = scatter(clique_scores[col], row, dim=0, dim_size=atom_x.size(0), reduce='mean')
            atom_score = softmax(self.atom_attn_lin(atom_scores), mol_batch)
            mol_pool_feat = self.mol_out(global_add_pool(atom_x * atom_score, mol_batch))

        residue_scores = torch.cat(residue_scores, dim=-1)
        residue_score = softmax(self.residue_attn_lin(residue_scores),prot_batch)
        prot_pool_feat = global_add_pool(residue_x * residue_score, prot_batch)
        
        prot_pool_feat = self.prot_out(prot_pool_feat)
        if protein_only:
            mol_pool_feat = prot_pool_feat.new_zeros(prot_pool_feat.shape)
        
        mol_prot_feat = torch.cat([mol_pool_feat, prot_pool_feat], dim=-1)
        
        mu, sigma = self.prediction_head(
            mol_prot_feat, data.task_index, all_task_outputs=all_task_outputs
        )

        attention_dict = {
            'residue_final_score':residue_score,
            'atom_final_score': atom_score,
            'clique_layer_scores':clique_scores,
            'residue_layer_scores':residue_scores,
            'drug_atom_index':mol_batch,
            'drug_clique_index':clique_batch,
            'protein_residue_index':prot_batch, 
            'mol_feature': mol_pool_feat,
            'prot_feature':prot_pool_feat,
            'cluster_s': layer_s,
            'interaction_fingerprint': mol_prot_feat
        }
        
        return mu, sigma, ortho_loss, cluster_loss, attention_dict

    def arctan_transform(self, tensor, median=5.0):
        """
        Apply arctangent normalization to a PyTorch tensor, centering at the specified median.
        
        Parameters:
            tensor (torch.Tensor): The input tensor.
            median (float): The value to center the arctangent transformation.

        Returns:
            torch.Tensor: The normalized tensor with elements transformed to the range [-pi/2, pi/2].
        """
        # Center the data by subtracting the median
        centered_data = tensor - median
        
        # Apply arctangent transformation
        normalized_data = torch.atan(centered_data)

        return normalized_data
    

    def inverse_arctan_transform(self, transformed_tensor, median=5.0):
        """
        Inverse the arctangent normalization to recover original tensor values, assuming it was centered at the specified median.
        
        Parameters:
            transformed_tensor (torch.Tensor): The tensor that has undergone arctangent transformation.
            median (float): The median value used in the original centering before applying arctangent.

        Returns:
            torch.Tensor: The tensor with original values recovered.
        """
        # Apply tangent to the transformed data to get back the centered values
        recovered_centered_data = torch.tan(transformed_tensor)

        # Re-shift the data by adding the median
        original_data = recovered_centered_data + median

        return original_data

    
    def temperature_clamp(self):
        pass
        # with torch.no_grad():
        #     for m in self.cluster:
        #         m.logit_scale.clamp_(0, math.log(100))
    
    def connect_mol_prot(self, mol_batch, prot_batch):
        mol_num_nodes = mol_batch.size(0)
        prot_num_nodes = prot_batch.size(0)
        mol_adj = mol_batch.reshape(-1, 1).repeat(1, prot_num_nodes)
        pro_adj = prot_batch.repeat(mol_num_nodes, 1)

        m2p_edge_index = (mol_adj == pro_adj).nonzero(as_tuple=False).t().contiguous()

        return m2p_edge_index
    
    def freeze_backbone_optimizers(self, finetune_module, weight_decay, learning_rate, betas, eps, amsgrad): ## only for fineTune Pretrain model
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()  
        whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear,  torch.nn.Bilinear)
        blacklist_weight_modules = (LayerNorm, torch.nn.Embedding, GraphNorm, PosLinear, torch.nn.Parameter)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...

                ################## THIS BLOCK TO FREEZE NOT FINE TUNED LAYERS ##################
                if not any([mn.startswith(name) for name in finetune_module]):
                    p.requires_grad = False
                    continue
                else:
                    p.requires_grad = True
                    print(fpn,' will be finetuned')
                ################## THIS BLOCK TO FREEZE NOT FINE TUNED LAYERS ##################                


                if pn.endswith('bias') or pn.endswith('mean_scale') or pn.endswith('logit_scale'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                    # if mn.startswith('cluster'):
                    #     print(mn, 'not decayed!')
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, torch.nn.MultiheadAttention):
                    if fpn.endswith('in_proj_weight'):
                        decay.add(fpn)  # Assuming you decide to include it in the whitelist
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad)

        return optimizer
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, amsgrad):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()  
        whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear, torch.nn.Bilinear)
        blacklist_weight_modules = (LayerNorm, torch.nn.Embedding, GraphNorm, PosLinear, torch.nn.Parameter)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias') or pn.endswith('mean_scale') or pn.endswith('logit_scale'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                    # if mn.startswith('cluster'):
                    #     print(mn, 'not decayed!')
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, torch.nn.MultiheadAttention):
                    if fpn.endswith('in_proj_weight'):
                        decay.add(fpn)  # Assuming you decide to include it in the whitelist
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad)

        return optimizer
    
    # Update the configure_optimizers method
    def configure_modular_optimizers(self, weight_decay, learning_rate, backbone_lrate, betas, eps, amsgrad,
                                head_modules = {'mol_out', 'prot_out', 'mu_out', 'sigma_out', 'mcls_out'}):
        """
        Configures the optimizer with differential learning rates.
        Assigns 'learning_rate' to prediction heads and final layers, 
        and 'backbone_lrate' to all other backbone layers.
        Also separates parameters for weight decay application.
        """
        
        # Define the names of modules considered part of the "head" (high learning rate)
        # Includes output layers and final processing layers
        
        
        # Parameters requiring gradient
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Separate parameters for decay and no_decay based on name and module type
        decay_params = set()
        no_decay_params = set()
        whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear, torch.nn.Bilinear)
        blacklist_weight_modules = (LayerNorm, torch.nn.Embedding, GraphNorm, PosLinear, torch.nn.Parameter)

        for mn, m in self.named_modules():
            # Skip module if it has no parameters requiring gradients
            # Check if any parameter within the module requires grad
            if not any(p.requires_grad for p in m.parameters(recurse=False)):
                 continue
            
            for pn, p in m.named_parameters(recurse=False): # Use recurse=False here
                if not p.requires_grad:
                     continue # Skip parameters that don't require gradients

                fpn = f'{mn}.{pn}' if mn else pn # Full parameter name

                if pn.endswith('bias') or pn.endswith('mean_scale') or pn.endswith('logit_scale'):
                    no_decay_params.add(fpn)
                elif pn.endswith('weight'):
                    if isinstance(m, whitelist_weight_modules):
                        decay_params.add(fpn)
                    elif isinstance(m, blacklist_weight_modules):
                        no_decay_params.add(fpn)
                    elif isinstance(m, torch.nn.MultiheadAttention) and pn == 'in_proj_weight':
                         decay_params.add(fpn)
                    else:
                        # Default behavior for unclassified weights (e.g. Conv layers if not explicitly handled)
                        # Decide whether to decay them or not. Decaying is common.
                        # print(f"Warning: Weight {fpn} of type {type(m)} not explicitly classified for decay. Adding to decay set.")
                        decay_params.add(fpn) 
                # Add other parameter types if needed (e.g., PosLinear scale/bias if they exist)
                elif isinstance(m, PosLinear): # Example: Handle PosLinear parameters
                     no_decay_params.add(fpn)
                # Add other specific parameter handling here if necessary

        # Validate parameter separation
        inter_params = decay_params & no_decay_params
        union_params = decay_params | no_decay_params
        assert len(inter_params) == 0, f"Parameters {inter_params} in both decay/no_decay sets!"
        
        # Check only against parameters requiring gradients
        missing_params = param_dict.keys() - union_params
        if len(missing_params) > 0:
             print(f"Warning: Parameters {missing_params} not assigned decay status!")
             # Decide how to handle missing: often add to no_decay
             for p_name in missing_params: no_decay_params.add(p_name)
             union_params = decay_params | no_decay_params # Update union after adding missing
             assert len(param_dict.keys() - union_params) == 0 # Final check

        # Create optimizer groups based on LR and decay status
        optim_groups = [
            {'params': [], 'lr': learning_rate, 'weight_decay': weight_decay, 'group_name': 'head_decay'},
            {'params': [], 'lr': learning_rate, 'weight_decay': 0.0, 'group_name': 'head_no_decay'},
            {'params': [], 'lr': backbone_lrate, 'weight_decay': weight_decay, 'group_name': 'backbone_decay'},
            {'params': [], 'lr': backbone_lrate, 'weight_decay': 0.0, 'group_name': 'backbone_no_decay'},
        ]

        assigned_params = set()
        # Assign parameters to the correct group
        for mn, m in self.named_modules():
            is_head_module = mn in head_modules
            
            for pn, p in m.named_parameters(recurse=False):
                 if not p.requires_grad:
                     continue

                 fpn = f'{mn}.{pn}' if mn else pn
                 if fpn in assigned_params: continue # Avoid double assignment from recursion

                 if is_head_module:
                     if fpn in decay_params:
                         optim_groups[0]['params'].append(p)
                         # print(f"Assigning {fpn} to head_decay")
                     elif fpn in no_decay_params:
                         optim_groups[1]['params'].append(p)
                         # print(f"Assigning {fpn} to head_no_decay")
                     else: # Should not happen based on validation above
                          print(f"Error: Head param {fpn} not in decay/no_decay sets!")
                 else: # Backbone module
                     if fpn in decay_params:
                         optim_groups[2]['params'].append(p)
                         # print(f"Assigning {fpn} to backbone_decay")
                     elif fpn in no_decay_params:
                         optim_groups[3]['params'].append(p)
                         # print(f"Assigning {fpn} to backbone_no_decay")
                     else: # Should not happen
                          print(f"Error: Backbone param {fpn} not in decay/no_decay sets!")
                 assigned_params.add(fpn)
        
        # Filter out empty groups
        final_optim_groups = [group for group in optim_groups if group['params']]

        print("--- Optimizer Groups ---")
        total_params = 0
        for group in final_optim_groups:
            group_params = sum(p.numel() for p in group['params'])
            total_params += group_params
            print(f"Group: {group['group_name']}, Params: {group_params:,}, LR: {group['lr']}, WD: {group['weight_decay']}")
        print(f"Total Trainable Params in Optimizer: {total_params:,}")
        print("------------------------")

        optimizer = torch.optim.AdamW(final_optim_groups, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad)
        # Note: The base 'lr' passed to AdamW here is just a default; the group LRs override it.

        return optimizer
        
def _rbf(D, D_min=0., D_max=1., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D = torch.where(D < D_max, D, torch.tensor(D_max).float().to(device) )
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


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



def unbatch_edge_index(edge_index, batch):
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.

    :rtype: :class:`List[Tensor]`

    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
        ...                            [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]])
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        >>> unbatch_edge_index(edge_index, batch)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]))
    """
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)


def compute_connectivity(edge_index, batch):  ## for numerical stability (i.e. we cap inv_con at 100)

    edges_by_batch = unbatch_edge_index(edge_index, batch)

    nodes_counts = torch.unique(batch, return_counts=True)[1]

    connectivity = torch.tensor([nodes_in_largest_graph(e, n) for e, n in zip(edges_by_batch, nodes_counts)])
    isolation = torch.tensor([isolated_nodes(e, n) for e, n in zip(edges_by_batch, nodes_counts)])

    return connectivity, isolation


def nodes_in_largest_graph(edge_index, num_nodes):
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

    num_components, component = sp.csgraph.connected_components(adj)

    _, count = np.unique(component, return_counts=True)
    subset = np.in1d(component, count.argsort()[-1:])

    return subset.sum() / num_nodes


def isolated_nodes(edge_index, num_nodes):
    r"""Find isolate nodes """
    edge_attr = None

    out = segregate_self_loops(edge_index, edge_attr)
    edge_index, edge_attr, loop_edge_index, loop_edge_attr = out

    mask = torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)
    mask[edge_index.view(-1)] = 0

    return mask.sum() / num_nodes

def dropout_node(edge_index, p, num_nodes, batch, training):
    r"""Randomly drops nodes from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    indicating which edges were retained. (3) the node mask indicating
    which nodes were retained.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor`, :class:`BoolTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask, node_mask = dropout_node(edge_index)
        >>> edge_index
        tensor([[0, 1],
                [1, 0]])
        >>> edge_mask
        tensor([ True,  True, False, False, False, False])
        >>> node_mask
        tensor([ True,  True, False, False])
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask
    
    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p
    
    ## ensure no graph is totally dropped out
    batch_tf = global_add_pool(node_mask.view(-1,1),batch).flatten()
    unbatched_node_mask = unbatch(node_mask, batch)
    node_mask_list = []
    
    for true_false, sub_node_mask in zip(batch_tf, unbatched_node_mask):
        if true_false.item():
            node_mask_list.append(sub_node_mask)
        else:
            perm = torch.randperm(sub_node_mask.size(0))
            idx = perm[:1]
            sub_node_mask[idx] = True
            node_mask_list.append(sub_node_mask)
            
    node_mask = torch.cat(node_mask_list)
    
    edge_index, _, edge_mask = subgraph(node_mask, edge_index,
                                        num_nodes=num_nodes,
                                        return_edge_mask=True)
    return edge_index, edge_mask, node_mask
