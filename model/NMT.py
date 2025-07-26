import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from scipy.optimize import linear_sum_assignment

import utils.backbone
from model.sconv_archs import SConv, MGMSplineCNN
from model.positionalEmbedding import Pointwise2DPositionalEncoding
from utils.config import cfg
from utils.feature_align import feature_align
from utils.utils import lexico_iter
from utils.evaluation_metric import make_perm_mat_pred
from utils.visualization import easy_visualize
from model.nGPT_decoder import NGPT_DECODER
from model.nGPT_encoder import NGPT_ENCODER


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)

def cosine_norm(x: torch.Tensor, dim=-1) -> torch.Tensor:
    """
    Places vectors onto the unit-hypersphere

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    # calculate the magnitude of the vectors
    norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min=1e-6)
    # divide by the magnitude to place on the unit hypersphere
    return x / norm

def dense_edge_from_sparse(edge_index, edge_attr,
                           num_nodes, d_edge, fill_value=0., device=None):
    """(Ni,*)  ->  (Ni,Ni,d_e) dense tensor"""
    out = torch.full((num_nodes, num_nodes, d_edge),
                     fill_value, dtype=edge_attr.dtype,
                     device=device or edge_attr.device)
    src, dst = edge_index
    out[src, dst] = edge_attr
    out[dst, src] = edge_attr            # make it symmetric
    return out

class Scale(nn.Module):
    """
    A module that manages learnable scaling parameters to ensure different learning rates
    from the rest of the parameters in the model (see pages 5 and 19)
    
    Args:
        dim (int): Dimension of the scaling parameter
        scale (float): Initial scale value
        init (float): Initial value for the scaling parameter
        device (str, optional): Device to store the parameter on
    """
    def __init__(self, dim: int, heads: int = 1, scale: float = 1.0, init: float = 1.0, device=None):
        super().__init__()
        self.device = (('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
                      if device is None else device)
        self.init = init
        self.scale = scale
        self.s = nn.Parameter(torch.ones(heads, dim, device=self.device) * scale)
            # heads == 1 gives us a single regular vector
            # heads > 1 gets used in attention mechanism for different scaling vector for each head
    
    def forward(self):
        """Compute the effective scaling factor."""
        return self.s * (self.init / self.scale) # shape (heads, dim)

class ModelConfig:
    """
    Design your N-GPT here
    """
    dim: int = 128
    device: str = None
        # defaults to best available GPU/CPU
    num_layers: int = 6
    num_heads: int = 4 # number of heads in the multi-head attention mechanism
    d_edge = int = 128
    mlp_hidden_mult: float = 4
    layer_loss_param: float = 0.3

class NMT(utils.backbone.VGG16_bn):
    def __init__(self):
        super(NMT, self).__init__()
        self.model_name = 'Transformer'
        self.d_edge = cfg.Matching_TF.d_edge
        self.psi_2d = SConv(input_features=cfg.SPLINE_CNN.input_features, 
                            output_features=cfg.Matching_TF.d_model,
                            num_layers=2,
                            dim = 2,
                            kernel_size=5,
                            aggr="max")
        
        # simple MLP to learn virtual coordinate z
        self.mlp_z = nn.Sequential(
            nn.Linear(cfg.Matching_TF.d_model, cfg.Matching_TF.d_model // 2),
            nn.ReLU(),
            nn.Linear(cfg.Matching_TF.d_model//2, 1),
            nn.Sigmoid()
        )
        
        self.psi_3d = MGMSplineCNN(in_channels=cfg.Matching_TF.d_model,
                                   hidden_channels=cfg.Matching_TF.d_model,
                                   out_channels=cfg.Matching_TF.d_model,
                                   dim=3,
                                   num_layers=2,
                                   dropout=0.2,
                                   aggr="max")
        
        self.vgg_to_node_dim = nn.Linear(cfg.SPLINE_CNN.input_features, cfg.Matching_TF.d_model)
        self.glob_to_node_dim = nn.Linear(512, cfg.Matching_TF.d_model)

        self.s_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
        self.t_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
        self.cls_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
        self.cls_enc_edges = nn.Parameter(torch.randn(cfg.Matching_TF.d_edge))      # shape = (d_edge,)     
        
        self.pos_encoding = Pointwise2DPositionalEncoding(cfg.Matching_TF.d_model, 256, 256).cuda()
        
        # add token-type embedding like BERT for graphs embeddings to let decoder know from which graph the attention comes
        self.graph_embed = nn.Embedding(cfg.TRAIN.num_graphs_in_matching_instance, cfg.Matching_TF.d_model) 

        nGPT_decoder_config = ModelConfig()
        nGPT_decoder_config.dim = cfg.Matching_TF.d_model
        nGPT_decoder_config.num_layers = cfg.Matching_TF.n_decoder
        nGPT_decoder_config.num_heads = cfg.Matching_TF.n_head # number of heads in the multi-head attention mechanism
        nGPT_decoder_config.d_edge = cfg.Matching_TF.d_edge
        nGPT_decoder_config.mlp_hidden_mult = cfg.Matching_TF.nGPT_mlp_hidden_mult
        
        self.n_gpt_decoder = NGPT_DECODER(nGPT_decoder_config)
        
        self.w_cosine = PairwiseCosineSimilarity(cfg.Matching_TF.d_model)
        
    
    def normalize_linear(self, module):
        """
        Helper method to normalize Linear layer weights where one dimension matches model dim
        """
        # Find the dimension that matches cfg.dim
        dim_to_normalize = None
        for dim, size in enumerate(module.weight.shape):
            if size == cfg.Matching_TF.d_model:
                dim_to_normalize = dim
                break
        
        if dim_to_normalize is not None:
            # Normalize the weights
            module.weight.data = cosine_norm(module.weight.data, dim=dim_to_normalize)
    
    def enforce_constraints(self):
        """
        Enforces constraints after each optimization step:
        2. Cosine normalization on Linear layer weights where one dimension matches model dim
        """
        # for layer in self.n_gpt_encoder.layers:
        #     layer.alpha_A.s.data.abs_()
        #     layer.alpha_M.s.data.abs_()
            
        for layer in self.n_gpt_decoder.layers:
            layer.alpha_A.s.data.abs_()
            layer.alpha_C.s.data.abs_()
            layer.alpha_G.s.data.abs_()
            layer.alpha_M.s.data.abs_()
        
        # Cosine normalize relevant Linear layers
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                self.normalize_linear(module)
    
    
    
    def update_order(self, source_nodes, input_order):
        B, _, _ = source_nodes.shape
        for b in range(B):
            source_nodes[b, :, :] = source_nodes[b, input_order[b], :]
        return source_nodes
    

    def forward(
        self,
        images,
        points,
        graphs,
        n_points,
        n_points_sample, 
        perm_mats,
        eval_pred_points=None,
        in_training=True,
        input_order=None,
        matched_points_mask=None,
        matched_padding_mask_hs=None,
        matched_padding_mask_ht=None,
    ):
        batch_size = graphs[0].num_graphs
        orig_graph_list = []
        K = len(graphs)
        
        # for visualisation purposes only
        graph_list, edge_list = [], []
        global_feats = []
        for graph_idx,(image, p, n_p, graph) in enumerate(zip(images, points, n_points, graphs)):
            # extract feature
            # with torch.no_grad():
            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)
            
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, p, n_p, (256, 256)), n_p)
            F = concat_features(feature_align(edges, p, n_p, (256, 256)), n_p)

            node_features = torch.cat((U, F), dim=-1)
            
            graph.x = node_features
            # for visualisation purposes only
            graph_list.append(graph.to_data_list())

            # node + edge features from vgg
            vgg_features = self.vgg_to_node_dim(node_features)
            
            # splineCNN spatial features 
            h2d = self.psi_2d(graph)
            
            # predict depth
            z = self.mlp_z(h2d)

            # build 3-D edge_attr
            src, tgt = graph.edge_index                      # [E]
            dz   = 0.5*(z[src] - z[tgt])+0.5     # [E,1]
            edge_attr_3d = torch.cat([graph.edge_attr, dz], dim=1)  # [E,3]

            graph3d         = graph.clone()
            graph3d.edge_attr = edge_attr_3d
            graph3d.x        = h2d                          # use same node feats

            #  3-D SplineCNN & residual 
            gnn3d_out = self.psi_3d(graph3d) 
            h_3d = gnn3d_out.x                      # [Ni, D]
            h_res = h2d + 0.3 * h_3d + vgg_features
                            
            (h_res, mask) = to_dense_batch(h_res, graph.batch, fill_value=0)

            if cfg.Matching_TF.pos_encoding: # set to true for keypoints positional encodings
                h_res = h_res + self.pos_encoding(p)
                # add positional encoding for graphs    
                gid = torch.full((batch_size, h_res.size(1)), graph_idx, dtype=torch.long, device=h_res.device)
                h_res = h_res + self.graph_embed(gid)
            
            global_feature = self.final_layers(edges)[0].reshape((nodes.shape[0], -1))
            global_feature = self.glob_to_node_dim(global_feature)
            global_feature = global_feature + self.cls_enc
            global_feature = global_feature.unsqueeze(1).expand(-1,1, -1)
            global_feats.append(global_feature)
            
            h_res = torch.cat([global_feature, h_res], dim=1)

            global_feature_mask = torch.tensor([True]).unsqueeze(0).expand(h_res.size(0), -1).to(global_feature.device)
            mask = torch.cat([global_feature_mask, mask], dim=1)
            
            e_dense_per_batch = []
            for b in range(batch_size):
                n_i = int(n_points[graph_idx][b])
                
                edge_src = graph.edge_index[0]
                edge_dst = graph.edge_index[1]
                node_mask = (graph.batch == b)
                edge_mask = node_mask[edge_src] & node_mask[edge_dst]

                remap = -torch.ones_like(graph.batch)
                remap[node_mask] = torch.arange(n_i, device=h_res.device)
                ei_local = remap[graph.edge_index[:, edge_mask]]   # (2,E_b)  in 0…Ni‑1
                ea_local = graph.edge_attr[edge_mask]              # (E_b,d_edge)
                
                e_b  = dense_edge_from_sparse(
                           ei_local,   # (2,E_b)
                           ea_local,    # (E_b,d_e)
                           n_i, self.d_edge, 0., device=h_res.device)
                # (Ni,Ni,d_e)  -> pad to M later
                    # CLS→node  and node→CLS are the same learnable vector
                cls_vec = self.cls_enc_edges           # (d_e,)
                # create a (Ni+1,Ni+1,d_e) tensor filled with cls_vec
                e_full = cls_vec.expand(n_i+1, n_i+1, -1).clone()
                # write the real edges into the lower‑right block
                e_full[1:, 1:] = e_b
                # keep symmetry
                e_full[0, 0] = 0.                      # optional: CLS‑to‑CLS zero
                e_dense_per_batch.append(e_full)       # (Ni+1,Ni+1,d_e)
            edge_list.append(e_dense_per_batch) 

            orig_graph_list.append((h_res,mask))

        # pad all h_res to same seq-length M, expand mask to [B, M, M]
        lengths = [h.shape[1] for h,_ in orig_graph_list]
        M = max(lengths)
        
        padded, padded_mask, edge_pad = [], [], []
        for (h, mask), e_list in zip(orig_graph_list, edge_list):
            D = h.shape[-1]
            # pad features
            pad_feat = h.new_zeros((batch_size, M - h.size(1), D))
            h_p = torch.cat([h, pad_feat], dim=1)      # [B, M, D]

            # pad valid mask
            pad_mask = torch.zeros((batch_size, M - mask.size(1)), dtype=torch.bool, device=mask.device)
            valid_p = torch.cat([mask, pad_mask], dim=1)  # [B, M]

            padded.append(h_p), padded_mask.append(valid_p)
            
            e_padded = torch.zeros(batch_size, M, M, self.d_edge, device=h.device)
            for b, e_b in enumerate(e_list):          # e_b is (Ni,Ni,d_e)
                ni = e_b.size(0)
                e_padded[b, :ni, :ni] = e_b
            edge_pad.append(e_padded)                 # keep B‐sized tensor
            
        total_layer_loss = 0.0
        embeddings, edges = [], []
        for i in range(K):
            # source = graph i
            src_h  = padded[i]       # [B, M, D]
            src_pm = padded_mask[i]    # [B, M, M]
            src_e   = edge_pad[i]      # [B,N,N,d_e]

            # memory features & valid mask for all j != i
            other_h   = [padded[j]   for j in range(K) if j!=i]
            H_mem     = torch.cat(other_h, dim=1)  # [B, (K-1)M, D]
            other_val = [padded_mask[j]  for j in range(K) if j!=i]
            mem_valid = torch.cat(other_val, dim=1) # [B, (K-1)M]

            # rectangular cross-attention ban-mask [B, M, (K-1)M]
            src_inv   = ~src_pm             # [B, M]
            mem_inv   = ~mem_valid              # [B, (K-1)M]
            cross_mask= src_inv.unsqueeze(2) | mem_inv.unsqueeze(1) # [B, M, 1] OR [B, 1, (K-1)M]

            out_i, out_e, loss_i = self.n_gpt_decoder(
                source_nodes=src_h,
                source_edges = src_e,    # [B, M, M, d_e]
                padding_mask=cross_mask,
                encoder_output=H_mem
            )
            out_i = out_i * (1.0 / math.sqrt(K)) # scale using K graphs
            total_layer_loss += loss_i        

            feat_i = out_i[:, 1:, :]     
            embeddings.append(feat_i)        # [B, M, D]
            edges.append(out_e)
        
        sims = []
        for i in range(K):
            feat_i = embeddings[i]
            for j in range(i+1, K):
                feat_j = embeddings[j]
                sim = self.w_cosine(feat_i, feat_j)
                sims.append(sim)
        
        avg_layer_loss = total_layer_loss / K
        
        return sims, embeddings, edges, avg_layer_loss

class PairwiseCosineSimilarity(nn.Module):
    def __init__(self, node_feature_dim):
        super(PairwiseCosineSimilarity, self).__init__()
    
    def forward(self, x, y):
        
        y_transposed = y.transpose(-2, -1)  # Shape: [batch_size, node_feature, nodes_y]
        numerator = torch.bmm(x, y_transposed)  # Shape: [batch_size, nodes_x, nodes_y]
        
        x_norm = torch.norm(x, p=2, dim=2).clamp(min=1e-8)  # Shape: [batch_size, nodes_x]
        y_norm = torch.norm(y, p=2, dim=2).clamp(min=1e-8)  # Shape: [batch_size, nodes_y]
        
        denominator = torch.bmm(x_norm.unsqueeze(2), y_norm.unsqueeze(1))  # Shape: [batch_size, nodes_x, nodes_y]
        
        # Compute cosine similarity matrix
        cosine_similarity = numerator / denominator  # Shape: [batch_size, nodes_x, nodes_y]
        
        return cosine_similarity