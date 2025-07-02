import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from scipy.optimize import linear_sum_assignment

import utils.backbone
from model.sconv_archs import SConv
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
    mlp_hidden_mult: float = 4
    layer_loss_param: float = 0.3

class NMT(utils.backbone.VGG16_bn):
    def __init__(self):
        super(NMT, self).__init__()
        self.model_name = 'Transformer'
        self.psi_2d = SConv(input_features=cfg.SPLINE_CNN.input_features, 
                            output_features=cfg.Matching_TF.d_model,
                            num_layers=2,
                            dim = 2,
                            kernel_size=5,
                            aggr="max")
        
        # simple MLP to learn virtual coordinate z
        self.mlp_z = nn.Sequential(
            nn.Linear(cfg.Matching_TF.d_model, cfg.Matching_TF.d_model // 2),
            nn.SiLU(),
            nn.Linear(cfg.Matching_TF.d_model//2, 1)
        )
        
        self.psi_3d = SConv(input_features=cfg.Matching_TF.d_model, 
                            output_features=cfg.Matching_TF.d_model,
                            num_layers=2,
                            dim = 3,
                            kernel_size=5,
                            aggr="max")
        
        self.vgg_to_node_dim = nn.Linear(cfg.SPLINE_CNN.input_features, cfg.Matching_TF.d_model)
        self.glob_to_node_dim = nn.Linear(512, cfg.Matching_TF.d_model)

        self.s_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
        self.t_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
        self.cls_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))     
        
        self.pos_encoding = Pointwise2DPositionalEncoding(cfg.Matching_TF.d_model, 256, 256).cuda()

        
        nGPT_decoder_config = ModelConfig()
        nGPT_decoder_config.dim = cfg.Matching_TF.d_model
        nGPT_decoder_config.num_layers = cfg.Matching_TF.n_decoder
        nGPT_decoder_config.num_heads = cfg.Matching_TF.n_head # number of heads in the multi-head attention mechanism
        nGPT_decoder_config.mlp_hidden_mult = cfg.Matching_TF.nGPT_mlp_hidden_mult
        
        self.n_gpt_decoder = NGPT_DECODER(nGPT_decoder_config)
        self.n_gpt_decoder_2 = NGPT_DECODER(nGPT_decoder_config)
        
        self.global_state_dim = 1024
        self.d_hidden = 256
        
        # alpha for gated costs
        self.gatedMLP = MLP(cfg.Matching_TF.d_model, self.d_hidden, cfg.Matching_TF.d_model)
        
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
        
        for layer in self.n_gpt_decoder_2.layers:
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
        graph_list = []
        global_feats = []
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
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
            h = self.psi_2d(graph)

            h_res = h + vgg_features
                            
            (h_res, mask) = to_dense_batch(h_res, graph.batch, fill_value=0)

            if cfg.Matching_TF.pos_encoding:
                h_res = h_res + self.pos_encoding(p)
                
            global_feature = self.final_layers(edges)[0].reshape((nodes.shape[0], -1))
            global_feature = self.glob_to_node_dim(global_feature)
            global_feature = global_feature + self.cls_enc
            global_feature = global_feature.unsqueeze(1).expand(-1,1, -1)
            global_feats.append(global_feature)
            
            
            h_res = torch.cat([global_feature, h_res], dim=1)

            global_feature_mask = torch.tensor([True]).unsqueeze(0).expand(h_res.size(0), -1).to(global_feature.device)
            mask = torch.cat([global_feature_mask, mask], dim=1)

            orig_graph_list.append((h_res,mask))

        # pad all h_res to same seq-length M, expand mask to [B, M, M]
        lengths = [h.shape[1] for h,_ in orig_graph_list]
        M = max(lengths)
        
        padded, padded_mask = [], []
        for (h, mask), n_pts in zip(orig_graph_list, n_points):
            D = h.shape[-1]
            # pad features
            pad_feat = h.new_zeros((batch_size, M - h.size(1), D))
            h_p = torch.cat([h, pad_feat], dim=1)      # [B, M, D]

            # pad valid mask
            pad_mask = torch.zeros((batch_size, M - mask.size(1)), dtype=torch.bool, device=mask.device)
            valid_p = torch.cat([mask, pad_mask], dim=1)  # [B, M]

            padded.append(h_p), padded_mask.append(valid_p)
            
        total_layer_loss = 0.0
        embeddings = []
        for i in range(K):
            # source = graph i
            src_h  = padded[i]       # [B, M, D]
            src_pm = padded_mask[i]    # [B, M, M]

            # build memory features & valid mask for all j != i
            other_h   = [padded[j]   for j in range(K) if j!=i]
            H_mem     = torch.cat(other_h, dim=1)  # [B, (K-1)M, D]
            other_val = [padded_mask[j]  for j in range(K) if j!=i]
            mem_valid = torch.cat(other_val, dim=1) # [B, (K-1)M]

            # rectangular cross-attention ban-mask [B, M, (K-1)M]
            src_inv   = ~src_pm             # [B, M]
            mem_inv   = ~mem_valid              # [B, (K-1)M]
            cross_mask= src_inv.unsqueeze(2) | mem_inv.unsqueeze(1)

            out_i, loss_i = self.n_gpt_decoder(
                source_nodes=src_h,
                padding_mask=cross_mask,
                encoder_output=H_mem
            )
            total_layer_loss += loss_i         # accumulate

            # compute all sims between out_i and every out_j
            feat_i = out_i[:, 1:, :]     # drop any CLS token at position 0
            embeddings.append(feat_i)        # save the [B, M, D]
        
        sims = []
        for i in range(K):
            feat_i = embeddings[i]
            for j in range(i+1, K):
                feat_j = embeddings[j]
                sim = self.w_cosine(feat_i, feat_j)
                sims.append(sim)
        
        avg_layer_loss = total_layer_loss / K
        
        return sims, embeddings, avg_layer_loss

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

class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) module with optional gating and dropout.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output features.
        device (str or torch.device): Device to run the module on.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        device = None
    ):
        super().__init__()
        self.device = (('cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')
                        if device is None else device)

        # the up, down, and gate projections
        self.Wup = nn.Linear(input_dim, hidden_dim, bias=False, device=self.device)
        self.Wgate = nn.Linear(input_dim, hidden_dim, bias=False, device=self.device)
        self.Wdown = nn.Linear(hidden_dim, output_dim, bias=False, device=self.device)

        # this flag designates Wdown to have a different parameter initialization as defined in model.py
        self.Wdown.GPT_scale_init = 1

        # the learnable scaling factors
        self.s_u = Scale(hidden_dim, device=device)
        self.s_v = Scale(hidden_dim, device=device)

        # the varaince-controlling scaling term, needed to benefit from SiLU (see appendix A.1)
        self.scale = math.sqrt(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim).
        """
        # our up & gate projections
        u = self.Wup(x) # (batch_size, seq_len, hidden_dim)
        v = self.Wgate(x)
        # scale them
        u = u * self.s_u()
        v = v * self.s_v() * self.scale 
        # now perform the nonlinearity gate
        hidden = u * F.silu(v) # (batch_size, seq_len, hidden_dim)
        return self.Wdown(hidden) # (batch_size, seq_len, output_dim)