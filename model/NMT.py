import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
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
        self.psi = SConv(input_features=cfg.SPLINE_CNN.input_features, output_features=cfg.Matching_TF.d_model)
        # self.mlp = MLPQuery(cfg.Matching_TF.d_model, 1024, cfg.Matching_TF.d_model, batch_norm=cfg.Matching_TF.batch_norm)
        
        self.vgg_to_node_dim = nn.Linear(cfg.SPLINE_CNN.input_features, cfg.Matching_TF.d_model)
        self.glob_to_node_dim = nn.Linear(512, cfg.Matching_TF.d_model)

        self.s_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
        self.t_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
        self.cls_enc = nn.Parameter(torch.randn(cfg.Matching_TF.d_model))
        # self.scaled_mlp = MLP_scaled(cfg.Matching_TF.d_model*2, cfg.Matching_TF.d_model//2, cfg.Matching_TF.d_model)      
        
        self.pos_encoding = Pointwise2DPositionalEncoding(cfg.Matching_TF.d_model, 256, 256).cuda()

        
        nGPT_decoder_config = ModelConfig()
        nGPT_decoder_config.dim = cfg.Matching_TF.d_model
        nGPT_decoder_config.num_layers = cfg.Matching_TF.n_decoder
        nGPT_decoder_config.num_heads = cfg.Matching_TF.n_head # number of heads in the multi-head attention mechanism
        nGPT_decoder_config.mlp_hidden_mult = cfg.Matching_TF.nGPT_mlp_hidden_mult
        
        self.n_gpt_decoder = NGPT_DECODER(nGPT_decoder_config)
        self.n_gpt_decoder_2 = NGPT_DECODER(nGPT_decoder_config)
        
        self.w_cosine = PairwiseCosineSimilarity(cfg.Matching_TF.d_model)
        
        self.global_state_dim = 1024
        
    
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
        # for visualisation purposes only
        graph_list = []
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
            h = self.psi(graph)

            h_res = h + vgg_features
                            
            (h_res, mask) = to_dense_batch(h_res, graph.batch, fill_value=0)

            if cfg.Matching_TF.pos_encoding:
                h_res = h_res + self.pos_encoding(p)
                
            global_feature = self.final_layers(edges)[0].reshape((nodes.shape[0], -1))
            global_feature = self.glob_to_node_dim(global_feature)
            global_feature = global_feature + self.cls_enc
            global_feature = global_feature.unsqueeze(1).expand(-1,1, -1)
            
            h_res = torch.cat([global_feature, h_res], dim=1)

            global_feature_mask = torch.tensor([True]).unsqueeze(0).expand(h_res.size(0), -1).to(global_feature.device)
            mask = torch.cat([global_feature_mask, mask], dim=1)


            orig_graph_list.append((h_res,mask))

        h_s, s_mask = orig_graph_list[0]
        h_t, t_mask = orig_graph_list[1]

        assert h_s.size(0) == h_t.size(0), 'batch-sizes are not equal'
        
        batch_size, seq_len, _ = h_s.shape
        padding_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool).to(h_s.device)
        if in_training is True:
            for idx, e in enumerate(n_points_sample):
                h_s[idx, e+1:, :] = 0
                h_t[idx, e+1:, :] = 0
                
                padding_mask[idx, :, e+1:] = 1
                padding_mask[idx, e+1:, :] = 1
            
        hs_dec_output, layer_losses1 = self.n_gpt_decoder(source_nodes = h_s, padding_mask=padding_mask, encoder_output=h_t)
        ht_dec_output, layer_losses2 = self.n_gpt_decoder_2(source_nodes = h_t, padding_mask=padding_mask, encoder_output=h_s)
        
        layer_loss = (layer_losses1 + layer_losses2) / 2
        
        hs_dec_output = hs_dec_output[:, 1:, :]
        ht_dec_output = ht_dec_output[:, 1:, :]
        
        sim_score = self.w_cosine(hs_dec_output, ht_dec_output) #self.w_cosine(hs_dec_output, h_t_norm)
        
        return sim_score, hs_dec_output, ht_dec_output, layer_loss

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
